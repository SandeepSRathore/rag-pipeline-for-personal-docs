"""Incremental ingestion pipeline.

scan -> diff vs manifest -> for each new/changed doc:
    load -> normalize -> chunk -> injection-scan -> embed (cached) -> upsert -> drop stale
for each deleted doc: delete its points.

Properties:
- Idempotent: running twice in a row does nothing the second time.
- Cheap: unchanged docs are skipped; unchanged chunks of changed docs hit the embedding cache.
- Consistent: new chunks are written BEFORE stale ones are deleted, so a doc never vanishes
  mid-update. If a changed doc fails to load, its previous version keeps serving.
- Serialized: a file lock prevents two concurrent ingests from interleaving.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from filelock import FileLock, Timeout

from ragkit.config import Settings
from ragkit.db import connect
from ragkit.embeddings.base import DenseEmbedder, SparseEmbedder
from ragkit.embeddings.cache import CachedEmbedder, CacheStats, text_hash
from ragkit.embeddings.providers import build_dense, build_sparse
from ragkit.guardrails.injection_detect import detect_injection
from ragkit.ingestion.chunker import chunk_document
from ragkit.ingestion.loaders import load_file
from ragkit.ingestion.manifest import ChangePlan, Manifest, ManifestEntry
from ragkit.ingestion.normalize import normalize_text
from ragkit.ingestion.scanner import SourceFile, scan_directory
from ragkit.observability.logging import get_logger
from ragkit.store.qdrant_store import PointData, QdrantStore

log = get_logger(__name__)
CHUNK_NAMESPACE = uuid.UUID("6f1c0e52-4f4e-4c55-9a51-1b2f3c4d5e6f")


class IngestBusyError(RuntimeError):
    pass


@dataclass
class IngestReport:
    source: str
    plan: dict[str, int]
    dry_run: bool
    collection: str | None = None
    chunks_written: int = 0
    embedded_via_api: int = 0
    embedding_cache_hits: int = 0
    flagged_chunks: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    planned: dict[str, list[str]] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_point_id(doc_id: str, text: str, occurrence: int) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{doc_id}:{text_hash(text)}:{occurrence}"))


class Ingestor:
    def __init__(
        self,
        settings: Settings,
        store: QdrantStore | None = None,
        dense: DenseEmbedder | None = None,
        sparse: SparseEmbedder | None = None,
    ):
        self.settings = settings
        self.store = store or QdrantStore(settings)
        self.dense = CachedEmbedder(dense or build_dense(settings), settings.db_path)
        self.sparse = sparse or build_sparse(settings)
        self.manifest = Manifest(settings.db_path)
        self.lock = FileLock(str(settings.data_dir / "ingest.lock"))

    # ------------------------------------------------------------------ public
    def ingest(
        self,
        root: Path,
        source: str | None = None,
        dry_run: bool = False,
        force: bool = False,
        collection: str | None = None,
    ) -> IngestReport:
        source = source or root.resolve().name
        t0 = time.perf_counter()
        self.dense.stats = CacheStats()
        files, warnings = scan_directory(root, source, self.settings.max_upload_bytes)
        plan = self.manifest.plan(files, source, self.settings.pipeline_version, force=force)
        report = IngestReport(
            source=source, plan=plan.summary(), dry_run=dry_run, warnings=warnings
        )
        report.planned = {
            "new": [f.rel_path for f in plan.new],
            "changed": [f.rel_path for f in plan.changed],
            "deleted": [e.path.split(":", 1)[1] for e in plan.deleted],
        }
        if dry_run:
            return report

        try:
            with self.lock.acquire(timeout=0):
                self._register_source(source, root)
                target = collection or self.store.ensure_active(
                    self.settings.pipeline_version, self.dense.dim
                )
                report.collection = target
                self._apply(plan, target, report)
        except Timeout as e:
            raise IngestBusyError("another ingest is already running") from e

        report.embedded_via_api = self.dense.stats.misses
        report.embedding_cache_hits = self.dense.stats.hits
        report.duration_s = round(time.perf_counter() - t0, 2)
        self.manifest.log_run(report.to_dict())
        log.info("ingest_done", **{k: v for k, v in report.to_dict().items() if k != "planned"})
        return report

    def reindex(self) -> dict:
        """Blue/green rebuild of every registered source into a fresh collection."""
        new_coll = self.store.new_collection_name(self.settings.pipeline_version)
        self.store.create_collection(new_coll, self.dense.dim)
        reports = []
        try:
            for name, root in self._sources():
                if not Path(root).is_dir():
                    log.warning("reindex_source_missing", source=name, root=root)
                    continue
                reports.append(
                    self.ingest(Path(root), source=name, force=True, collection=new_coll).to_dict()
                )
        except Exception:
            self.store.drop_collection(new_coll)
            raise
        old = self.store.swap_alias(new_coll)
        if old and old != new_coll:
            self.store.drop_collection(old)
        return {"new_collection": new_coll, "dropped": old, "sources": reports}

    # ----------------------------------------------------------------- internals
    def _register_source(self, source: str, root: Path) -> None:
        with connect(self.settings.db_path) as conn:
            conn.execute(
                "INSERT INTO sources(name, root) VALUES (?,?)"
                " ON CONFLICT(name) DO UPDATE SET root=excluded.root",
                (source, str(root.resolve())),
            )

    def _sources(self) -> list[tuple[str, str]]:
        with connect(self.settings.db_path) as conn:
            return [(r["name"], r["root"]) for r in conn.execute("SELECT name, root FROM sources")]

    def _apply(self, plan: ChangePlan, collection: str, report: IngestReport) -> None:
        for entry in plan.deleted:
            self.store.delete_doc(collection, entry.doc_id)
            self.manifest.delete(entry.doc_id)
            log.info("doc_deleted", path=entry.path)

        for f in [*plan.new, *plan.changed]:
            previous = self.manifest.get(f.doc_id)
            try:
                written, flagged = self._process(f, collection)
                report.chunks_written += written
                report.flagged_chunks += flagged
            except Exception as e:
                log.error("doc_failed", path=f.rel_path, error=str(e))
                report.failed.append({"path": f.rel_path, "error": str(e)})
                if previous is None or previous.status != "ok":
                    # Never successfully indexed: remove any partial points.
                    self.store.delete_doc(collection, f.doc_id)
                    chunk_ids: list[str] = []
                    content_hash = f.content_hash
                else:
                    # Keep serving the last good version; retry next run.
                    chunk_ids = previous.chunk_ids
                    content_hash = previous.content_hash
                self.manifest.upsert(
                    ManifestEntry(
                        doc_id=f.doc_id,
                        path=f"{f.source}:{f.rel_path}",
                        content_hash=content_hash,
                        chunk_ids=chunk_ids,
                        pipeline_version=self.settings.pipeline_version,
                        status="failed",
                        error=str(e)[:500],
                    )
                )

    def _process(self, f: SourceFile, collection: str) -> tuple[int, int]:
        s = self.settings
        loaded = load_file(f.abs_path, s.max_pdf_pages)
        norm = normalize_text(loaded.text)
        meta = loaded.metadata
        title = normalize_text(meta["title"]).text
        chunks = chunk_document(norm.text, title, s.chunk_tokens, s.chunk_overlap_tokens)
        if not chunks:
            raise ValueError("no extractable text")

        texts = [c.text for c in chunks]
        dense = self.dense.embed_documents(texts)
        sparse = self.sparse.embed_documents(texts)

        seen: dict[str, int] = {}
        points: list[PointData] = []
        flagged_count = 0
        for c, dv, sv in zip(chunks, dense, sparse, strict=True):
            h = text_hash(c.text)
            seen[h] = seen.get(h, -1) + 1
            pid = chunk_point_id(f.doc_id, c.text, seen[h])
            verdict = detect_injection(c.text)
            flagged = verdict.is_injection()
            flagged_count += int(flagged)
            points.append(
                PointData(
                    id=pid,
                    dense=dv,
                    sparse=sv,
                    payload={
                        "doc_id": f.doc_id,
                        "source": f.source,
                        "path": f.rel_path,
                        "title": title,
                        "section": c.section,
                        "chunk_index": c.index,
                        "text": c.text,
                        "acl": meta.get("acl", "public"),
                        "department": meta.get("department", "general"),
                        "effective_date": meta.get("effective_date"),
                        "version": meta.get("version"),
                        "doc_family": meta.get("doc_family"),
                        "flagged": flagged,
                        "flag_reasons": verdict.reasons,
                        "hidden_chars_removed": norm.hidden_chars_removed,
                    },
                )
            )
            if flagged:
                log.warning(
                    "chunk_quarantined", path=f.rel_path, section=c.section, reasons=verdict.reasons
                )

        new_ids = [p.id for p in points]
        self.store.upsert(collection, points)  # 1) write new version
        self.store.delete_stale(collection, f.doc_id, new_ids)  # 2) then drop old chunks
        self.manifest.upsert(
            ManifestEntry(
                doc_id=f.doc_id,
                path=f"{f.source}:{f.rel_path}",
                content_hash=f.content_hash,
                chunk_ids=new_ids,
                pipeline_version=s.pipeline_version,
                status="ok",
                flagged_chunks=flagged_count,
            )
        )
        return len(points), flagged_count

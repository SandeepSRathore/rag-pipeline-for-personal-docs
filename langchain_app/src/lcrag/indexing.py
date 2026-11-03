"""Incremental ingestion with LangChain's indexing API.

`langchain_core.indexing.index()` + `SQLRecordManager` is LangChain's answer to
"don't re-ingest everything". For each chunk it computes an ID from a hash of
content + metadata and records it (with the chunk's `source` as group) in SQLite. On the
next run:
  - chunks whose ID already exists              -> skipped (no embedding, no write)
  - new/edited chunks                           -> embedded + upserted
  - cleanup="scoped_full": chunks of sources SEEN in this run that weren't re-emitted
    (paragraph removed/edited) -> deleted after the new ones are written

What we had to add around it (gaps in LangChain):
  - Deleted FILES: scoped_full only cleans sources it sees, so a removed file's chunks
    would linger. We find sources the record manager knows but disk doesn't, and delete them.
    ("full" cleanup would catch them, but it would also delete the last good version of a
    file that failed to load this run.)
  - Dry run: index() has none. We compute the same chunk IDs ourselves (our own
    `key_encoder`) and compare with the record manager.
  - Qdrant IDs must be UUIDs, so the key encoder returns uuid5(sha256(content+metadata))
    (the default sha1 hex isn't a valid Qdrant ID, and sha1 isn't collision-resistant).
  - Locking, blue/green reindex, per-file error isolation.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout
from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.indexing import index
from qdrant_client import QdrantClient
from sqlalchemy import text

from lcrag.callbacks import BudgetedEmbeddings, Ledger
from lcrag.config import Settings
from lcrag.embeddings import build_base_dense, build_sparse, embedding_dim, with_cache
from lcrag.loaders import LumoraDirectoryLoader
from lcrag.splitting import split_documents
from lcrag.vectorstore import Collections, make_client, vector_store

NAMESPACE = uuid.UUID("0b6c5c8e-1d1e-4a77-9a3a-6f7b0a6f2c11")


def chunk_id(doc: Document) -> str:
    """Deterministic, Qdrant-compatible ID from content + metadata."""
    payload = doc.page_content + "\x00" + json.dumps(doc.metadata, sort_keys=True, default=str)
    return str(uuid.uuid5(NAMESPACE, hashlib.sha256(payload.encode()).hexdigest()))


class IngestBusyError(RuntimeError):
    pass


@dataclass
class IngestReport:
    source: str
    plan: dict[str, int]
    dry_run: bool
    planned: dict[str, list[str]] = field(default_factory=dict)
    collection: str | None = None
    indexing_result: dict[str, int] = field(default_factory=dict)  # LangChain's own counters
    embedded_via_api: int = 0
    flagged_chunks: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class LCIndexer:
    def __init__(
        self,
        settings: Settings,
        client: QdrantClient | None = None,
        base_dense: BudgetedEmbeddings | None = None,
        sparse=None,  # type: ignore[no-untyped-def]
    ):
        self.s = settings
        self.client = client or make_client(settings)
        self.collections = Collections(self.client, settings)
        self.base_dense = base_dense or build_base_dense(settings, Ledger(settings))
        self.dense = with_cache(self.base_dense, settings)
        self.sparse = sparse or build_sparse(settings)
        self.dim = embedding_dim(settings, self.base_dense)
        self.lock = FileLock(str(settings.lc_data_dir / "ingest.lock"))
        self.sources_file = settings.lc_data_dir / "sources.json"

    # -------------------------------------------------------------- record manager
    def record_manager(self, collection: str, source: str) -> SQLRecordManager:
        rm = SQLRecordManager(namespace=f"{collection}/{source}", db_url=self.s.db_url)
        rm.create_schema()
        return rm

    @staticmethod
    def known_sources(rm: SQLRecordManager) -> set[str]:
        """Gap: RecordManager has no 'list groups' API, so we query its table directly."""
        with rm.engine.connect() as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                text("SELECT DISTINCT group_id FROM upsertion_record WHERE namespace = :ns"),
                {"ns": rm.namespace},
            )
            return {r[0] for r in rows if r[0]}

    # -------------------------------------------------------------- public
    def ingest(
        self,
        root: Path,
        source: str | None = None,
        dry_run: bool = False,
        force: bool = False,
        collection: str | None = None,
    ) -> IngestReport:
        t0 = time.perf_counter()
        source = source or root.resolve().name
        loader = LumoraDirectoryLoader(root, self.s.max_upload_bytes)
        on_disk = {rel for rel, _ in loader.list_files()}
        docs = list(loader.lazy_load())  # LangChain Documents, one per file
        chunks = split_documents(docs, self.s.chunk_tokens, self.s.chunk_overlap_tokens)
        failed_paths = {e["path"] for e in loader.errors}

        target = collection or self.collections.target()
        rm = self.record_manager(target, source) if target else None
        known = self.known_sources(rm) if rm else set()

        # ---- plan (also our dry run) ------------------------------------------
        by_src: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            by_src[c.metadata["source"]].append(chunk_id(c))
        planned: dict[str, list[str]] = {"new": [], "changed": [], "unchanged": [], "deleted": []}
        for src in sorted(by_src):
            existing = set(rm.list_keys(group_ids=[src])) if rm and src in known else set()
            if not existing:
                planned["new"].append(src)
            elif force or existing != set(by_src[src]):
                planned["changed"].append(src)
            else:
                planned["unchanged"].append(src)
        planned["deleted"] = sorted(known - on_disk)
        report = IngestReport(
            source=source,
            plan={k: len(v) for k, v in planned.items()},
            dry_run=dry_run,
            planned={k: v for k, v in planned.items() if k != "unchanged"},
            failed=loader.errors,
            warnings=loader.warnings,
            flagged_chunks=sum(1 for c in chunks if c.metadata["flagged"]),
        )
        if dry_run:
            return report

        try:
            with self.lock.acquire(timeout=0):
                self._register_source(source, root)
                target = collection or self.collections.ensure_active(self.dim)
                report.collection = target
                rm = self.record_manager(target, source)
                vs = vector_store(self.client, target, self.dense, self.sparse)
                self.base_dense.calls = 0

                # ---- the LangChain part: one call does diff + upsert + stale cleanup
                result = index(
                    chunks,
                    rm,
                    vs,
                    cleanup="scoped_full",
                    source_id_key="source",
                    key_encoder=chunk_id,
                    force_update=force,
                    batch_size=128,
                )

                # ---- gap: whole files deleted from disk (but not ones that just failed)
                for src in (self.known_sources(rm) - on_disk) - failed_paths:
                    keys = rm.list_keys(group_ids=[src])
                    vs.delete(ids=list[str | int](keys))
                    rm.delete_keys(keys)
                    result["num_deleted"] += len(keys)
        except Timeout as e:
            raise IngestBusyError("another ingest is already running") from e

        report.indexing_result = cast(dict[str, int], dict(result))
        report.embedded_via_api = self.base_dense.calls
        report.duration_s = round(time.perf_counter() - t0, 2)
        return report

    def reindex(self) -> dict:
        """Blue/green: build a fresh collection (fresh record-manager namespace), swap alias."""
        new = self.collections.new_name()
        self.collections.create(new, self.dim)
        try:
            reports = [
                self.ingest(Path(root), source=name, collection=new).to_dict()
                for name, root in self._sources().items()
                if Path(root).is_dir()
            ]
        except Exception:
            self.collections.drop(new)
            raise
        old = self.collections.swap(new)
        if old and old != new:
            for name in self._sources():
                rm = self.record_manager(old, name)
                rm.delete_keys(rm.list_keys())
            self.collections.drop(old)
        return {"new_collection": new, "dropped": old, "sources": reports}

    # -------------------------------------------------------------- sources registry
    def _sources(self) -> dict[str, str]:
        if self.sources_file.exists():
            return json.loads(self.sources_file.read_text())
        return {}

    def _register_source(self, name: str, root: Path) -> None:
        srcs = self._sources()
        srcs[name] = str(root.resolve())
        self.sources_file.write_text(json.dumps(srcs, indent=2))

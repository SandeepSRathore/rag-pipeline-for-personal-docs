"""Ingestion manifest: what we've indexed, so the next run only does the diff."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ragkit.db import connect
from ragkit.ingestion.scanner import SourceFile


@dataclass
class ManifestEntry:
    doc_id: str
    path: str
    content_hash: str
    chunk_ids: list[str]
    pipeline_version: str
    status: str
    flagged_chunks: int = 0
    error: str | None = None


@dataclass
class ChangePlan:
    new: list[SourceFile] = field(default_factory=list)
    changed: list[SourceFile] = field(default_factory=list)
    unchanged: list[SourceFile] = field(default_factory=list)
    deleted: list[ManifestEntry] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "new": len(self.new),
            "changed": len(self.changed),
            "unchanged": len(self.unchanged),
            "deleted": len(self.deleted),
        }


class Manifest:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _row(self, r: sqlite3.Row) -> ManifestEntry:
        return ManifestEntry(
            doc_id=r["doc_id"],
            path=r["path"],
            content_hash=r["content_hash"],
            chunk_ids=json.loads(r["chunk_ids"]),
            pipeline_version=r["pipeline_version"],
            status=r["status"],
            flagged_chunks=r["flagged_chunks"],
            error=r["error"],
        )

    def entries_for_source(self, source: str) -> dict[str, ManifestEntry]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE path LIKE ? ESCAPE '\\'",
                (source.replace("%", r"\%").replace("_", r"\_") + ":%",),
            ).fetchall()
        return {r["doc_id"]: self._row(r) for r in rows}

    def all_entries(self) -> list[ManifestEntry]:
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY path").fetchall()
        return [self._row(r) for r in rows]

    def get(self, doc_id: str) -> ManifestEntry | None:
        with connect(self.db_path) as conn:
            r = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return self._row(r) if r else None

    def upsert(self, e: ManifestEntry) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO documents(doc_id, path, content_hash, chunk_ids, pipeline_version,"
                " status, error, flagged_chunks, ingested_at) VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(doc_id) DO UPDATE SET path=excluded.path,"
                " content_hash=excluded.content_hash, chunk_ids=excluded.chunk_ids,"
                " pipeline_version=excluded.pipeline_version, status=excluded.status,"
                " error=excluded.error, flagged_chunks=excluded.flagged_chunks,"
                " ingested_at=excluded.ingested_at",
                (
                    e.doc_id,
                    e.path,
                    e.content_hash,
                    json.dumps(e.chunk_ids),
                    e.pipeline_version,
                    e.status,
                    e.error,
                    e.flagged_chunks,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def delete(self, doc_id: str) -> None:
        with connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    def log_run(self, summary: dict) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO ingest_runs(started_at, summary) VALUES (?,?)",
                (datetime.now(UTC).isoformat(), json.dumps(summary)),
            )

    def plan(
        self, files: list[SourceFile], source: str, pipeline_version: str, force: bool = False
    ) -> ChangePlan:
        """Diff files on disk against the manifest."""
        known = self.entries_for_source(source)
        plan = ChangePlan()
        seen: set[str] = set()
        for f in files:
            seen.add(f.doc_id)
            e = known.get(f.doc_id)
            if e is None:
                plan.new.append(f)
            elif (
                force
                or e.content_hash != f.content_hash
                or e.pipeline_version != pipeline_version
                or e.status != "ok"
            ):
                plan.changed.append(f)
            else:
                plan.unchanged.append(f)
        plan.deleted = [e for doc_id, e in known.items() if doc_id not in seen]
        return plan

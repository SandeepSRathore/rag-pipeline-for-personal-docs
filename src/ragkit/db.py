"""SQLite: one small local DB for the ingestion manifest, embedding cache and cost ledger.

SQLite is fine for a single-node app. In a multi-replica deployment you'd move these
tables to Postgres (same schema) so every replica sees the same state.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id           TEXT PRIMARY KEY,
    path             TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    chunk_ids        TEXT NOT NULL,          -- JSON list
    pipeline_version TEXT NOT NULL,
    status           TEXT NOT NULL,          -- ok | failed
    error            TEXT,
    flagged_chunks   INTEGER NOT NULL DEFAULT 0,
    ingested_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embedding_cache (
    model      TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    vector     BLOB NOT NULL,               -- float32 bytes
    PRIMARY KEY (model, text_hash)
);
CREATE TABLE IF NOT EXISTS cost_ledger (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    day           TEXT NOT NULL,
    kind          TEXT NOT NULL,             -- llm | embedding
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    usd           REAL NOT NULL,
    request_id    TEXT
);
CREATE INDEX IF NOT EXISTS idx_cost_day ON cost_ledger(day);
CREATE TABLE IF NOT EXISTS sources (
    name  TEXT PRIMARY KEY,
    root  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    summary     TEXT NOT NULL               -- JSON
);
"""


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

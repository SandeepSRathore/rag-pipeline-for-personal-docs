"""Content-addressed embedding cache.

Key = (model, sha256(text)). When one paragraph of a 30-chunk document changes, the
other 29 chunks hit the cache and cost nothing. Switching models never returns stale
vectors, because the model name is part of the key.

Only document chunks are cached. User queries are NOT stored (privacy).
"""

from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass
from pathlib import Path

from ragkit.db import connect
from ragkit.embeddings.base import DenseEmbedder


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class CachedEmbedder:
    def __init__(self, inner: DenseEmbedder, db_path: Path):
        self.inner = inner
        self.db_path = db_path
        self.model_name = inner.model_name
        self.dim = inner.dim
        self.stats = CacheStats()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        hashes = [text_hash(t) for t in texts]
        found: dict[str, list[float]] = {}
        with connect(self.db_path) as conn:
            for i in range(0, len(hashes), 500):
                part = hashes[i : i + 500]
                q = ",".join("?" * len(part))
                for r in conn.execute(
                    f"SELECT text_hash, vector FROM embedding_cache WHERE model=? AND text_hash IN ({q})",  # noqa: S608,E501 - placeholders only
                    (self.model_name, *part),
                ):
                    found[r["text_hash"]] = array("f", r["vector"]).tolist()

        missing_idx = [i for i, h in enumerate(hashes) if h not in found]
        self.stats.hits += len(texts) - len(missing_idx)
        self.stats.misses += len(missing_idx)
        if missing_idx:
            # De-duplicate identical texts inside the batch.
            uniq: dict[str, str] = {hashes[i]: texts[i] for i in missing_idx}
            vecs = self.inner.embed_documents(list(uniq.values()))
            with connect(self.db_path) as conn:
                for h, v in zip(uniq.keys(), vecs, strict=True):
                    found[h] = v
                    conn.execute(
                        "INSERT OR REPLACE INTO embedding_cache(model, text_hash, vector)"
                        " VALUES (?,?,?)",
                        (self.model_name, h, array("f", v).tobytes()),
                    )
        return [found[h] for h in hashes]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)

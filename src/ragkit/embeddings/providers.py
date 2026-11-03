"""Concrete embedders: OpenAI (paid), fastembed local (free), fake (tests)."""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

from ragkit.config import Settings
from ragkit.embeddings.base import DenseEmbedder, SparseEmbedder, SparseVec
from ragkit.observability import cost
from ragkit.observability.logging import get_logger

log = get_logger(__name__)

OPENAI_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class OpenAIEmbedder:
    def __init__(self, settings: Settings, batch_size: int = 128):
        from openai import OpenAI

        if settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not set (or use EMBEDDINGS_PROVIDER=local)")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(), max_retries=3, timeout=30
        )
        self.model_name = settings.openai_embedding_model
        self.dim = OPENAI_DIMS.get(self.model_name, 1536)
        self.batch_size = batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            cost.check_budget(self.settings)
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            cost.record(self.settings, "embedding", self.model_name, resp.usage.prompt_tokens)
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class LocalEmbedder:
    def __init__(self, settings: Settings):
        from fastembed import TextEmbedding

        self.model_name = settings.local_embedding_model
        self.model = TextEmbedding(self.model_name)
        self.dim = len(next(iter(self.model.embed(["dim probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Deterministic hashed bag-of-words. Zero cost, no network; only for tests/offline."""

    def __init__(self, dim: int = 256):
        self.model_name = "fake-hash-embedder"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in _WORD.findall(text.lower()):
            h = int(hashlib.md5(w.encode(), usedforsecurity=False).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class BM25Embedder:
    """BM25 sparse vectors via fastembed (runs locally, free). Qdrant applies IDF."""

    def __init__(self, model_name: str):
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        return [SparseVec(e.indices.tolist(), e.values.tolist()) for e in self.model.embed(texts)]

    def embed_query(self, text: str) -> SparseVec:
        e = next(iter(self.model.query_embed(text)))
        return SparseVec(e.indices.tolist(), e.values.tolist())


def build_dense(settings: Settings) -> DenseEmbedder:
    if settings.embeddings_provider == "openai":
        return OpenAIEmbedder(settings)
    if settings.embeddings_provider == "local":
        return LocalEmbedder(settings)
    return FakeEmbedder()


@lru_cache(maxsize=2)
def _bm25(model_name: str) -> BM25Embedder:
    return BM25Embedder(model_name)


def build_sparse(settings: Settings) -> SparseEmbedder:
    return _bm25(settings.sparse_model)

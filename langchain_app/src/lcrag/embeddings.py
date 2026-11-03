"""Embeddings: LangChain integrations + `CacheBackedEmbeddings`.

- Dense: `OpenAIEmbeddings` (paid) or a local fastembed model (free) behind the
  `Embeddings` interface.
- Sparse: `FastEmbedSparse("Qdrant/bm25")` from langchain-qdrant, for hybrid search.
- Cache: `CacheBackedEmbeddings` + `LocalFileStore`, keyed by sha256(text) in a namespace
  per model. Same idea as the custom app's SQLite cache, ready-made.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, SparseEmbeddings, SparseVector

from lcrag.callbacks import BudgetedEmbeddings, Ledger
from lcrag.config import Settings
from ragkit.embeddings.providers import FakeEmbedder


class FakeEmbeddings(Embeddings):
    """Adapter: the custom app's hashed bag-of-words embedder as a LangChain Embeddings.
    Unlike langchain_core's DeterministicFakeEmbedding (random vectors), it keeps lexical
    similarity, so retrieval tests are meaningful offline."""

    def __init__(self) -> None:
        self.inner = FakeEmbedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


class LocalFastEmbedEmbeddings(Embeddings):
    """Free local embeddings via fastembed. (langchain-community's FastEmbedEmbeddings was
    sunset with that package; implementing the `Embeddings` interface takes 10 lines.)"""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


class FakeSparse(SparseEmbeddings):
    def _v(self, text: str) -> SparseVector:
        import hashlib
        import re

        counts: dict[int, float] = {}
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            i = int(hashlib.md5(w.encode(), usedforsecurity=False).hexdigest()[:8], 16)
            counts[i] = counts.get(i, 0.0) + 1.0
        return SparseVector(indices=list(counts), values=list(counts.values()))

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._v(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._v(text)


OPENAI_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


def build_base_dense(settings: Settings, ledger: Ledger) -> BudgetedEmbeddings:
    """Always wrapped in BudgetedEmbeddings so we can count real (non-cached) embeddings."""
    if settings.embeddings_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not set (or use EMBEDDINGS_PROVIDER=local)")
        inner = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            max_retries=3,
        )
        return BudgetedEmbeddings(inner, ledger, settings.openai_embedding_model)
    if settings.embeddings_provider == "local":
        local = LocalFastEmbedEmbeddings(settings.local_embedding_model)
        return BudgetedEmbeddings(local, None, settings.local_embedding_model)
    return BudgetedEmbeddings(FakeEmbeddings(), None, "fake")


def embedding_dim(settings: Settings, base: Embeddings) -> int:
    if settings.embeddings_provider == "openai":
        return OPENAI_DIMS.get(settings.openai_embedding_model, 1536)
    return len(base.embed_query("dimension probe"))  # free for local/fake


def with_cache(base: Embeddings, settings: Settings) -> CacheBackedEmbeddings:
    store = LocalFileStore(str(settings.lc_data_dir / "embedding_cache"))
    return CacheBackedEmbeddings.from_bytes_store(
        base,
        store,
        namespace=settings.embedding_model_name,  # model in the key: no stale vectors
        key_encoder="sha256",
        query_embedding_cache=False,  # don't persist user queries
    )


@lru_cache(maxsize=2)
def _bm25(model: str) -> FastEmbedSparse:
    return FastEmbedSparse(model_name=model)


def build_sparse(settings: Settings) -> SparseEmbeddings:
    if settings.embeddings_provider == "fake":
        return FakeSparse()
    return _bm25(settings.sparse_model)

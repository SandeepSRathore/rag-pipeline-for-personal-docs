"""Shared fixtures. Everything runs offline and free:
in-memory Qdrant, hash-based fake embedders, a fake LLM, no reranker download.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import pytest

from ragkit.config import Settings, override_settings
from ragkit.embeddings.base import SparseVec
from ragkit.embeddings.providers import FakeEmbedder
from ragkit.generation.llm import FakeLLM
from ragkit.ingestion.pipeline import Ingestor
from ragkit.service import RAGService
from ragkit.store.qdrant_store import QdrantStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"


class FakeSparse:
    """Hashed term-frequency sparse vectors (Qdrant applies IDF)."""

    def _v(self, text: str) -> SparseVec:
        counts: dict[int, float] = {}
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            i = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
            counts[i] = counts.get(i, 0.0) + 1.0
        return SparseVec(list(counts), list(counts.values()))

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        return [self._v(t) for t in texts]

    def embed_query(self, text: str) -> SparseVec:
        return self._v(text)


class KeywordReranker:
    """Deterministic stand-in for the cross-encoder: word-overlap score, centered at 0."""

    STOP = {
        "the",
        "a",
        "an",
        "is",
        "of",
        "what",
        "how",
        "do",
        "i",
        "to",
        "for",
        "in",
        "my",
        "are",
        "does",
        "can",
        "get",
        "on",
        "and",
        "who",
        "when",
        "much",
        "many",
    }

    def score(self, query: str, docs: list[str]) -> list[float]:
        q = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in self.STOP}
        out = []
        for d in docs:
            words = set(re.findall(r"[a-z0-9]+", d.lower()))
            out.append(float(len(q & words)) - 1.0)
        return out


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "var",
        qdrant_path=":memory:",
        embeddings_provider="fake",
        llm_provider="fake",
        moderation_enabled=False,
        rerank_enabled=False,
        rerank_min_score=0.0,
        api_keys="emp-key:employee,hr-key:hr,admin-key:admin",  # type: ignore[arg-type]
        rate_limit_per_minute=1000,
        corpus_dir=tmp_path / "corpus",
    )
    override_settings(s)
    yield s
    override_settings(None)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    dst = tmp_path / "corpus"
    shutil.copytree(CORPUS, dst)
    return dst


@pytest.fixture
def store(settings: Settings) -> QdrantStore:
    return QdrantStore(settings)


@pytest.fixture
def ingestor(settings: Settings, store: QdrantStore) -> Ingestor:
    return Ingestor(settings, store=store, dense=FakeEmbedder(), sparse=FakeSparse())


@pytest.fixture
def indexed(ingestor: Ingestor, corpus: Path) -> Ingestor:
    rep = ingestor.ingest(corpus, source="corpus")
    assert not rep.failed, rep.failed
    return ingestor


@pytest.fixture
def service(settings: Settings, store: QdrantStore, indexed: Ingestor) -> RAGService:
    return RAGService(
        settings,
        store=store,
        dense=FakeEmbedder(),
        sparse=FakeSparse(),
        reranker=KeywordReranker(),
        llm=FakeLLM(),
    )

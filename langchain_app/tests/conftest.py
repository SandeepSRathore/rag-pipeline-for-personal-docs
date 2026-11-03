"""Offline, free fixtures: in-memory Qdrant, fake embeddings/LLM, keyword reranker."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from lcrag.chain import RAGChain
from lcrag.config import REPO_DIR, Settings, override_settings
from lcrag.indexing import LCIndexer


class KeywordReranker:
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
        return [float(len(q & set(re.findall(r"[a-z0-9]+", d.lower())))) - 1.0 for d in docs]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus"
    shutil.copytree(REPO_DIR / "data" / "corpus", corpus)
    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
        lc_data_dir=tmp_path / "var",
        qdrant_path=":memory:",
        embeddings_provider="fake",
        llm_provider="fake",
        moderation_enabled=False,
        rerank_min_score=0.0,
        corpus_dir=corpus,
        api_keys="emp-key:employee,hr-key:hr,admin-key:admin",  # type: ignore[arg-type]
        rate_limit_per_minute=1000,
    )
    override_settings(s)
    yield s
    override_settings(None)


@pytest.fixture
def indexer(settings: Settings) -> LCIndexer:
    return LCIndexer(settings, client=QdrantClient(location=":memory:"))


@pytest.fixture
def indexed(indexer: LCIndexer, settings: Settings) -> LCIndexer:
    rep = indexer.ingest(settings.corpus_dir, source="corpus")
    assert not rep.failed, rep.failed
    return indexer


@pytest.fixture
def rag(settings: Settings, indexed: LCIndexer) -> RAGChain:
    return RAGChain(
        settings,
        client=indexed.client,
        dense=indexed.base_dense,
        sparse=indexed.sparse,
        scorer=KeywordReranker(),
    )

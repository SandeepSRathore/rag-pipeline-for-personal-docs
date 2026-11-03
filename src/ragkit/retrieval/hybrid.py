"""Retrieval: hybrid search -> ACL filter -> rerank -> threshold -> recency annotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import models

from ragkit.config import Settings, allowed_acls
from ragkit.embeddings.base import DenseEmbedder, SparseEmbedder
from ragkit.retrieval.rerank import Reranker
from ragkit.store.qdrant_store import QdrantStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    path: str
    title: str
    section: str
    text: str
    acl: str
    effective_date: str | None
    doc_family: str | None
    fused_score: float
    rerank_score: float | None = None
    superseded_by: str | None = None  # title of a newer doc in the same family

    @classmethod
    def from_point(cls, p: models.ScoredPoint) -> RetrievedChunk:
        pl: dict[str, Any] = p.payload or {}
        return cls(
            chunk_id=str(p.id),
            doc_id=pl.get("doc_id", ""),
            path=pl.get("path", ""),
            title=pl.get("title", ""),
            section=pl.get("section", ""),
            text=pl.get("text", ""),
            acl=pl.get("acl", "public"),
            effective_date=pl.get("effective_date"),
            doc_family=pl.get("doc_family"),
            fused_score=float(p.score),
        )


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    candidates: list[RetrievedChunk] = field(default_factory=list)  # pre-threshold, for debug
    abstain_reason: str | None = None


def build_filter(role: str, exclude_flagged: bool) -> models.Filter:
    """ACL is enforced HERE, in the database query, never by asking the LLM to behave."""
    must: list[Any] = [
        models.FieldCondition(key="acl", match=models.MatchAny(any=allowed_acls(role)))
    ]
    must_not: list[Any] = []
    if exclude_flagged:
        must_not.append(models.FieldCondition(key="flagged", match=models.MatchValue(value=True)))
    return models.Filter(must=must, must_not=must_not or None)


def mark_superseded(chunks: list[RetrievedChunk]) -> None:
    """If several versions of the same doc family were retrieved, flag the older ones.

    We annotate instead of dropping: a user may legitimately ask about the old policy.
    The prompt tells the model to prefer current versions.
    """
    latest: dict[str, RetrievedChunk] = {}
    for c in chunks:
        if not c.doc_family or not c.effective_date:
            continue
        cur = latest.get(c.doc_family)
        if cur is None or c.effective_date > (cur.effective_date or ""):
            latest[c.doc_family] = c
    for c in chunks:
        newest = latest.get(c.doc_family or "")
        if (
            newest
            and c.effective_date
            and newest.effective_date
            and (c.effective_date < newest.effective_date)
        ):
            c.superseded_by = newest.title


class Retriever:
    def __init__(
        self,
        settings: Settings,
        store: QdrantStore,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        reranker: Reranker | None,
    ):
        self.s = settings
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker

    def retrieve(self, query: str, role: str) -> RetrievalResult:
        flt = build_filter(role, self.s.exclude_flagged_chunks)
        points = self.store.hybrid_query(
            dense=self.dense.embed_query(query),
            sparse=self.sparse.embed_query(query),
            query_filter=flt,
            limit=self.s.prefetch_k,
        )
        candidates = [RetrievedChunk.from_point(p) for p in points]
        if not candidates:
            return RetrievalResult([], [], abstain_reason="no_candidates")

        if self.reranker is not None:
            scores = self.reranker.score(query, [c.text for c in candidates])
            for c, sc in zip(candidates, scores, strict=True):
                c.rerank_score = sc
            candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
            kept = [c for c in candidates if (c.rerank_score or 0.0) >= self.s.rerank_min_score]
        else:
            kept = [c for c in candidates if c.fused_score >= self.s.rrf_min_score]

        kept = kept[: self.s.final_k]
        if not kept:
            return RetrievalResult([], candidates, abstain_reason="below_relevance_threshold")
        mark_superseded(kept)
        return RetrievalResult(kept, candidates)

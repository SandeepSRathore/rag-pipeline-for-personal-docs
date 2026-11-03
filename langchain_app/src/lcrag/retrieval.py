"""Retrieval: `QdrantVectorStore.as_retriever` (hybrid) wrapped in a
`ContextualCompressionRetriever` with a custom reranking compressor.

LangChain concepts used:
- VectorStoreRetriever: `vs.as_retriever(search_kwargs={"k": 20, "filter": acl_filter})`.
  The filter is a native Qdrant Filter, so the ACL is enforced INSIDE the database query.
- BaseDocumentCompressor: the extension point for "post-process retrieved docs". Built-in
  rerankers (Cohere, FlashRank, HuggingFace cross-encoder) exist, but we wrap the same
  fastembed cross-encoder the custom app uses so the two apps are comparable.
- ContextualCompressionRetriever: base retriever + compressor = one Retriever (a Runnable).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from pydantic import ConfigDict
from qdrant_client import models

from lcrag.config import ROLE_ACCESS, Settings


def acl_filter(role: str, exclude_flagged: bool) -> models.Filter:
    # LangChain stores metadata under the "metadata" payload key.
    must: list[Any] = [
        models.FieldCondition(
            key="metadata.acl", match=models.MatchAny(any=ROLE_ACCESS.get(role, ["public"]))
        )
    ]
    must_not: list[Any] = []
    if exclude_flagged:
        must_not.append(
            models.FieldCondition(key="metadata.flagged", match=models.MatchValue(value=True))
        )
    return models.Filter(must=must, must_not=must_not or None)


class CrossEncoderRerank(BaseDocumentCompressor):
    """Scores every candidate with a cross-encoder and sorts by score.

    It does NOT drop anything: thresholding happens in the next step so the debug view can
    still show what was dropped and why.
    """

    scorer: Any  # anything with .score(query, docs) -> list[float]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        if not documents:
            return []
        scores = self.scorer.score(query, [d.page_content for d in documents])
        out = []
        for d, sc in zip(documents, scores, strict=True):
            out.append(
                Document(
                    page_content=d.page_content,
                    id=d.id,
                    metadata={**d.metadata, "rerank_score": float(sc)},
                )
            )
        return sorted(out, key=lambda d: d.metadata["rerank_score"], reverse=True)


def build_retriever(
    vs: QdrantVectorStore, settings: Settings, role: str, scorer: Any | None
) -> BaseRetriever:
    base = vs.as_retriever(
        search_kwargs={
            "k": settings.prefetch_k,
            "filter": acl_filter(role, settings.exclude_flagged_chunks),
        }
    )
    if scorer is None:
        return base
    return ContextualCompressionRetriever(
        base_compressor=CrossEncoderRerank(scorer=scorer), base_retriever=base
    )


def apply_threshold(docs: list[Document], settings: Settings, reranked: bool) -> list[Document]:
    if reranked:
        docs = [d for d in docs if d.metadata["rerank_score"] >= settings.rerank_min_score]
    return docs[: settings.final_k]


def mark_superseded(docs: list[Document]) -> None:
    latest: dict[str, Document] = {}
    for d in docs:
        fam, date = d.metadata.get("doc_family"), d.metadata.get("effective_date")
        if fam and date and (fam not in latest or date > latest[fam].metadata["effective_date"]):
            latest[fam] = d
    for d in docs:
        newest = latest.get(d.metadata.get("doc_family") or "")
        date = d.metadata.get("effective_date")
        if newest and date and date < newest.metadata["effective_date"]:
            d.metadata["superseded_by"] = newest.metadata["title"]

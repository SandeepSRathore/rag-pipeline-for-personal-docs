"""Cross-encoder reranking (local, free).

Embedding search compares a query vector to chunk vectors computed independently
("bi-encoder"): fast, but coarse. A cross-encoder reads the query and the chunk
TOGETHER and outputs a relevance score: much more accurate, too slow for millions of
docs, perfect for re-ordering the top ~20 candidates.

Its scores are also absolute (a logit), so we can apply a relevance threshold and
abstain when nothing is relevant. RRF scores are rank-based and can't do that.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name)

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, docs)]


@lru_cache(maxsize=2)
def get_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)

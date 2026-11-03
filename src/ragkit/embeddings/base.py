"""Embedder interfaces.

Dense embedders map text -> vector of floats (semantic similarity).
The sparse embedder (BM25) maps text -> {token_id: weight} (exact keyword match).
Hybrid search uses both because each fails where the other shines:
dense misses exact codes/names ("ERR-4012"), BM25 misses paraphrases ("time off" vs "PTO").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class DenseEmbedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class SparseEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[SparseVec]: ...

    def embed_query(self, text: str) -> SparseVec: ...

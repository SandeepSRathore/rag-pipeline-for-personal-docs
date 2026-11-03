"""Qdrant via `langchain_qdrant.QdrantVectorStore` in HYBRID mode.

LangChain gives us hybrid search (dense + BM25 prefetch, RRF fusion) and upserts/deletes.
It does NOT manage aliases or blue/green rebuilds, so collection lifecycle is plain
qdrant-client code here, the same pattern as the custom app.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from qdrant_client import QdrantClient, models

from lcrag.config import Settings

DENSE = "dense"
SPARSE = "langchain-sparse"


class PipelineVersionMismatchError(RuntimeError):
    pass


def make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_path == ":memory:":
        return QdrantClient(location=":memory:")
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=key, timeout=30)


class Collections:
    """Physical collections `lc_docs_<pipeline_version>_<ts>` behind the alias `lc_docs`."""

    def __init__(self, client: QdrantClient, settings: Settings):
        self.client = client
        self.s = settings
        self.alias = settings.lc_collection_alias

    def prefix(self) -> str:
        return f"{self.alias}_{self.s.pipeline_version}_"

    def new_name(self) -> str:
        return self.prefix() + datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")

    def target(self) -> str | None:
        for a in self.client.get_aliases().aliases:
            if a.alias_name == self.alias:
                return a.collection_name
        return None

    def create(self, name: str, dim: int) -> None:
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            name,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if self.s.qdrant_path:  # embedded mode ignores payload indexes
            return
        for field, schema in [
            ("metadata.source", models.PayloadSchemaType.KEYWORD),
            ("metadata.acl", models.PayloadSchemaType.KEYWORD),
            ("metadata.flagged", models.PayloadSchemaType.BOOL),
        ]:
            self.client.create_payload_index(name, field_name=field, field_schema=schema)

    def ensure_active(self, dim: int) -> str:
        target = self.target()
        if target is None:
            name = self.new_name()
            self.create(name, dim)
            self.swap(name)
            return name
        if not target.startswith(self.prefix()):
            raise PipelineVersionMismatchError(
                f"Index '{target}' was built with different settings "
                f"(pipeline {self.s.pipeline_version}). Run `lcrag reindex`."
            )
        return target

    def swap(self, new: str) -> str | None:
        old = self.target()
        ops: list[Any] = []
        if old:
            ops.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=self.alias))
            )
        ops.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=new, alias_name=self.alias)
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=ops)
        return old

    def drop(self, name: str) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)


def vector_store(
    client: QdrantClient, collection: str, dense: Embeddings, sparse: SparseEmbeddings
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE,
        sparse_vector_name=SPARSE,
    )

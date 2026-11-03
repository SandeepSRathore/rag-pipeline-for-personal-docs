"""Qdrant vector store with blue/green collections behind an alias.

Physical collection: `docs_<pipeline_version>`   (immutable "shape": model, dims, chunking)
Alias:               `docs`                       (what queries use)

A full reindex builds a new collection and swaps the alias atomically, so queries
never see a half-built index, and rollback is just pointing the alias back.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient, models

from ragkit.config import Settings
from ragkit.embeddings.base import SparseVec

DENSE = "dense"
SPARSE = "bm25"


class PipelineVersionMismatchError(RuntimeError):
    pass


@dataclass
class PointData:
    id: str
    dense: list[float]
    sparse: SparseVec
    payload: dict[str, Any]


class QdrantStore:
    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.settings = settings
        self.alias = settings.collection_alias
        if client is not None:
            self.client = client
        elif settings.qdrant_path:
            self.client = (
                QdrantClient(location=":memory:")
                if settings.qdrant_path == ":memory:"
                else QdrantClient(path=settings.qdrant_path)
            )
        else:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=(
                    settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
                ),
                timeout=30,
            )
        self.embedded = client is None and bool(settings.qdrant_path)

    # ---- collections & aliases -------------------------------------------------
    def collection_prefix(self, pipeline_version: str) -> str:
        return f"{self.alias}_{pipeline_version}_"

    def new_collection_name(self, pipeline_version: str) -> str:
        return self.collection_prefix(pipeline_version) + datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    def alias_target(self) -> str | None:
        for a in self.client.get_aliases().aliases:
            if a.alias_name == self.alias:
                return a.collection_name
        return None

    def create_collection(self, name: str, dim: int) -> None:
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # Payload indexes make filtered search fast (server mode only; embedded ignores them).
        if self.embedded:
            return
        for field, schema in [
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("acl", models.PayloadSchemaType.KEYWORD),
            ("doc_family", models.PayloadSchemaType.KEYWORD),
            ("flagged", models.PayloadSchemaType.BOOL),
        ]:
            self.client.create_payload_index(name, field_name=field, field_schema=schema)

    def ensure_active(self, pipeline_version: str, dim: int) -> str:
        """Return the collection to write to for incremental ingests."""
        target = self.alias_target()
        if target is None:
            name = self.new_collection_name(pipeline_version)
            self.create_collection(name, dim)
            self.swap_alias(name)
            return name
        if not target.startswith(self.collection_prefix(pipeline_version)):
            raise PipelineVersionMismatchError(
                f"Index '{target}' was built with different settings than the current config "
                f"(pipeline version {pipeline_version}). Run `rag reindex` to rebuild it "
                "with a blue/green swap."
            )
        return target

    def swap_alias(self, new_collection: str) -> str | None:
        old = self.alias_target()
        ops: list[Any] = []
        if old:
            ops.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=self.alias))
            )
        ops.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=new_collection, alias_name=self.alias
                )
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=ops)  # atomic
        return old

    def drop_collection(self, name: str) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)

    # ---- writes ------------------------------------------------------------------
    def upsert(self, collection: str, points: list[PointData]) -> None:
        for i in range(0, len(points), 128):
            batch = points[i : i + 128]
            self.client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=p.id,
                        vector={
                            DENSE: p.dense,
                            SPARSE: models.SparseVector(
                                indices=p.sparse.indices, values=p.sparse.values
                            ),
                        },
                        payload=p.payload,
                    )
                    for p in batch
                ],
                wait=True,
            )

    def delete_ids(self, collection: str, ids: list[str]) -> None:
        if ids:
            self.client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=list(ids)),
                wait=True,
            )

    def delete_stale(self, collection: str, doc_id: str, keep_ids: list[str]) -> None:
        """Delete this doc's points that are NOT in keep_ids (old chunks after an edit)."""
        self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                    ],
                    must_not=[models.HasIdCondition(has_id=list(keep_ids))],
                )
            ),
            wait=True,
        )

    def delete_doc(self, collection: str, doc_id: str) -> None:
        self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                    ]
                )
            ),
            wait=True,
        )

    # ---- reads -------------------------------------------------------------------
    def hybrid_query(
        self,
        dense: list[float],
        sparse: SparseVec,
        query_filter: models.Filter,
        limit: int,
    ) -> list[models.ScoredPoint]:
        """Dense + BM25 candidates, fused with Reciprocal Rank Fusion inside Qdrant."""
        res = self.client.query_points(
            collection_name=self.alias,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE, limit=limit, filter=query_filter),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE,
                    limit=limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return list(res.points)

    def count(self) -> int:
        if self.alias_target() is None:
            return 0
        return self.client.count(self.alias, exact=True).count

    def flagged_chunks(self, limit: int = 50) -> list[dict[str, Any]]:
        if self.alias_target() is None:
            return []
        pts, _ = self.client.scroll(
            self.alias,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="flagged", match=models.MatchValue(value=True))]
            ),
            limit=limit,
            with_payload=["path", "section", "flag_reasons"],
        )
        return [dict(p.payload or {}) for p in pts]

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

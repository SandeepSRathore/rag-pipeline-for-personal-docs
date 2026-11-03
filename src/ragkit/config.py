"""Central configuration.

Every tunable lives here and is read from environment variables / `.env`.
Secrets use `SecretStr` so they never show up in logs, reprs or tracebacks.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Role = Literal["employee", "hr", "admin"]

# Which document ACL tags each role may read. Retrieval filters on this INSIDE
# the vector DB query, so restricted text never reaches the LLM.
ROLE_ACCESS: dict[str, list[str]] = {
    "employee": ["public"],
    "hr": ["public", "hr"],
    "admin": ["public", "hr"],
}

# Bump when normalization logic changes; it is part of pipeline_version.
NORMALIZER_VERSION = "1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- providers ---
    openai_api_key: SecretStr | None = None
    llm_provider: Literal["openai", "fake"] = "openai"
    llm_model: str = "gpt-5-nano"
    llm_reasoning_effort: str = "minimal"
    embeddings_provider: Literal["openai", "local", "fake"] = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_enabled: bool = True
    moderation_enabled: bool = True
    moderation_model: str = "omni-moderation-latest"

    # --- pricing (USD per 1M tokens) for the cost ledger ---
    llm_input_price_per_m: float = 0.05
    llm_output_price_per_m: float = 0.40
    embedding_price_per_m: float = 0.02

    # --- storage ---
    qdrant_url: str = "http://localhost:6333"
    # Embedded mode: set a directory (or ":memory:") to run Qdrant inside this process,
    # no Docker needed. Single-process only (CLI and API can't share it concurrently).
    qdrant_path: str | None = None
    qdrant_api_key: SecretStr | None = None
    collection_alias: str = "docs"
    data_dir: Path = Path("var")
    corpus_dir: Path = Path("data/corpus")

    # --- chunking ---
    chunk_tokens: int = 450
    chunk_overlap_tokens: int = 60

    # --- retrieval ---
    prefetch_k: int = 20
    final_k: int = 5
    # Cross-encoder logit below which a chunk is considered irrelevant. Tuned with
    # `rag eval`: relevant chunks score > -3, off-topic ones < -7 for ms-marco-MiniLM.
    rerank_min_score: float = -5.0
    # Used when reranking is disabled: minimum RRF score of the best hit.
    rrf_min_score: float = 0.0
    exclude_flagged_chunks: bool = True

    # --- generation / cost ---
    max_output_tokens: int = 600
    daily_budget_usd: float = 0.50

    # --- input limits ---
    max_query_chars: int = 1000
    max_upload_bytes: int = 5 * 1024 * 1024
    max_pdf_pages: int = 200

    # --- API ---
    api_keys: SecretStr = SecretStr("")
    rate_limit_per_minute: int = 20

    @field_validator("data_dir")
    @classmethod
    def _mkdir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ragkit.sqlite3"

    @property
    def embedding_model_name(self) -> str:
        return {
            "openai": self.openai_embedding_model,
            "local": self.local_embedding_model,
            "fake": "fake-hash-embedder",
        }[self.embeddings_provider]

    @property
    def pipeline_version(self) -> str:
        """Fingerprint of everything that affects stored vectors.

        If this changes, previously ingested docs are stale and get re-processed.
        """
        payload = json.dumps(
            {
                "embed": self.embedding_model_name,
                "sparse": self.sparse_model,
                "chunk": [self.chunk_tokens, self.chunk_overlap_tokens],
                "norm": NORMALIZER_VERSION,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def api_key_roles(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for pair in self.api_keys.get_secret_value().split(","):
            if ":" not in pair:
                continue
            key, role = pair.rsplit(":", 1)
            key, role = key.strip(), role.strip()
            if key and role in ROLE_ACCESS:
                out[key] = role
        return out


class _Holder:
    settings: Settings | None = None


@lru_cache(maxsize=1)
def _load() -> Settings:
    return Settings()


def get_settings() -> Settings:
    return _Holder.settings or _load()


def override_settings(settings: Settings | None) -> None:
    """Used by tests to inject settings."""
    _Holder.settings = settings


def allowed_acls(role: str) -> list[str]:
    return ROLE_ACCESS.get(role, ["public"])

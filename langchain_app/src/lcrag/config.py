"""Settings for the LangChain edition.

Same environment variable names as the custom `ragkit` app, so the repo-root `.env` works
for both. Differences: a separate Qdrant alias (`lc_docs`) and a separate state dir, so the
two apps can share one Qdrant server without touching each other's data.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parents[2]  # langchain_app/
REPO_DIR = APP_DIR.parent

ROLE_ACCESS: dict[str, list[str]] = {
    "employee": ["public"],
    "hr": ["public", "hr"],
    "admin": ["public", "hr"],
}
# Bump when chunking/ID logic changes: it's part of pipeline_version.
INDEXING_VERSION = "1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_DIR / ".env", APP_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

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
    rerank_min_score: float = -5.0
    moderation_enabled: bool = True

    llm_input_price_per_m: float = 0.05
    llm_output_price_per_m: float = 0.40
    embedding_price_per_m: float = 0.02
    daily_budget_usd: float = 0.50
    max_output_tokens: int = 600
    max_query_chars: int = 1000

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_path: str | None = None
    lc_collection_alias: str = "lc_docs"
    lc_data_dir: Path = APP_DIR / "var"
    corpus_dir: Path = REPO_DIR / "data" / "corpus"
    eval_dir: Path = REPO_DIR / "eval"

    chunk_tokens: int = 450
    chunk_overlap_tokens: int = 60
    prefetch_k: int = 20
    final_k: int = 5
    exclude_flagged_chunks: bool = True
    max_upload_bytes: int = 5 * 1024 * 1024

    api_keys: SecretStr = SecretStr("")
    rate_limit_per_minute: int = 20

    @field_validator("lc_data_dir")
    @classmethod
    def _mkdir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.lc_data_dir / 'record_manager.sqlite3'}"

    @property
    def ledger_path(self) -> Path:
        return self.lc_data_dir / "cost_ledger.sqlite3"

    @property
    def embedding_model_name(self) -> str:
        return {
            "openai": self.openai_embedding_model,
            "local": self.local_embedding_model,
            "fake": "fake-hash-embedder",
        }[self.embeddings_provider]

    @property
    def pipeline_version(self) -> str:
        payload = json.dumps(
            {
                "embed": self.embedding_model_name,
                "sparse": self.sparse_model,
                "chunk": [self.chunk_tokens, self.chunk_overlap_tokens],
                "indexing": INDEXING_VERSION,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def api_key_roles(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for pair in self.api_keys.get_secret_value().split(","):
            if ":" in pair:
                key, role = (x.strip() for x in pair.rsplit(":", 1))
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
    _Holder.settings = settings

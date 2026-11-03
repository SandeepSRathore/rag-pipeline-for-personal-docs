"""Cost ledger + budget cap, done the LangChain way: a callback handler.

LangChain fires callbacks around every model call (`on_chat_model_start`, `on_llm_end`, ...).
A handler with `raise_error = True` can abort a call by raising, which is how we enforce the
daily budget BEFORE money is spent. `on_llm_end` reads the token usage LangChain normalized
into `usage_metadata`.

Gap: embedding models don't emit callbacks, so `BudgetedEmbeddings` wraps them and counts
tokens itself (tiktoken), because LangChain's OpenAIEmbeddings doesn't expose API usage.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from typing import Any

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import LLMResult

from lcrag.config import Settings


class BudgetExceededError(RuntimeError):
    pass


class Ledger:
    def __init__(self, settings: Settings):
        self.s = settings
        with closing(sqlite3.connect(settings.ledger_path)) as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS cost (ts TEXT, day TEXT, kind TEXT, model TEXT,"
                " input_tokens INT, output_tokens INT, usd REAL)"
            )
            c.commit()

    def record(self, kind: str, model: str, in_t: int, out_t: int = 0) -> float:
        if kind == "llm":
            usd = (
                in_t * self.s.llm_input_price_per_m + out_t * self.s.llm_output_price_per_m
            ) / 1e6
        else:
            usd = in_t * self.s.embedding_price_per_m / 1e6
        now = datetime.now(UTC)
        with closing(sqlite3.connect(self.s.ledger_path)) as c:
            c.execute(
                "INSERT INTO cost VALUES (?,?,?,?,?,?,?)",
                (now.isoformat(), now.strftime("%Y-%m-%d"), kind, model, in_t, out_t, usd),
            )
            c.commit()
        return usd

    def spent_today(self) -> float:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        with closing(sqlite3.connect(self.s.ledger_path)) as c:
            row = c.execute("SELECT COALESCE(SUM(usd),0) FROM cost WHERE day=?", (day,)).fetchone()
        return float(row[0])

    def check(self) -> None:
        spent = self.spent_today()
        if spent >= self.s.daily_budget_usd:
            raise BudgetExceededError(
                f"Daily budget of ${self.s.daily_budget_usd:.2f} reached (spent ${spent:.4f})."
            )


class CostCallbackHandler(BaseCallbackHandler):
    """Checks the budget before each chat model call and records usage after it."""

    raise_error = True  # let BudgetExceededError propagate and abort the call

    def __init__(self, ledger: Ledger, model: str):
        self.ledger = ledger
        self.model = model
        self.last: dict[str, Any] = {}

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        self.ledger.check()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        in_t = out_t = 0
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                in_t += usage.get("input_tokens", 0)
                out_t += usage.get("output_tokens", 0)
        usd = self.ledger.record("llm", self.model, in_t, out_t)
        self.last = {"input_tokens": in_t, "output_tokens": out_t, "usd": round(usd, 6)}


class BudgetedEmbeddings(Embeddings):
    """Wraps an embedding model. Counts texts actually embedded (= cache misses, since it sits
    UNDER CacheBackedEmbeddings). For paid models (ledger set): budget check + cost record."""

    def __init__(self, inner: Embeddings, ledger: Ledger | None, model: str):
        self.inner = inner
        self.ledger = ledger
        self.model = model
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.calls = 0  # number of texts actually sent to the API (cache misses)

    def _count(self, texts: list[str]) -> int:
        return sum(len(self.enc.encode(t)) for t in texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.ledger:
            self.ledger.check()
        out = self.inner.embed_documents(texts)
        self.calls += len(texts)
        if self.ledger:
            self.ledger.record("embedding", self.model, self._count(texts))
        return out

    def embed_query(self, text: str) -> list[float]:
        if self.ledger:
            self.ledger.check()
        out = self.inner.embed_query(text)
        if self.ledger:
            self.ledger.record("embedding", self.model, self._count([text]))
        return out

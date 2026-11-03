"""Cost ledger + hard daily budget cap (OWASP LLM10: unbounded consumption)."""

from __future__ import annotations

from datetime import UTC, datetime

from ragkit.config import Settings
from ragkit.db import connect
from ragkit.observability.logging import request_id_var


class BudgetExceededError(RuntimeError):
    pass


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def record(
    settings: Settings, kind: str, model: str, input_tokens: int, output_tokens: int = 0
) -> float:
    if kind == "llm":
        usd = (
            input_tokens * settings.llm_input_price_per_m
            + output_tokens * settings.llm_output_price_per_m
        ) / 1_000_000
    else:
        usd = input_tokens * settings.embedding_price_per_m / 1_000_000
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO cost_ledger(ts, day, kind, model, input_tokens, output_tokens, usd,"
            " request_id) VALUES (?,?,?,?,?,?,?,?)",
            (
                datetime.now(UTC).isoformat(),
                _today(),
                kind,
                model,
                input_tokens,
                output_tokens,
                usd,
                request_id_var.get(),
            ),
        )
    return usd


def spent_today(settings: Settings) -> float:
    with connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(usd),0) AS s FROM cost_ledger WHERE day=?", (_today(),)
        ).fetchone()
    return float(row["s"])


def spent_total(settings: Settings) -> float:
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT COALESCE(SUM(usd),0) AS s FROM cost_ledger").fetchone()
    return float(row["s"])


def check_budget(settings: Settings) -> None:
    """Call BEFORE any paid API call. Fails closed."""
    if settings.llm_provider == "fake" and settings.embeddings_provider != "openai":
        return
    spent = spent_today(settings)
    if spent >= settings.daily_budget_usd:
        raise BudgetExceededError(
            f"Daily budget of ${settings.daily_budget_usd:.2f} reached (spent ${spent:.4f})."
        )

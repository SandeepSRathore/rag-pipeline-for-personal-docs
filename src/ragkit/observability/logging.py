"""Structured JSON logging with a per-request id.

Rules: never log secrets, never log raw user queries (log a hash + length instead).
"""

from __future__ import annotations

import hashlib
import logging
import sys
import uuid
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _add_request_id(_logger: Any, _name: str, event: MutableMapping[str, Any]) -> Any:
    event.setdefault("request_id", request_id_var.get())
    return event


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    renderer = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_request_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    set_request_id(rid)
    return rid


def set_request_id(rid: str) -> None:
    request_id_var.set(rid)


def redact_query(text: str) -> dict:
    """What we log instead of the raw query."""
    return {"q_sha": hashlib.sha256(text.encode()).hexdigest()[:10], "q_len": len(text)}


def get_logger(name: str = "ragkit") -> Any:
    return structlog.get_logger(name)

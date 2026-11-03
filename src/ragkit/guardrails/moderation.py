"""OpenAI moderation (free endpoint): blocks harmful content (violence, self-harm, ...).

Failure policy: FAIL OPEN with a warning. Moderation here is about harmful content, not
injection, and the other layers still run. For a public-facing app you might fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragkit.config import Settings
from ragkit.observability.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModerationVerdict:
    flagged: bool
    categories: list[str] = field(default_factory=list)
    error: str | None = None


def moderate(text: str, settings: Settings) -> ModerationVerdict:
    if not settings.moderation_enabled or settings.openai_api_key is None:
        return ModerationVerdict(False)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key.get_secret_value(), timeout=10)
        res = client.moderations.create(model=settings.moderation_model, input=text).results[0]
        cats = [k for k, v in res.categories.model_dump().items() if v]
        return ModerationVerdict(bool(res.flagged), cats)
    except Exception as e:
        log.warning("moderation_unavailable", error=type(e).__name__)
        return ModerationVerdict(False, error=type(e).__name__)

"""Input guardrail: runs BEFORE retrieval and before any paid call.

Order matters: cheap local checks first (length, injection heuristics), the network
moderation call last, so obvious attacks cost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragkit.config import Settings
from ragkit.guardrails.injection_detect import detect_injection
from ragkit.guardrails.moderation import moderate
from ragkit.ingestion.normalize import normalize_text

REFUSAL_INJECTION = (
    "I can only answer questions about Lumora Labs documents, and I can't change my "
    "instructions or reveal them."
)
REFUSAL_MODERATION = "I can't help with that request."


@dataclass
class InputVerdict:
    allowed: bool
    clean_query: str
    reason: str | None = None
    message: str | None = None
    details: dict = field(default_factory=dict)


def check_input(query: str, settings: Settings, use_moderation: bool = True) -> InputVerdict:
    norm = normalize_text(query)
    q = norm.text
    if not q:
        return InputVerdict(False, q, "empty", "Please enter a question.")
    if len(q) > settings.max_query_chars:
        return InputVerdict(
            False, q, "too_long", f"Questions are limited to {settings.max_query_chars} characters."
        )
    inj = detect_injection(q)
    details = {"injection_score": inj.score, "injection_reasons": inj.reasons}
    if norm.hidden_chars_removed:
        details["hidden_chars_removed"] = norm.hidden_chars_removed
    if inj.is_injection():
        return InputVerdict(False, q, "prompt_injection", REFUSAL_INJECTION, details)
    if use_moderation:
        mod = moderate(q, settings)
        details["moderation"] = {"flagged": mod.flagged, "categories": mod.categories}
        if mod.flagged:
            return InputVerdict(False, q, "moderation", REFUSAL_MODERATION, details)
    return InputVerdict(True, q, details=details)

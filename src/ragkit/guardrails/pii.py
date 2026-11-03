"""Redact secrets and personal data from model output (OWASP LLM02).

Secrets (API keys, private keys, card numbers, SSNs) are ALWAYS redacted.
Emails are redacted unless they belong to an allow-listed company domain, so the bot can
still tell you to "email benefits@lumora-labs.example".
"""

from __future__ import annotations

import re

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key", re.compile(r"\bsk-(proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b")),
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)"
        ),
    ),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

ALLOWED_EMAIL_DOMAINS = {"lumora-labs.example"}


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def redact(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for name, rx in _SECRET_PATTERNS:
        if rx.search(text):
            found.append(name)
            text = rx.sub(f"[REDACTED:{name}]", text)

    def _card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("card_number")
            return "[REDACTED:card_number]"
        return m.group(0)

    text = _CARD.sub(_card, text)

    def _email(m: re.Match[str]) -> str:
        if m.group(1).lower() in ALLOWED_EMAIL_DOMAINS:
            return m.group(0)
        found.append("email")
        return "[REDACTED:email]"

    text = _EMAIL.sub(_email, text)
    return text, found

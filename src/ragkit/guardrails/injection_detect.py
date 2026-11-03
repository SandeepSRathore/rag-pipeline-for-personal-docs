"""Heuristic prompt-injection detector.

Used in two places:
- on user queries (direct injection), before anything is sent to the LLM
- on document chunks at ingestion (indirect injection), to flag/quarantine them

Heuristics are a FIRST layer, not a guarantee: a determined attacker can phrase things
we don't match. That's why we also have spotlighting, instruction hierarchy in the
system prompt, a canary token and output checks (defense in depth). In a larger
deployment you'd add a trained classifier (e.g. a prompt-injection model) here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_RAW_PATTERNS: list[tuple[str, float, str]] = [
    (
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|"
        r"all|any|your|the|system)\b.{0,30}\b(instructions?|rules?|prompts?|directions?|"
        r"guidelines?|context)\b",
        0.9,
        "instruction_override",
    ),
    (
        r"\b(reveal|show|print|repeat|output|display|leak|tell me|what (is|are))\b.{0,30}"
        r"\b(system|initial|hidden|secret|original|developer)\s+(prompt|instructions?|message|"
        r"rules?|marker)\b",
        0.85,
        "prompt_extraction",
    ),
    (
        r"\b(you are now|from now on,? you|act as|pretend (to be|you are)|roleplay as|"
        r"simulate)\b.{0,40}\b(dan|developer mode|unrestricted|jailbroken|evil|no (rules|limits|"
        r"restrictions))\b",
        0.8,
        "role_hijack",
    ),
    (r"\b(do anything now|developer mode enabled|jailbreak(ed)?)\b", 0.7, "jailbreak_keyword"),
    (
        r"(^|\n)\s*(system|assistant)\s*:|<\|im_start\|>|\[/?INST\]|<\s*/?\s*system\s*>|"
        r"###\s*(new\s+)?instructions?",
        0.6,
        "fake_role_markers",
    ),
    (r"\bnew (instructions?|task|rules?)\s*:", 0.6, "new_instructions"),
    (r"!\[[^\]]*\]\(\s*https?://", 0.6, "markdown_image_exfil"),
    (
        r"\b(send|post|forward|upload|exfiltrate|email)\b.{0,40}\b(conversation|chat history|"
        r"data|secrets?|credentials|api keys?|this (chat|answer))\b.{0,30}\b(to|at)\b",
        0.6,
        "data_exfil",
    ),
    (r"https?://\S+[?&](q|d|data|payload|c)=", 0.35, "url_with_payload"),
    (r"[A-Za-z0-9+/]{120,}={0,2}", 0.3, "long_base64"),
]
_PATTERNS = [(re.compile(p, re.I | re.S), w, name) for p, w, name in _RAW_PATTERNS]

# Catches obfuscations like "i g n o r e  a l l  p r e v i o u s  i n s t r u c t i o n s".
_SQUASHED_PHRASES = [
    "ignoreallpreviousinstructions",
    "ignorepreviousinstructions",
    "ignoreyourinstructions",
    "disregardpreviousinstructions",
    "revealyoursystemprompt",
    "printyoursystemprompt",
]


@dataclass
class InjectionVerdict:
    score: float
    reasons: list[str] = field(default_factory=list)

    def is_injection(self, threshold: float = 0.6) -> bool:
        return self.score >= threshold


def detect_injection(text: str) -> InjectionVerdict:
    reasons: list[str] = []
    keep = 1.0
    for rx, weight, name in _PATTERNS:
        if rx.search(text):
            reasons.append(name)
            keep *= 1 - weight
    squashed = re.sub(r"[^a-z]", "", text.lower())
    if any(p in squashed for p in _SQUASHED_PHRASES) and "instruction_override" not in reasons:
        reasons.append("obfuscated_override")
        keep *= 1 - 0.85
    return InjectionVerdict(score=round(1 - keep, 3), reasons=reasons)

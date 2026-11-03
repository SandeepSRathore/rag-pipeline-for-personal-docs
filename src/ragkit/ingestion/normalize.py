"""Text normalization.

Security note: attackers hide instructions with characters humans can't see but models
can read, or use them to break up keywords so filters miss them ("ig<ZWSP>nore").
We strip them BEFORE chunking, embedding and injection scanning.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Zero-width chars, soft hyphen, bidi controls/overrides, word joiners, BOM.
_INVISIBLE = re.compile("[­᠎​-‏‪-‮⁠-⁤⁦-⁩﻿]")
# Unicode "tag" block (U+E0000-U+E007F): invisible ASCII look-alikes used for
# "ASCII smuggling" prompt injections.
_TAGS = re.compile("[\U000e0000-\U000e007f]")
# C0/C1 control chars except \t \n \r.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class NormalizedText:
    text: str
    hidden_chars_removed: int


def normalize_text(raw: str) -> NormalizedText:
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    hidden = len(_INVISIBLE.findall(text)) + len(_TAGS.findall(text))
    text = _INVISIBLE.sub("", text)
    text = _TAGS.sub("", text)
    text = _CONTROL.sub("", text)
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text).strip()
    return NormalizedText(text=text, hidden_chars_removed=hidden)

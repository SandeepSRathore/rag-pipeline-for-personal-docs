"""Structure-aware chunking.

1. Split on Markdown headings so a chunk never mixes unrelated sections.
2. Inside a section, pack paragraphs (then sentences, then raw tokens) greedily up to
   `max_tokens`, carrying `overlap` tokens of trailing context into the next chunk.
3. Prefix every chunk with a breadcrumb header ("Title > Section > Subsection").
   This "contextual chunk header" gives a short chunk the context it otherwise loses
   ("Eligibility" alone means nothing; "PTO Policy > Eligibility" does).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@lru_cache(maxsize=1)
def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc().encode(text))


@dataclass(frozen=True)
class Chunk:
    index: int
    section: str  # "Section > Subsection"
    body: str  # chunk text without header
    text: str  # header + body, what gets embedded and shown to the LLM


def split_sections(text: str) -> list[tuple[list[str], str]]:
    """Return [(heading_path, body)] in document order."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(([h for _, h in stack], body))
        buf.clear()

    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            buf.append(line)
    flush()
    return sections


def _units(text: str, max_t: int) -> Iterator[str]:
    """Yield paragraph/sentence/token units, each <= max_t tokens."""
    enc = _enc()
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) <= max_t:
            yield para
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            toks = enc.encode(sent)
            if len(toks) <= max_t:
                yield sent
            else:
                for i in range(0, len(toks), max_t):
                    yield enc.decode(toks[i : i + max_t])


def _pack(units: list[str], max_t: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[tuple[str, int]] = []
    cur_t = 0
    for u in units:
        t = count_tokens(u)
        if cur and cur_t + t > max_t:
            chunks.append("\n\n".join(x for x, _ in cur))
            tail: list[tuple[str, int]] = []
            tail_t = 0
            for x, xt in reversed(cur):
                if tail_t + xt > overlap:
                    break
                tail.insert(0, (x, xt))
                tail_t += xt
            cur, cur_t = tail, tail_t
            while cur and cur_t + t > max_t:
                cur_t -= cur.pop(0)[1]
        cur.append((u, t))
        cur_t += t
    if cur:
        chunks.append("\n\n".join(x for x, _ in cur))
    return chunks


def chunk_document(text: str, title: str, max_tokens: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for path, body in split_sections(text):
        # Drop a leading H1 that just repeats the document title.
        if path and path[0].strip().lower() == title.strip().lower():
            path = path[1:]
        section = " > ".join(path)
        header = " > ".join([title, *path])
        budget = max(max_tokens - count_tokens(header) - 2, 50)
        for piece in _pack(list(_units(body, budget)), budget, overlap):
            out.append(
                Chunk(index=len(out), section=section, body=piece, text=f"{header}\n\n{piece}")
            )
    return out

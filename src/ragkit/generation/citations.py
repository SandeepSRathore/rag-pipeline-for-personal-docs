"""Citation validation: the model may only cite documents we actually gave it."""

from __future__ import annotations

from dataclasses import dataclass

from ragkit.retrieval.hybrid import RetrievedChunk


@dataclass
class Citation:
    ref: str
    title: str
    section: str
    path: str
    doc_id: str
    chunk_id: str


def validate_citations(
    refs: list[str], ref_map: dict[str, RetrievedChunk]
) -> tuple[list[Citation], list[str]]:
    """Return (valid citations, invalid refs). Duplicates collapse, order is preserved."""
    valid: list[Citation] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for raw in refs:
        ref = raw.strip().strip("[]").upper()
        if ref in seen:
            continue
        seen.add(ref)
        c = ref_map.get(ref)
        if c is None:
            invalid.append(raw)
            continue
        valid.append(Citation(ref, c.title, c.section, c.path, c.doc_id, c.chunk_id))
    return valid, invalid

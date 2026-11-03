"""Find source files and fingerprint them.

doc_id       = sha256(source + ":" + relative path)  -> stable across content edits
content_hash = sha256(file bytes + sidecar bytes)    -> changes when content OR metadata changes
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ragkit.ingestion.loaders import SIDECAR_SUFFIX, SUPPORTED_EXTENSIONS, sidecar_path


@dataclass(frozen=True)
class SourceFile:
    doc_id: str
    source: str
    rel_path: str
    abs_path: Path
    content_hash: str


def make_doc_id(source: str, rel_path: str) -> str:
    return hashlib.sha256(f"{source}:{rel_path}".encode()).hexdigest()[:32]


def file_hash(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    side = sidecar_path(path)
    if side.exists():
        h.update(b"\x00sidecar\x00")
        h.update(side.read_bytes())
    return h.hexdigest()


def scan_directory(root: Path, source: str, max_bytes: int) -> tuple[list[SourceFile], list[str]]:
    """Return (files, warnings). Skips hidden files, symlinks escaping root, huge files."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    files: list[SourceFile] = []
    warnings: list[str] = []
    for p in sorted(root.rglob("*")):
        rel_parts = p.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if not p.is_file() or p.name.endswith(SIDECAR_SUFFIX):
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            warnings.append(f"skipped unsupported file: {p.relative_to(root)}")
            continue
        real = p.resolve()
        if not real.is_relative_to(root):
            warnings.append(f"skipped path escaping the source root: {p.relative_to(root)}")
            continue
        if real.stat().st_size > max_bytes:
            warnings.append(f"skipped file over size limit: {p.relative_to(root)}")
            continue
        rel = p.relative_to(root).as_posix()
        files.append(
            SourceFile(
                doc_id=make_doc_id(source, rel),
                source=source,
                rel_path=rel,
                abs_path=real,
                content_hash=file_hash(real),
            )
        )
    return files, warnings

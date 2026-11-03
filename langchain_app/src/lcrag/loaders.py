"""Document loading behind LangChain's `BaseLoader` interface.

Why not LangChain's built-in loaders (`PyPDFLoader`, `Docx2txtLoader`, `BSHTMLLoader`)?
- They lived in `langchain-community`, which LangChain sunset in May 2026 (repo archived,
  no replacement named). Depending on an unmaintained package in a security-sensitive path
  (parsing untrusted files) is a bad trade.
- `BSHTMLLoader` keeps HTML comments and `display:none` text: exactly where hidden prompt
  injections live. `Docx2txtLoader` drops heading styles, so DOCX sections are lost.

So the parsing itself is the custom app's hardened loader (magic-byte checks, zip-bomb guard,
safe YAML, hidden-HTML stripping, sidecar metadata). What LangChain contributes here is the
*interface*: a `BaseLoader` with `lazy_load()` yielding `Document`s, which plugs straight into
splitters and `index()`.

Every Document gets `metadata["source"]` = relative path. The indexing API uses it as the
group key for cleanup.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from ragkit.ingestion.loaders import SIDECAR_SUFFIX, SUPPORTED_EXTENSIONS, load_file
from ragkit.ingestion.normalize import normalize_text


class LumoraDirectoryLoader(BaseLoader):
    """One Document per file (whole file). Splitting happens later."""

    def __init__(self, root: Path, max_bytes: int = 5 * 1024 * 1024, max_pdf_pages: int = 200):
        self.root = root.resolve()
        self.max_bytes = max_bytes
        self.max_pdf_pages = max_pdf_pages
        self.errors: list[dict[str, str]] = []
        self.warnings: list[str] = []

    def list_files(self) -> list[tuple[str, Path]]:
        out = []
        for p in sorted(self.root.rglob("*")):
            rel = p.relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts) or not p.is_file():
                continue
            if p.name.endswith(SIDECAR_SUFFIX):
                continue
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                self.warnings.append(f"skipped unsupported file: {rel}")
                continue
            real = p.resolve()
            if not real.is_relative_to(self.root) or real.stat().st_size > self.max_bytes:
                self.warnings.append(f"skipped (outside root or too large): {rel}")
                continue
            out.append((rel.as_posix(), real))
        return out

    def lazy_load(self) -> Iterator[Document]:
        for rel, path in self.list_files():
            try:
                loaded = load_file(path, self.max_pdf_pages)
            except Exception as e:  # one bad file must not stop the run
                self.errors.append({"path": rel, "error": str(e)})
                continue
            norm = normalize_text(loaded.text)
            meta = dict(loaded.metadata)
            meta["title"] = normalize_text(meta["title"]).text
            meta["source"] = rel
            meta["hidden_chars_removed"] = norm.hidden_chars_removed
            yield Document(page_content=norm.text, metadata=meta)

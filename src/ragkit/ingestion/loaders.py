"""File loaders: turn MD/TXT/HTML/PDF/DOCX into Markdown-ish text + metadata.

Every format is converted to text with `#` headings so the chunker can treat them
uniformly. Metadata comes from YAML front matter (md/txt) or a sidecar file
`<name>.<ext>.meta.yaml` (any format).

Security:
- yaml.safe_load only (yaml.load can execute code).
- File type is checked by extension AND magic bytes.
- PDFs: page cap, encrypted PDFs rejected.
- DOCX is a zip: total uncompressed size and compression ratio are capped (zip bombs).
- HTML: scripts, styles, comments and hidden elements are dropped.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup, Comment, Tag

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"}
SIDECAR_SUFFIX = ".meta.yaml"
ALLOWED_META_KEYS = {"title", "department", "acl", "effective_date", "version", "doc_family"}
MAX_DOCX_UNCOMPRESSED = 50 * 1024 * 1024
MAX_ZIP_RATIO = 100


class LoaderError(ValueError):
    pass


@dataclass
class LoadedDocument:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + SIDECAR_SUFFIX)


def validate_file_bytes(ext: str, data: bytes) -> None:
    """Magic-byte check so a renamed executable can't pass as a PDF."""
    ext = ext.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoaderError(f"unsupported file type: {ext}")
    if ext == ".pdf" and not data.startswith(b"%PDF-"):
        raise LoaderError("file is not a valid PDF")
    if ext == ".docx" and not data.startswith(b"PK\x03\x04"):
        raise LoaderError("file is not a valid DOCX")
    if ext in {".md", ".markdown", ".txt", ".html", ".htm"}:
        if b"\x00" in data:
            raise LoaderError("text file contains NUL bytes")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise LoaderError("text file is not valid UTF-8") from e


def _clean_meta(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    meta: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in ALLOWED_META_KEYS or v is None:
            continue
        if isinstance(v, date):
            v = v.isoformat()
        meta[k] = str(v).strip()
    return meta


_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        raise LoaderError(f"invalid front matter: {e}") from e
    return _clean_meta(meta), text[m.end() :]


def _load_text(data: bytes) -> LoadedDocument:
    meta, body = _split_front_matter(data.decode("utf-8"))
    return LoadedDocument(text=body, metadata=meta)


_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0", re.I)


def _load_html(data: bytes) -> LoadedDocument:
    soup = BeautifulSoup(data.decode("utf-8"), "html.parser")
    for el in soup(["script", "style", "noscript", "template", "iframe", "object", "svg"]):
        el.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    for el in soup.find_all(True):
        if not isinstance(el, Tag) or el.attrs is None:
            continue
        style = el.get("style")
        if (
            el.has_attr("hidden")
            or el.get("aria-hidden") == "true"
            or (isinstance(style, str) and _HIDDEN_STYLE.search(style))
        ):
            el.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    body = soup.body or soup
    lines: list[str] = []
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td", "th"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name and el.name[0] == "h" and el.name[1:].isdigit():
            lines.append("#" * int(el.name[1:]) + " " + txt)
        elif el.name == "li":
            lines.append("- " + txt)
        else:
            lines.append(txt)
        lines.append("")
    meta = {"title": title} if title else {}
    return LoadedDocument(text="\n".join(lines), metadata=meta)


def _load_pdf(data: bytes, max_pages: int) -> LoadedDocument:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise LoaderError("encrypted PDFs are not supported")
    if len(reader.pages) > max_pages:
        raise LoaderError(f"PDF has {len(reader.pages)} pages (limit {max_pages})")
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    meta: dict[str, Any] = {}
    if reader.metadata and reader.metadata.title:
        meta["title"] = str(reader.metadata.title)
    return LoadedDocument(text="\n\n".join(p for p in pages if p), metadata=meta)


def _check_zip(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = sum(i.file_size for i in zf.infolist())
    except zipfile.BadZipFile as e:
        raise LoaderError("corrupt DOCX archive") from e
    if total > MAX_DOCX_UNCOMPRESSED or total > MAX_ZIP_RATIO * max(len(data), 1):
        raise LoaderError("DOCX expands too much (possible zip bomb)")


def _load_docx(data: bytes) -> LoadedDocument:
    import docx

    _check_zip(data)
    d = docx.Document(io.BytesIO(data))
    lines: list[str] = []
    for p in d.paragraphs:
        txt = p.text.strip()
        if not txt:
            continue
        style = (p.style.name if p.style is not None else "") or ""
        m = re.match(r"Heading (\d)", style)
        if m:
            lines.append("#" * int(m.group(1)) + " " + txt)
        elif style == "Title":
            lines.append("# " + txt)
        elif "List" in style:
            lines.append("- " + txt)
        else:
            lines.append(txt)
        lines.append("")
    for table in d.tables:
        for row in table.rows:
            lines.append(" | ".join(c.text.strip() for c in row.cells))
        lines.append("")
    meta: dict[str, Any] = {}
    if d.core_properties.title:
        meta["title"] = d.core_properties.title
    return LoadedDocument(text="\n".join(lines), metadata=meta)


def load_file(path: Path, max_pdf_pages: int = 200) -> LoadedDocument:
    ext = path.suffix.lower()
    data = path.read_bytes()
    validate_file_bytes(ext, data)
    if ext in {".md", ".markdown", ".txt"}:
        doc = _load_text(data)
    elif ext in {".html", ".htm"}:
        doc = _load_html(data)
    elif ext == ".pdf":
        doc = _load_pdf(data, max_pdf_pages)
    else:
        doc = _load_docx(data)

    side = sidecar_path(path)
    if side.exists():
        try:
            doc.metadata.update(_clean_meta(yaml.safe_load(side.read_text("utf-8"))))
        except yaml.YAMLError as e:
            raise LoaderError(f"invalid sidecar metadata: {e}") from e

    doc.metadata.setdefault("title", path.stem.replace("-", " ").replace("_", " ").title())
    doc.metadata.setdefault("acl", "public")
    doc.metadata.setdefault("department", "general")
    return doc

"""Chunking with LangChain's text splitters.

Two splitters chained, the standard LangChain recipe for structured docs:
1. `MarkdownHeaderTextSplitter` splits on `#` headings and records them in metadata.
2. `RecursiveCharacterTextSplitter.from_tiktoken_encoder` packs each section into
   ~450-token chunks with 60 tokens of overlap (paragraph -> line -> word fallbacks).

Then we add what LangChain doesn't: a breadcrumb header in the text ("Title > Section")
and an injection scan that flags chunks for quarantine.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ragkit.guardrails.injection_detect import detect_injection

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def split_documents(docs: list[Document], chunk_tokens: int, overlap: int) -> list[Document]:
    md = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    out: list[Document] = []
    for doc in docs:
        title = doc.metadata["title"]
        header_budget = 40  # tokens reserved for the breadcrumb header
        rec = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=max(chunk_tokens - header_budget, 50),
            chunk_overlap=overlap,
        )
        for section in md.split_text(doc.page_content):
            path = [section.metadata[k] for _, k in HEADERS if k in section.metadata]
            if path and path[0].strip().lower() == title.strip().lower():
                path = path[1:]  # drop an H1 that repeats the title
            section_name = " > ".join(path)
            header = " > ".join([title, *path])
            for piece in rec.split_text(section.page_content):
                if not piece.strip():
                    continue
                text = f"{header}\n\n{piece}"
                verdict = detect_injection(text)
                meta = {k: v for k, v in doc.metadata.items()}
                meta.update(
                    section=section_name,
                    flagged=verdict.is_injection(),
                    flag_reasons=verdict.reasons,
                )
                out.append(Document(page_content=text, metadata=meta))
    return out

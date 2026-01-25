from __future__ import annotations

import pymupdf
import pymupdf.layout  # noqa: F401
import pymupdf4llm


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def pdf_bytes_to_markdown(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    with doc:
        markdown = pymupdf4llm.to_markdown(doc)

    return _normalize_newlines(str(markdown)).strip()

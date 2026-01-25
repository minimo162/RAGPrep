from __future__ import annotations

from typing import cast

import pytest

from ragprep import pymupdf4llm_markdown


def _make_pdf_bytes() -> bytes:
    import fitz

    doc = fitz.open()
    doc.new_page()
    return cast(bytes, doc.tobytes())


def test_pdf_bytes_to_markdown_normalizes_newlines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pymupdf4llm_markdown.pymupdf4llm,
        "to_markdown",
        lambda _doc: "line1\r\nline2\r",
    )

    assert pymupdf4llm_markdown.pdf_bytes_to_markdown(_make_pdf_bytes()) == "line1\nline2"


def test_pdf_bytes_to_markdown_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pymupdf4llm_markdown.pdf_bytes_to_markdown(b"")

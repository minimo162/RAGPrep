from __future__ import annotations

from typing import cast

import pytest

from ragprep.pipeline import pdf_to_markdown


def _make_pdf_bytes(page_texts: list[str]) -> bytes:
    import fitz

    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    return cast(bytes, doc.tobytes())


def _squash_ws(text: str) -> str:
    return "".join(text.split())


def test_pdf_to_markdown_e2e_contains_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_PDF_BACKEND", "pymupdf")
    pdf_bytes = _make_pdf_bytes(["Hello E2E 1", "Hello E2E 2"])

    markdown = pdf_to_markdown(pdf_bytes)
    squashed = _squash_ws(markdown)

    assert _squash_ws("Hello E2E 1") in squashed
    assert _squash_ws("Hello E2E 2") in squashed


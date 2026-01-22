from __future__ import annotations

from typing import cast

import pytest

from ragprep.pipeline import pdf_to_markdown


def _make_pdf_bytes(page_count: int) -> bytes:
    import fitz

    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 72), f"Hello {i + 1}")
    return cast(bytes, doc.tobytes())


def test_pdf_to_markdown_concatenates_pages_and_normalizes_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_bytes = _make_pdf_bytes(page_count=2)

    calls: dict[str, int] = {"n": 0}

    def fake_ocr_image(_image: object) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "page1\r\n"
        if calls["n"] == 2:
            return "page2\r"
        raise AssertionError("Unexpected extra page")

    monkeypatch.setattr("ragprep.pipeline.ocr_image", fake_ocr_image)

    assert pdf_to_markdown(pdf_bytes) == "page1\n\npage2"


def test_pdf_to_markdown_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pdf_to_markdown(b"")

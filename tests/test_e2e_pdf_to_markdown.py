from __future__ import annotations

import pytest

from ragprep.pipeline import pdf_to_markdown


def _squash_ws(text: str) -> str:
    return "".join(text.split())


def test_pdf_to_markdown_e2e_contains_text(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded_pages = ["BASE64_PAGE_1", "BASE64_PAGE_2"]

    def _fake_iter_pdf_page_png_base64(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, object]:
        return 2, iter(encoded_pages)

    monkeypatch.setattr(
        "ragprep.pdf_render.iter_pdf_page_png_base64",
        _fake_iter_pdf_page_png_base64,
    )

    outputs = iter(["Hello E2E 1", "Hello E2E 2"])

    def _fake_ocr_image(_encoded: str) -> str:
        return next(outputs)

    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image_base64", _fake_ocr_image)

    markdown = pdf_to_markdown(b"%PDF")
    squashed = _squash_ws(markdown)

    assert _squash_ws("Hello E2E 1") in squashed
    assert _squash_ws("Hello E2E 2") in squashed


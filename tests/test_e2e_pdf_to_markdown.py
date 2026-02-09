from __future__ import annotations

import pytest

from ragprep.pipeline import pdf_to_markdown


def _squash_ws(text: str) -> str:
    return "".join(text.split())


def test_pdf_to_markdown_e2e_contains_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)

    def _fake_iter_pdf_page_png_base64(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, object]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 2, iter(["BASE64_PAGE_1", "BASE64_PAGE_2"])

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_page_png_base64", _fake_iter_pdf_page_png_base64)

    outputs = iter(["Hello E2E 1", "Hello E2E 2"])

    def _fake_glm(_encoded: str, *, settings: object) -> str:
        _ = settings
        return next(outputs)

    monkeypatch.setattr("ragprep.ocr.glm_ocr.ocr_image_base64", _fake_glm)

    markdown = pdf_to_markdown(b"%PDF")
    squashed = _squash_ws(markdown)

    assert _squash_ws("Hello E2E 1") in squashed
    assert _squash_ws("Hello E2E 2") in squashed


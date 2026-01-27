from __future__ import annotations

import pytest

import ragprep.pipeline as pipeline
from ragprep.config import get_settings


def _patch_iter_pdf_page_png_base64(
    monkeypatch: pytest.MonkeyPatch, page_count: int
) -> None:
    encoded_pages = [f"BASE64_PAGE_{i}" for i in range(1, page_count + 1)]

    def _fake_iter_pdf_page_png_base64(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, object]:
        return page_count, iter(encoded_pages)

    monkeypatch.setattr(
        "ragprep.pdf_render.iter_pdf_page_png_base64",
        _fake_iter_pdf_page_png_base64,
    )


def test_pdf_to_markdown_lightonocr_normalizes_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=1)
    monkeypatch.setattr(
        "ragprep.ocr.lightonocr.ocr_image_base64",
        lambda _encoded: "line1\r\nline2\r",
    )

    settings = get_settings()
    result = pipeline._pdf_to_markdown_lightonocr(b"%PDF", settings=settings)
    assert result == "line1\nline2"


def test_pdf_to_markdown_lightonocr_concatenates_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=2)

    outputs = iter(["PAGE1", "PAGE2"])

    def _fake_ocr_image(_encoded: str) -> str:
        return next(outputs)

    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image_base64", _fake_ocr_image)

    settings = get_settings()
    result = pipeline._pdf_to_markdown_lightonocr(b"%PDF", settings=settings)
    assert result == "PAGE1\n\nPAGE2"


def test_pdf_to_markdown_lightonocr_calls_on_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=2)

    outputs = iter(["PAGE1\r", " PAGE2 "])

    def _fake_ocr_image(_encoded: str) -> str:
        return next(outputs)

    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image_base64", _fake_ocr_image)

    pages: list[tuple[int, str]] = []

    def on_page(page_index: int, markdown: str) -> None:
        pages.append((page_index, markdown))

    settings = get_settings()
    result = pipeline._pdf_to_markdown_lightonocr(
        b"%PDF",
        settings=settings,
        on_page=on_page,
    )
    assert result == "PAGE1\n\nPAGE2"
    assert pages == [(1, "PAGE1"), (2, "PAGE2")]


def test_pdf_to_json_lightonocr_error_includes_root_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=1)

    def _raise(_encoded: str) -> str:
        raise RuntimeError("GGUF model file not found: missing.gguf")

    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image_base64", _raise)

    settings = get_settings()
    with pytest.raises(RuntimeError) as excinfo:
        pipeline._pdf_to_markdown_lightonocr(b"%PDF", settings=settings)
    message = str(excinfo.value)
    assert "LightOnOCR failed on page 1" in message
    assert "GGUF model file not found" in message

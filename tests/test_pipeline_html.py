from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from ragprep.pipeline import PdfToHtmlProgress, ProgressPhase, pdf_to_html


def test_pdf_to_html_reports_progress_and_renders_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 2, iter(["PAGE1", "PAGE2"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["# Heading\n\nParagraph", "| A | B |\n|---|---|\n| 1 | 2 |"],
    )

    def _fake_ocr(image_b64: str, *, settings: object) -> str:
        _ = settings
        if image_b64 == "PAGE1":
            return "# Heading\n\nParagraph"
        return "| A | B |\n|---|---|\n| 1 | 2 |"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    updates: list[PdfToHtmlProgress] = []
    pages: list[int] = []

    html = pdf_to_html(
        b"%PDF",
        full_document=False,
        on_progress=updates.append,
        on_page=lambda page_index, _html: pages.append(page_index),
    )

    assert "<h1>Heading</h1>" in html
    assert "<p>Paragraph</p>" in html
    assert '<table data-kind="table">' in html
    assert pages == [1, 2]
    assert [(u.phase, u.current, u.total) for u in updates] == [
        (ProgressPhase.rendering, 0, 2),
        (ProgressPhase.rendering, 1, 2),
        (ProgressPhase.rendering, 2, 2),
        (ProgressPhase.done, 2, 2),
    ]


def test_pdf_to_html_keeps_on_page_order_under_parallel_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_PAGE_CONCURRENCY", "3")

    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 3, iter(["P1", "P2", "P3"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["one", "two", "three"],
    )

    def _fake_ocr(image_b64: str, *, settings: object) -> str:
        _ = settings
        if image_b64 == "P1":
            time.sleep(0.15)
            return "one"
        if image_b64 == "P2":
            return "two"
        time.sleep(0.05)
        return "three"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    pages: list[int] = []
    _ = pdf_to_html(
        b"%PDF",
        full_document=False,
        on_page=lambda page_index, _html: pages.append(page_index),
    )
    assert pages == [1, 2, 3]


def test_pdf_to_html_aborts_when_ocr_page_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 2, iter(["A", "B"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["A", "B"],
    )

    def _fake_ocr(image_b64: str, *, settings: object) -> str:
        _ = settings
        if image_b64 == "B":
            raise RuntimeError("upstream failure")
        return "A"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    with pytest.raises(RuntimeError, match="LightOn OCR failed"):
        _ = pdf_to_html(b"%PDF", full_document=False)


def test_pdf_to_html_uses_lighton_render_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_RENDER_DPI", "123")
    monkeypatch.setenv("RAGPREP_LIGHTON_RENDER_MAX_EDGE", "456")

    captured: dict[str, int] = {}

    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        captured["dpi"] = int(dpi or 0)
        captured["max_edge"] = int(max_edge or 0)
        _ = max_pages, max_bytes
        return 1, iter(["P1"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["p1"])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "p1",
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "p1" in html
    assert captured["dpi"] == 123
    assert captured["max_edge"] == 456

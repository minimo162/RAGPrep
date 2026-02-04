from __future__ import annotations

from collections.abc import Iterator

import pytest
from PIL import Image

from ragprep.pdf_text import Span
from ragprep.pipeline import PdfToHtmlProgress, ProgressPhase, pdf_to_html


def test_pdf_to_html_reports_progress_and_renders_html(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_iter_pdf_images(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[int, Iterator[Image.Image]]:
        def generate() -> Iterator[Image.Image]:
            yield Image.new("RGB", (10, 10), color=(255, 255, 255))
            yield Image.new("RGB", (10, 10), color=(255, 255, 255))

        return 2, generate()

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)

    spans_by_page = [
        [Span(x0=0, y0=0, x1=10, y1=10, text="Heading")],
        [Span(x0=0, y0=150, x1=10, y1=160, text="Body")],
    ]
    monkeypatch.setattr("ragprep.pdf_text.extract_pymupdf_page_spans", lambda _pdf: spans_by_page)
    monkeypatch.setattr(
        "ragprep.pdf_text.extract_pymupdf_page_sizes",
        lambda _pdf: [(1000.0, 1000.0), (1000.0, 1000.0)],
    )

    def _fake_analyze_layout(_image_b64: str, *, settings: object) -> dict[str, object]:
        _ = settings
        return {
            "schema_version": "v1",
            "elements": [
                {"bbox": (0.0, 0.0, 10.0, 1.0), "label": "title", "score": 0.9},
                {"bbox": (0.0, 1.0, 10.0, 3.0), "label": "text", "score": 0.9},
            ],
            "raw": "{}",
        }

    monkeypatch.setattr(
        "ragprep.layout.glm_doclayout.analyze_layout_image_base64",
        _fake_analyze_layout,
    )

    updates: list[PdfToHtmlProgress] = []

    def on_progress(update: PdfToHtmlProgress) -> None:
        updates.append(update)

    html = pdf_to_html(b"%PDF", full_document=False, on_progress=on_progress)
    assert "<section data-page=\"1\">" in html
    assert "<h1>Heading</h1>" in html
    assert "<p>Body</p>" in html

    assert [(u.phase, u.current, u.total) for u in updates] == [
        (ProgressPhase.rendering, 0, 2),
        (ProgressPhase.rendering, 1, 2),
        (ProgressPhase.rendering, 2, 2),
        (ProgressPhase.done, 2, 2),
    ]


def test_pdf_to_html_requires_layout_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_iter_pdf_images(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[int, Iterator[Image.Image]]:
        def generate() -> Iterator[Image.Image]:
            yield Image.new("RGB", (10, 10), color=(255, 255, 255))

        return 1, generate()

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)
    monkeypatch.setattr(
        "ragprep.pdf_text.extract_pymupdf_page_spans",
        lambda _pdf: [[Span(x0=0, y0=0, x1=10, y1=10, text="X")]],
    )
    monkeypatch.setattr(
        "ragprep.pdf_text.extract_pymupdf_page_sizes",
        lambda _pdf: [(1000.0, 1000.0)],
    )

    def _raise_layout(_image_b64: str, *, settings: object) -> dict[str, object]:
        _ = settings
        raise RuntimeError("Layout analysis currently requires RAGPREP_LAYOUT_MODE=server.")

    monkeypatch.setattr("ragprep.layout.glm_doclayout.analyze_layout_image_base64", _raise_layout)

    with pytest.raises(RuntimeError, match="Layout analysis currently requires"):
        _ = pdf_to_html(b"%PDF", full_document=False)

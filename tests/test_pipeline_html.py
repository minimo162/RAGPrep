from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from threading import Barrier, BrokenBarrierError, Lock

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


def test_pdf_to_html_pipelines_layout_requests_in_server_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    monkeypatch.setenv("RAGPREP_LAYOUT_CONCURRENCY", "2")

    def _fake_iter_pdf_images(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[int, Iterator[Image.Image]]:
        def generate() -> Iterator[Image.Image]:
            yield Image.new("RGB", (10, 10), color=(255, 255, 255))
            yield Image.new("RGB", (10, 10), color=(255, 255, 255))
            yield Image.new("RGB", (10, 10), color=(255, 255, 255))

        return 3, generate()

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)

    spans_by_page = [
        [Span(x0=0, y0=0, x1=10, y1=10, text="A")],
        [Span(x0=0, y0=0, x1=10, y1=10, text="B")],
        [Span(x0=0, y0=0, x1=10, y1=10, text="C")],
    ]
    monkeypatch.setattr("ragprep.pdf_text.extract_pymupdf_page_spans", lambda _pdf: spans_by_page)
    monkeypatch.setattr(
        "ragprep.pdf_text.extract_pymupdf_page_sizes",
        lambda _pdf: [(1000.0, 1000.0), (1000.0, 1000.0), (1000.0, 1000.0)],
    )

    barrier = Barrier(2)
    call_lock = Lock()
    call_count = 0

    def _fake_analyze_layout(_image_b64: str, *, settings: object) -> dict[str, object]:
        nonlocal call_count
        _ = settings
        with call_lock:
            call_count += 1
            index = call_count
        if index in {1, 2}:
            try:
                barrier.wait(timeout=2.0)
            except BrokenBarrierError as exc:  # pragma: no cover
                raise AssertionError("expected concurrent layout requests") from exc
        return {
            "schema_version": "v1",
            "elements": [{"bbox": (0.0, 0.0, 10.0, 10.0), "label": "text", "score": 0.9}],
            "raw": "{}",
        }

    monkeypatch.setattr(
        "ragprep.layout.glm_doclayout.analyze_layout_image_base64",
        _fake_analyze_layout,
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "<section data-page=\"1\">" in html
    assert "<section data-page=\"2\">" in html
    assert "<section data-page=\"3\">" in html
    assert call_count == 3


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


def test_pdf_to_html_uses_layout_render_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    monkeypatch.setenv("RAGPREP_LAYOUT_RENDER_DPI", "123")
    monkeypatch.setenv("RAGPREP_LAYOUT_RENDER_MAX_EDGE", "456")
    monkeypatch.setenv("RAGPREP_LAYOUT_CONCURRENCY", "1")

    captured: dict[str, int] = {}

    def _fake_iter_pdf_images(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[Image.Image]]:
        captured["dpi"] = int(dpi or 0)
        captured["max_edge"] = int(max_edge or 0)
        _ = max_pages, max_bytes

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

    def _fake_analyze_layout(_image_b64: str, *, settings: object) -> dict[str, object]:
        _ = settings
        return {
            "schema_version": "v1",
            "elements": [{"bbox": (0.0, 0.0, 10.0, 10.0), "label": "text", "score": 0.9}],
            "raw": "{}",
        }

    monkeypatch.setattr(
        "ragprep.layout.glm_doclayout.analyze_layout_image_base64",
        _fake_analyze_layout,
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "X" in html
    assert captured["dpi"] == 123
    assert captured["max_edge"] == 456


def test_pdf_to_html_layout_render_defaults_are_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    monkeypatch.setenv("RAGPREP_LAYOUT_CONCURRENCY", "1")
    monkeypatch.setenv("RAGPREP_RENDER_DPI", "999")
    monkeypatch.setenv("RAGPREP_RENDER_MAX_EDGE", "999")

    captured: dict[str, int] = {}

    def _fake_iter_pdf_images(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[Image.Image]]:
        captured["dpi"] = int(dpi or 0)
        captured["max_edge"] = int(max_edge or 0)
        _ = max_pages, max_bytes

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

    def _fake_analyze_layout(_image_b64: str, *, settings: object) -> dict[str, object]:
        _ = settings
        return {
            "schema_version": "v1",
            "elements": [{"bbox": (0.0, 0.0, 10.0, 10.0), "label": "text", "score": 0.9}],
            "raw": "{}",
        }

    monkeypatch.setattr(
        "ragprep.layout.glm_doclayout.analyze_layout_image_base64",
        _fake_analyze_layout,
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "X" in html
    assert captured["dpi"] == 250
    assert captured["max_edge"] == 1024


def test_pdf_to_html_adaptive_layout_rerenders_on_empty_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    monkeypatch.setenv("RAGPREP_LAYOUT_CONCURRENCY", "1")
    monkeypatch.setenv("RAGPREP_LAYOUT_RENDER_AUTO", "1")
    monkeypatch.setenv("RAGPREP_LAYOUT_RENDER_AUTO_SMALL_DPI", "200")
    monkeypatch.setenv("RAGPREP_LAYOUT_RENDER_AUTO_SMALL_MAX_EDGE", "1024")
    monkeypatch.setenv("RAGPREP_LAYOUT_RENDER_DPI", "400")
    monkeypatch.setenv("RAGPREP_LAYOUT_RENDER_MAX_EDGE", "1540")

    captured: dict[str, object] = {"iter": None, "rerender": None}
    calls = 0

    def _fake_iter_pdf_images(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[Image.Image]]:
        captured["iter"] = (int(dpi or 0), int(max_edge or 0))
        _ = max_pages, max_bytes

        def generate() -> Iterator[Image.Image]:
            yield Image.new("RGB", (500, 700), color=(255, 255, 255))

        return 1, generate()

    def _fake_render_pdf_page_image(
        _pdf_bytes: bytes,
        *,
        page_index: int,
        dpi: int,
        max_edge: int,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> Image.Image:
        captured["rerender"] = (page_index, dpi, max_edge)
        _ = _pdf_bytes, max_pages, max_bytes
        return Image.new("RGB", (1000, 1400), color=(255, 255, 255))

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)
    monkeypatch.setattr("ragprep.pdf_render.render_pdf_page_image", _fake_render_pdf_page_image)
    monkeypatch.setattr(
        "ragprep.pdf_text.extract_pymupdf_page_spans",
        lambda _pdf: [[Span(x0=0, y0=0, x1=10, y1=10, text="X")]],
    )
    monkeypatch.setattr(
        "ragprep.pdf_text.extract_pymupdf_page_sizes",
        lambda _pdf: [(1000.0, 1000.0)],
    )

    def _fake_analyze_layout(image_b64: str, *, settings: object) -> dict[str, object]:
        nonlocal calls
        _ = settings
        calls += 1

        png_bytes = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(png_bytes)) as img:
            w, _h = img.size

        if w <= 600:
            return {"schema_version": "v1", "elements": [], "raw": "{}"}
        return {
            "schema_version": "v1",
            "elements": [{"bbox": (0.0, 0.0, 10.0, 10.0), "label": "text", "score": 0.9}],
            "raw": "{}",
        }

    monkeypatch.setattr(
        "ragprep.layout.glm_doclayout.analyze_layout_image_base64",
        _fake_analyze_layout,
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "X" in html
    assert captured["iter"] == (200, 1024)
    assert captured["rerender"] == (0, 400, 1540)
    assert calls == 2

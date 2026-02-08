from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PIL import Image

from ragprep.pipeline import (
    PdfToJsonProgress,
    PdfToMarkdownProgress,
    ProgressPhase,
    pdf_to_json,
    pdf_to_markdown,
)


def _patch_lighton_page_refs(monkeypatch: pytest.MonkeyPatch, page_count: int) -> None:
    monkeypatch.setattr(
        "ragprep.pdf_text.extract_pymupdf_page_sizes",
        lambda _pdf: [(1000.0, 1000.0)] * page_count,
    )
    monkeypatch.setattr(
        "ragprep.pdf_text.extract_pymupdf_page_words",
        lambda _pdf: [[] for _ in range(page_count)],
    )


def _patch_iter_pdf_images(monkeypatch: pytest.MonkeyPatch, page_count: int) -> None:
    def _fake_iter_pdf_images(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[Image.Image]]:
        _ = dpi, max_edge, max_pages, max_bytes

        def _generate() -> Iterator[Image.Image]:
            for _idx in range(page_count):
                yield Image.new("RGB", (1000, 1000), color=(255, 255, 255))

        return page_count, _generate()

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)


def _sequence_lighton_pages(texts: list[str]) -> Callable[..., dict[str, object]]:
    iterator = iter(texts)

    def _fake_lighton(_encoded: str, *, settings: object) -> dict[str, object]:
        _ = settings
        text = next(iterator)
        return {
            "schema_version": "v1",
            "elements": [],
            "lines": [{"bbox": (80.0, 80.0, 220.0, 120.0), "text": text}],
            "raw": "{}",
        }

    return _fake_lighton


def _patch_iter_pdf_page_png_base64(
    monkeypatch: pytest.MonkeyPatch, page_count: int
) -> list[str]:
    encoded_pages = [f"BASE64_PAGE_{i}" for i in range(1, page_count + 1)]

    def _fake_iter_pdf_page_png_base64(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, object]:
        _ = dpi, max_edge, max_pages, max_bytes
        return page_count, iter(encoded_pages)

    monkeypatch.setattr(
        "ragprep.pdf_render.iter_pdf_page_png_base64",
        _fake_iter_pdf_page_png_base64,
    )
    return encoded_pages


def _sequence_glm_texts(texts: list[str]) -> Callable[..., str]:
    iterator = iter(texts)

    def _fake_glm(_encoded: str, *, settings: object) -> str:
        _ = settings
        return next(iterator)

    return _fake_glm


def test_pdf_to_markdown_normalizes_newlines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    _patch_lighton_page_refs(monkeypatch, page_count=2)
    _patch_iter_pdf_images(monkeypatch, page_count=2)
    monkeypatch.setattr(
        "ragprep.ocr.lighton_ocr.analyze_ocr_layout_image_base64",
        _sequence_lighton_pages(["line1\r\nline2\r", "line3\r"]),
    )

    assert pdf_to_markdown(b"%PDF") == "line1 line2\n\nline3"


def test_pdf_to_markdown_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pdf_to_markdown(b"")


def test_pdf_to_markdown_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    _patch_lighton_page_refs(monkeypatch, page_count=3)
    _patch_iter_pdf_images(monkeypatch, page_count=3)
    monkeypatch.setattr(
        "ragprep.ocr.lighton_ocr.analyze_ocr_layout_image_base64",
        _sequence_lighton_pages(["ok", "ok", "ok"]),
    )

    updates: list[PdfToMarkdownProgress] = []

    def on_progress(update: PdfToMarkdownProgress) -> None:
        updates.append(update)

    assert pdf_to_markdown(b"%PDF", on_progress=on_progress) == "ok\n\nok\n\nok"
    assert [(u.phase, u.current, u.total) for u in updates] == [
        (ProgressPhase.rendering, 0, 3),
        (ProgressPhase.rendering, 1, 3),
        (ProgressPhase.rendering, 2, 3),
        (ProgressPhase.rendering, 3, 3),
        (ProgressPhase.done, 3, 3),
    ]


def test_pdf_to_markdown_writes_document_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    _patch_lighton_page_refs(monkeypatch, page_count=1)
    _patch_iter_pdf_images(monkeypatch, page_count=1)
    monkeypatch.setattr(
        "ragprep.ocr.lighton_ocr.analyze_ocr_layout_image_base64",
        _sequence_lighton_pages(["hello"]),
    )

    out_dir = tmp_path / "artifacts"
    result = pdf_to_markdown(b"%PDF", page_output_dir=out_dir)
    assert result == "hello"

    artifact = out_dir / "document.md"
    assert artifact.exists()
    assert artifact.read_text(encoding="utf-8") == "hello\n"


def test_pdf_to_markdown_invalid_pdf_propagates_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    _patch_lighton_page_refs(monkeypatch, page_count=1)

    def _fake_iter_pdf_images(*_args: object, **_kwargs: object) -> tuple[int, object]:
        raise ValueError("Invalid PDF data")

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)

    with pytest.raises(ValueError, match="Invalid PDF data"):
        pdf_to_markdown(b"not a pdf")


def test_pdf_to_json_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pdf_to_json(b"")


def test_pdf_to_json_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    _patch_lighton_page_refs(monkeypatch, page_count=2)
    _patch_iter_pdf_images(monkeypatch, page_count=2)
    monkeypatch.setattr(
        "ragprep.ocr.lighton_ocr.analyze_ocr_layout_image_base64",
        _sequence_lighton_pages(["PAGE1", "PAGE2"]),
    )

    updates: list[PdfToJsonProgress] = []

    def on_progress(update: PdfToJsonProgress) -> None:
        updates.append(update)

    markdown = pdf_to_json(b"%PDF", on_progress=on_progress)
    assert markdown == "PAGE1\n\nPAGE2"
    assert [(u.phase, u.current, u.total) for u in updates] == [
        (ProgressPhase.rendering, 0, 2),
        (ProgressPhase.rendering, 1, 2),
        (ProgressPhase.rendering, 2, 2),
        (ProgressPhase.done, 2, 2),
    ]


def test_pdf_to_json_calls_on_page_per_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    _patch_lighton_page_refs(monkeypatch, page_count=2)
    _patch_iter_pdf_images(monkeypatch, page_count=2)
    monkeypatch.setattr(
        "ragprep.ocr.lighton_ocr.analyze_ocr_layout_image_base64",
        _sequence_lighton_pages(["PAGE1\r\n", " PAGE2 "]),
    )

    pages: list[tuple[int, str]] = []

    def on_page(page_index: int, markdown: str) -> None:
        pages.append((page_index, markdown))

    markdown = pdf_to_json(b"%PDF", on_page=on_page)
    assert markdown == "PAGE1\n\nPAGE2"
    assert pages == [(1, "PAGE1"), (2, "PAGE2")]


def test_pdf_to_json_writes_document_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    _patch_lighton_page_refs(monkeypatch, page_count=1)
    _patch_iter_pdf_images(monkeypatch, page_count=1)
    monkeypatch.setattr(
        "ragprep.ocr.lighton_ocr.analyze_ocr_layout_image_base64",
        _sequence_lighton_pages(["only"]),
    )

    out_dir = tmp_path / "artifacts"
    result = pdf_to_json(b"%PDF", page_output_dir=out_dir)
    assert result == "only"

    json_artifact = out_dir / "document.json"
    assert not json_artifact.exists()

    md_artifact = out_dir / "document.md"
    assert md_artifact.exists()
    assert md_artifact.read_text(encoding="utf-8") == "only\n"


def test_pdf_to_json_invalid_pdf_propagates_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    _patch_lighton_page_refs(monkeypatch, page_count=1)

    def _fake_iter_pdf_images(*_args: object, **_kwargs: object) -> tuple[int, object]:
        raise ValueError("Invalid PDF data")

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)

    with pytest.raises(ValueError, match="Invalid PDF data"):
        pdf_to_json(b"not a pdf")


def test_pdf_to_markdown_default_uses_lighton_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    _patch_lighton_page_refs(monkeypatch, page_count=1)
    _patch_iter_pdf_images(monkeypatch, page_count=1)

    monkeypatch.setattr(
        "ragprep.ocr.glm_ocr.ocr_image_base64",
        lambda _enc, *, settings: (_ for _ in ()).throw(AssertionError("GLM should not be used")),
    )
    monkeypatch.setattr(
        "ragprep.ocr.lighton_ocr.analyze_ocr_layout_image_base64",
        _sequence_lighton_pages(["LIGHTON"]),
    )

    assert pdf_to_markdown(b"%PDF") == "LIGHTON"


def test_pdf_to_markdown_uses_glm_backend_when_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_PDF_BACKEND", "glm-ocr")
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=2)
    monkeypatch.setattr(
        "ragprep.ocr.glm_ocr.ocr_image_base64",
        _sequence_glm_texts(["PAGE1\r", " PAGE2 "]),
    )

    assert pdf_to_markdown(b"%PDF") == "PAGE1\n\nPAGE2"

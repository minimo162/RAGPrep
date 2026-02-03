from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ragprep.pipeline import (
    PdfToJsonProgress,
    PdfToMarkdownProgress,
    ProgressPhase,
    pdf_to_json,
    pdf_to_markdown,
)


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
        return page_count, iter(encoded_pages)

    monkeypatch.setattr(
        "ragprep.pdf_render.iter_pdf_page_png_base64",
        _fake_iter_pdf_page_png_base64,
    )
    return encoded_pages


def _sequence_texts(texts: list[str]) -> Callable[..., str]:
    iterator = iter(texts)

    def _fake_ocr_image(_encoded: str, *, settings: object) -> str:
        _ = settings
        return next(iterator)

    return _fake_ocr_image


def test_pdf_to_markdown_normalizes_newlines(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=2)
    monkeypatch.setattr(
        "ragprep.ocr.glm_ocr.ocr_image_base64",
        _sequence_texts(["line1\r\nline2\r", "line3\r"]),
    )

    assert pdf_to_markdown(b"%PDF") == "line1\nline2\n\nline3"


def test_pdf_to_markdown_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pdf_to_markdown(b"")


def test_pdf_to_markdown_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=3)
    monkeypatch.setattr("ragprep.ocr.glm_ocr.ocr_image_base64", lambda _enc, *, settings: "ok")

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
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=1)
    monkeypatch.setattr("ragprep.ocr.glm_ocr.ocr_image_base64", lambda _enc, *, settings: "hello")

    out_dir = tmp_path / "artifacts"
    result = pdf_to_markdown(b"%PDF", page_output_dir=out_dir)
    assert result == "hello"

    artifact = out_dir / "document.md"
    assert artifact.exists()
    assert artifact.read_text(encoding="utf-8") == "hello\n"


def test_pdf_to_markdown_invalid_pdf_propagates_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_iter_pdf_page_png_base64(*_args: object, **_kwargs: object) -> tuple[int, object]:
        raise ValueError("Invalid PDF data")

    monkeypatch.setattr(
        "ragprep.pdf_render.iter_pdf_page_png_base64",
        _fake_iter_pdf_page_png_base64,
    )

    with pytest.raises(ValueError, match="Invalid PDF data"):
        pdf_to_markdown(b"not a pdf")


def test_pdf_to_json_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pdf_to_json(b"")


def test_pdf_to_json_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=2)
    monkeypatch.setattr(
        "ragprep.ocr.glm_ocr.ocr_image_base64",
        _sequence_texts(["PAGE1", "PAGE2"]),
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
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=2)
    monkeypatch.setattr(
        "ragprep.ocr.glm_ocr.ocr_image_base64",
        _sequence_texts(["PAGE1\r\n", " PAGE2 "]),
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
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=1)
    monkeypatch.setattr("ragprep.ocr.glm_ocr.ocr_image_base64", lambda _enc, *, settings: "only")

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
    def _fake_iter_pdf_page_png_base64(*_args: object, **_kwargs: object) -> tuple[int, object]:
        raise ValueError("Invalid PDF data")

    monkeypatch.setattr(
        "ragprep.pdf_render.iter_pdf_page_png_base64",
        _fake_iter_pdf_page_png_base64,
    )

    with pytest.raises(ValueError, match="Invalid PDF data"):
        pdf_to_json(b"not a pdf")


def test_pdf_to_markdown_uses_glm_ocr_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_iter_pdf_page_png_base64(monkeypatch, page_count=2)

    outputs = iter(["PAGE1\r", " PAGE2 "])

    def _fake_glm(_encoded: str, *, settings: object) -> str:
        return next(outputs)

    monkeypatch.setattr("ragprep.ocr.glm_ocr.ocr_image_base64", _fake_glm)

    assert pdf_to_markdown(b"%PDF") == "PAGE1\n\nPAGE2"

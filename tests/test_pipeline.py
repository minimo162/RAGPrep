from __future__ import annotations

import json
from collections.abc import Callable
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


def _make_images(count: int) -> list[Image.Image]:
    return [Image.new("RGB", (2, 2)) for _ in range(count)]


def _patch_iter_pdf_images(
    monkeypatch: pytest.MonkeyPatch, page_count: int
) -> list[Image.Image]:
    images = _make_images(page_count)

    def _fake_iter_pdf_images(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, object]:
        return page_count, iter(images)

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)
    return images


def _sequence_texts(texts: list[str]) -> Callable[[Image.Image], str]:
    iterator = iter(texts)

    def _fake_ocr_image(_image: Image.Image) -> str:
        return next(iterator)

    return _fake_ocr_image


def test_pdf_to_markdown_normalizes_newlines(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_iter_pdf_images(monkeypatch, page_count=2)
    monkeypatch.setattr(
        "ragprep.ocr.lightonocr.ocr_image",
        _sequence_texts(["line1\r\nline2\r", "line3\r"]),
    )

    assert pdf_to_markdown(b"%PDF") == "line1\nline2\n\nline3"


def test_pdf_to_markdown_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pdf_to_markdown(b"")


def test_pdf_to_markdown_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_iter_pdf_images(monkeypatch, page_count=3)
    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image", lambda _img: "ok")

    updates: list[PdfToMarkdownProgress] = []

    def on_progress(update: PdfToMarkdownProgress) -> None:
        updates.append(update)

    assert pdf_to_markdown(b"%PDF", on_progress=on_progress) == "ok\n\nok\n\nok"
    assert [(u.phase, u.current, u.total) for u in updates] == [
        (ProgressPhase.rendering, 0, 3),
        (ProgressPhase.done, 3, 3),
    ]


def test_pdf_to_markdown_writes_document_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_iter_pdf_images(monkeypatch, page_count=1)
    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image", lambda _img: "hello")

    out_dir = tmp_path / "artifacts"
    result = pdf_to_markdown(b"%PDF", page_output_dir=out_dir)
    assert result == "hello"

    artifact = out_dir / "document.md"
    assert artifact.exists()
    assert artifact.read_text(encoding="utf-8") == "hello\n"


def test_pdf_to_markdown_invalid_pdf_propagates_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_iter_pdf_images(*_args: object, **_kwargs: object) -> tuple[int, object]:
        raise ValueError("Invalid PDF data")

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)

    with pytest.raises(ValueError, match="Invalid PDF data"):
        pdf_to_markdown(b"not a pdf")


def test_pdf_to_json_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pdf_to_json(b"")


def test_pdf_to_json_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_iter_pdf_images(monkeypatch, page_count=2)
    monkeypatch.setattr(
        "ragprep.ocr.lightonocr.ocr_image",
        _sequence_texts(["PAGE1", "PAGE2"]),
    )

    updates: list[PdfToJsonProgress] = []

    def on_progress(update: PdfToJsonProgress) -> None:
        updates.append(update)

    payload = json.loads(pdf_to_json(b"%PDF", on_progress=on_progress))
    assert payload["meta"]["backend"] == "lightonocr"
    assert payload["meta"]["page_count"] == 2
    assert payload["pages"][0]["markdown"] == "PAGE1"
    assert payload["pages"][1]["markdown"] == "PAGE2"
    assert [(u.phase, u.current, u.total) for u in updates] == [
        (ProgressPhase.rendering, 0, 2),
        (ProgressPhase.done, 2, 2),
    ]


def test_pdf_to_json_writes_document_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_iter_pdf_images(monkeypatch, page_count=1)
    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image", lambda _img: "only")

    out_dir = tmp_path / "artifacts"
    result = pdf_to_json(b"%PDF", page_output_dir=out_dir)
    data = json.loads(result)
    assert data["pages"][0]["markdown"] == "only"

    artifact = out_dir / "document.json"
    assert artifact.exists()
    assert artifact.read_text(encoding="utf-8").endswith("\n")


def test_pdf_to_json_invalid_pdf_propagates_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_iter_pdf_images(*_args: object, **_kwargs: object) -> tuple[int, object]:
        raise ValueError("Invalid PDF data")

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)

    with pytest.raises(ValueError, match="Invalid PDF data"):
        pdf_to_json(b"not a pdf")

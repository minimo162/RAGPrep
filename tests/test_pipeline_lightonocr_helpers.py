from __future__ import annotations

import json

import pytest
from PIL import Image

import ragprep.pipeline as pipeline
from ragprep.config import get_settings


def _patch_iter_pdf_images(monkeypatch: pytest.MonkeyPatch, page_count: int) -> None:
    images = [Image.new("RGB", (2, 2)) for _ in range(page_count)]

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


def test_pdf_to_markdown_lightonocr_normalizes_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_iter_pdf_images(monkeypatch, page_count=1)
    monkeypatch.setattr(
        "ragprep.ocr.lightonocr.ocr_image",
        lambda _img: "line1\r\nline2\r",
    )

    settings = get_settings()
    result = pipeline._pdf_to_markdown_lightonocr(b"%PDF", settings=settings)
    assert result == "line1\nline2"


def test_pdf_to_json_lightonocr_builds_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_iter_pdf_images(monkeypatch, page_count=2)

    outputs = iter(["PAGE1", "PAGE2"])

    def _fake_ocr_image(_img: Image.Image) -> str:
        return next(outputs)

    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image", _fake_ocr_image)

    settings = get_settings()
    result = pipeline._pdf_to_json_lightonocr(b"%PDF", settings=settings)
    data = json.loads(result)

    assert data["meta"]["backend"] == "lightonocr"
    assert data["meta"]["page_count"] == 2
    assert data["pages"][0]["page"] == 1
    assert data["pages"][0]["markdown"] == "PAGE1"
    assert data["pages"][1]["page"] == 2
    assert data["pages"][1]["markdown"] == "PAGE2"


def test_pdf_to_json_lightonocr_calls_on_page(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_iter_pdf_images(monkeypatch, page_count=2)

    outputs = iter(["PAGE1\r", " PAGE2 "])

    def _fake_ocr_image(_img: Image.Image) -> str:
        return next(outputs)

    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image", _fake_ocr_image)

    pages: list[tuple[int, str]] = []

    def on_page(page_index: int, markdown: str) -> None:
        pages.append((page_index, markdown))

    settings = get_settings()
    result = pipeline._pdf_to_json_lightonocr(b"%PDF", settings=settings, on_page=on_page)
    data = json.loads(result)

    assert data["pages"][0]["markdown"] == "PAGE1"
    assert data["pages"][1]["markdown"] == "PAGE2"
    assert pages == [(1, "PAGE1"), (2, "PAGE2")]


def test_pdf_to_json_lightonocr_error_includes_root_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_iter_pdf_images(monkeypatch, page_count=1)

    def _raise(_img: Image.Image) -> str:
        raise RuntimeError("GGUF model file not found: missing.gguf")

    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image", _raise)

    settings = get_settings()
    with pytest.raises(RuntimeError) as excinfo:
        pipeline._pdf_to_json_lightonocr(b"%PDF", settings=settings)
    message = str(excinfo.value)
    assert "LightOnOCR failed on page 1" in message
    assert "GGUF model file not found" in message

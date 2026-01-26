from __future__ import annotations

import pytest

from ragprep.pipeline import pdf_to_markdown


def _squash_ws(text: str) -> str:
    return "".join(text.split())


def test_pdf_to_markdown_e2e_contains_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]

    def _fake_iter_pdf_images(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, object]:
        return 2, iter(images)

    monkeypatch.setattr("ragprep.pdf_render.iter_pdf_images", _fake_iter_pdf_images)

    outputs = iter(["Hello E2E 1", "Hello E2E 2"])

    def _fake_ocr_image(_image: Image.Image) -> str:
        return next(outputs)

    monkeypatch.setattr("ragprep.ocr.lightonocr.ocr_image", _fake_ocr_image)

    markdown = pdf_to_markdown(b"%PDF")
    squashed = _squash_ws(markdown)

    assert _squash_ws("Hello E2E 1") in squashed
    assert _squash_ws("Hello E2E 2") in squashed


from __future__ import annotations

from collections.abc import Iterator

import pytest
from PIL import Image

from ragprep.pipeline import pdf_to_markdown


def _squash_ws(text: str) -> str:
    return "".join(text.split())


def test_pdf_to_markdown_e2e_contains_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    monkeypatch.setattr(
        "ragprep.pdf_text.extract_pymupdf_page_sizes",
        lambda _pdf: [(1000.0, 1000.0), (1000.0, 1000.0)],
    )
    monkeypatch.setattr(
        "ragprep.pdf_text.extract_pymupdf_page_words",
        lambda _pdf: [[], []],
    )

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
            for _idx in range(2):
                yield Image.new("RGB", (1000, 1000), color=(255, 255, 255))

        return 2, _generate()

    monkeypatch.setattr(
        "ragprep.pdf_render.iter_pdf_images",
        _fake_iter_pdf_images,
    )

    outputs = iter(["Hello E2E 1", "Hello E2E 2"])

    def _fake_lighton(_encoded: str, *, settings: object) -> dict[str, object]:
        _ = settings
        text = next(outputs)
        return {
            "schema_version": "v1",
            "elements": [],
            "lines": [{"bbox": (100.0, 100.0, 400.0, 140.0), "text": text}],
            "raw": "{}",
        }

    monkeypatch.setattr("ragprep.ocr.lighton_ocr.analyze_ocr_layout_image_base64", _fake_lighton)

    markdown = pdf_to_markdown(b"%PDF")
    squashed = _squash_ws(markdown)

    assert _squash_ws("Hello E2E 1") in squashed
    assert _squash_ws("Hello E2E 2") in squashed


from __future__ import annotations

from typing import cast

import pytest

from ragprep.pdf_render import render_pdf_to_images


def _make_pdf_bytes(page_count: int) -> bytes:
    import fitz

    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 72), f"Hello {i + 1}")
    return cast(bytes, doc.tobytes())


def test_render_pdf_to_images_returns_one_image_per_page() -> None:
    pdf_bytes = _make_pdf_bytes(page_count=2)
    images = render_pdf_to_images(pdf_bytes, dpi=72)
    assert len(images) == 2
    assert all(image.mode == "RGB" for image in images)


def test_render_pdf_to_images_rejects_too_many_pages() -> None:
    pdf_bytes = _make_pdf_bytes(page_count=2)
    with pytest.raises(ValueError, match="max_pages"):
        render_pdf_to_images(pdf_bytes, max_pages=1)


def test_render_pdf_to_images_rejects_invalid_pdf() -> None:
    with pytest.raises(ValueError, match="Invalid PDF"):
        render_pdf_to_images(b"not a pdf")


def test_render_pdf_to_images_rejects_large_pdf_bytes() -> None:
    pdf_bytes = _make_pdf_bytes(page_count=1)
    with pytest.raises(ValueError, match="PDF too large"):
        render_pdf_to_images(pdf_bytes, max_bytes=1)

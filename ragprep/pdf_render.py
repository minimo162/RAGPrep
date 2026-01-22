from __future__ import annotations

import io
from typing import Any, Final

from PIL import Image

DEFAULT_DPI: Final[int] = 200
MAX_PDF_BYTES: Final[int] = 10 * 1024 * 1024
MAX_PAGES: Final[int] = 50


def _import_fitz() -> Any:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # noqa: BLE001
        raise ImportError("PyMuPDF is required for PDF rendering. Install `pymupdf`.") from exc

    return fitz


def render_pdf_to_images(
    pdf_bytes: bytes,
    *,
    dpi: int = DEFAULT_DPI,
    max_pages: int = MAX_PAGES,
    max_bytes: int = MAX_PDF_BYTES,
) -> list[Image.Image]:
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")
    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    if max_pages <= 0:
        raise ValueError("max_pages must be > 0")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if len(pdf_bytes) > max_bytes:
        raise ValueError(f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={max_bytes}")

    fitz = _import_fitz()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid PDF data") from exc

    with doc:
        page_count = int(doc.page_count)
        if page_count > max_pages:
            raise ValueError(f"PDF has {page_count} pages, max_pages={max_pages}")

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        images: list[Image.Image] = []
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pix.tobytes("png")
            with io.BytesIO(png_bytes) as buf:
                pil_image = Image.open(buf)
                pil_image.load()
                images.append(pil_image.convert("RGB"))

        return images

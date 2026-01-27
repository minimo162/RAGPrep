from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from PIL import Image

from ragprep.config import get_settings


def _import_fitz() -> Any:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # noqa: BLE001
        raise ImportError("PyMuPDF is required for PDF rendering. Install `pymupdf`.") from exc
    return fitz


def _pixmap_to_rgb_image(pixmap: Any) -> Image.Image:
    width = int(pixmap.width)
    height = int(pixmap.height)
    channels = int(getattr(pixmap, "n", 0))
    samples = pixmap.samples

    if channels == 4:
        return Image.frombytes("RGBA", (width, height), samples).convert("RGB")
    if channels == 3:
        return Image.frombytes("RGB", (width, height), samples)
    if channels == 1:
        return Image.frombytes("L", (width, height), samples).convert("RGB")

    raise ValueError(f"Unsupported pixmap channel count: {channels}")


def iter_pdf_images(
    pdf_bytes: bytes,
    *,
    dpi: int | None = None,
    max_edge: int | None = None,
    max_pages: int | None = None,
    max_bytes: int | None = None,
) -> tuple[int, Iterator[Image.Image]]:
    settings = get_settings()
    dpi = settings.render_dpi if dpi is None else dpi
    max_edge = settings.render_max_edge if max_edge is None else max_edge
    max_pages = settings.max_pages if max_pages is None else max_pages
    max_bytes = settings.max_upload_bytes if max_bytes is None else max_bytes

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")
    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    if max_edge <= 0:
        raise ValueError("max_edge must be > 0")
    if max_pages <= 0:
        raise ValueError("max_pages must be > 0")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if len(pdf_bytes) > max_bytes:
        raise ValueError(f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={max_bytes}")

    try:
        fitz = _import_fitz()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid PDF data") from exc

    try:
        page_count = int(len(doc))
    except Exception as exc:  # noqa: BLE001
        doc.close()
        raise RuntimeError("Failed to read PDF page count") from exc
    if page_count > max_pages:
        doc.close()
        raise ValueError(f"PDF has {page_count} pages, max_pages={max_pages}")

    scale = dpi / 72.0

    def generate() -> Iterator[Image.Image]:
        try:
            for page_index in range(page_count):
                page = doc.load_page(page_index)
                matrix = fitz.Matrix(scale, scale)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                rgb = _pixmap_to_rgb_image(pixmap)

                width, height = rgb.size
                current_max_edge = max(width, height)
                if current_max_edge > max_edge:
                    if width >= height:
                        new_width = max_edge
                        new_height = max(1, round(max_edge * height / width))
                    else:
                        new_height = max_edge
                        new_width = max(1, round(max_edge * width / height))
                    rgb = rgb.resize(
                        (int(new_width), int(new_height)),
                        resample=Image.Resampling.LANCZOS,
                    )

                yield rgb
        finally:
            doc.close()

    return page_count, generate()


def render_pdf_to_images(
    pdf_bytes: bytes,
    *,
    dpi: int | None = None,
    max_edge: int | None = None,
    max_pages: int | None = None,
    max_bytes: int | None = None,
) -> list[Image.Image]:
    _total_pages, images = iter_pdf_images(
        pdf_bytes,
        dpi=dpi,
        max_edge=max_edge,
        max_pages=max_pages,
        max_bytes=max_bytes,
    )
    return list(images)

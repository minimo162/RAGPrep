from __future__ import annotations

from ragprep.ocr.lightonocr import ocr_image
from ragprep.pdf_render import render_pdf_to_images


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def pdf_to_markdown(pdf_bytes: bytes) -> str:
    """
    Convert a PDF (bytes) into a Markdown/text string.

    This function is intentionally pure and synchronous; it orchestrates PDF rendering
    and per-page OCR, and applies small deterministic post-processing.
    """

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    images = render_pdf_to_images(pdf_bytes)
    page_texts: list[str] = []
    for image in images:
        text = _normalize_newlines(ocr_image(image)).strip()
        if text:
            page_texts.append(text)

    return "\n\n".join(page_texts).strip()

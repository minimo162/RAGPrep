from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from ragprep.ocr.lightonocr import ocr_image
from ragprep.pdf_render import iter_pdf_images


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


class ProgressPhase(str, Enum):
    rendering = "rendering"
    ocr = "ocr"
    done = "done"


@dataclass(frozen=True)
class PdfToMarkdownProgress:
    phase: ProgressPhase
    current: int
    total: int
    message: str | None = None


ProgressCallback = Callable[[PdfToMarkdownProgress], None]


def _notify_progress(on_progress: ProgressCallback | None, update: PdfToMarkdownProgress) -> None:
    if on_progress is None:
        return
    try:
        on_progress(update)
    except Exception:  # noqa: BLE001
        return


def pdf_to_markdown(pdf_bytes: bytes, *, on_progress: ProgressCallback | None = None) -> str:
    """
    Convert a PDF (bytes) into a Markdown/text string.

    This function is intentionally pure and synchronous; it orchestrates PDF rendering
    and per-page OCR, and applies small deterministic post-processing.
    """

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    _notify_progress(
        on_progress,
        PdfToMarkdownProgress(
            phase=ProgressPhase.rendering,
            current=0,
            total=0,
            message="rendering",
        ),
    )
    total_pages, images = iter_pdf_images(pdf_bytes)
    _notify_progress(
        on_progress,
        PdfToMarkdownProgress(
            phase=ProgressPhase.rendering,
            current=total_pages,
            total=total_pages,
            message="rendered",
        ),
    )
    _notify_progress(
        on_progress,
        PdfToMarkdownProgress(
            phase=ProgressPhase.ocr,
            current=0,
            total=total_pages,
            message="ocr",
        ),
    )
    page_texts: list[str] = []
    for page_number, image in enumerate(images, start=1):
        text = _normalize_newlines(ocr_image(image)).strip()
        if text:
            page_texts.append(text)
        _notify_progress(
            on_progress,
            PdfToMarkdownProgress(
                phase=ProgressPhase.ocr,
                current=page_number,
                total=total_pages,
                message=f"page {page_number}",
            ),
        )

    _notify_progress(
        on_progress,
        PdfToMarkdownProgress(
            phase=ProgressPhase.done,
            current=total_pages,
            total=total_pages,
            message="done",
        ),
    )
    return "\n\n".join(page_texts).strip()

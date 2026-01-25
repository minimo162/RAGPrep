from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ragprep.config import get_settings
from ragprep.pymupdf4llm_markdown import pdf_bytes_to_markdown


class ProgressPhase(str, Enum):
    rendering = "rendering"
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


def _write_text_artifact(path: Path, text: str) -> None:
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def pdf_to_markdown(
    pdf_bytes: bytes,
    *,
    on_progress: ProgressCallback | None = None,
    page_output_dir: Path | None = None,
) -> str:
    """
    Convert a PDF (bytes) into a Markdown/text string.

    This function is intentionally pure and synchronous; it converts the entire document
    in a single pass via pymupdf-layout + pymupdf4llm.
    """

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    if page_output_dir is not None:
        page_output_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={settings.max_upload_bytes}"
        )

    try:
        import pymupdf

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid PDF data") from exc

    with doc:
        total_pages = int(doc.page_count)

    if total_pages > settings.max_pages:
        raise ValueError(f"PDF has {total_pages} pages, max_pages={settings.max_pages}")

    _notify_progress(
        on_progress,
        PdfToMarkdownProgress(
            phase=ProgressPhase.rendering,
            current=0,
            total=total_pages,
            message="converting",
        ),
    )

    markdown = pdf_bytes_to_markdown(pdf_bytes)

    if page_output_dir is not None:
        _write_text_artifact(page_output_dir / "document.md", markdown)

    _notify_progress(
        on_progress,
        PdfToMarkdownProgress(
            phase=ProgressPhase.done,
            current=total_pages,
            total=total_pages,
            message="done",
        ),
    )
    return markdown


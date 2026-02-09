from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ragprep.config import get_settings
from ragprep.html_render import render_document_html, render_page_html, wrap_html_document
from ragprep.ocr import lighton_ocr
from ragprep.ocr_html import page_from_ocr_markdown
from ragprep.pdf_render import iter_pdf_page_png_base64
from ragprep.pdf_text import extract_pymupdf_page_texts
from ragprep.structure_ir import Document, Page
from ragprep.text_merge import merge_ocr_with_pymupdf


class ProgressPhase(str, Enum):
    rendering = "rendering"
    done = "done"


@dataclass(frozen=True)
class PdfToHtmlProgress:
    phase: ProgressPhase
    current: int
    total: int
    message: str | None = None


HtmlProgressCallback = Callable[[PdfToHtmlProgress], None]
PageCallback = Callable[[int, str], None]


def _notify_html_progress(
    on_progress: HtmlProgressCallback | None,
    update: PdfToHtmlProgress,
) -> None:
    if on_progress is None:
        return
    try:
        on_progress(update)
    except Exception:  # noqa: BLE001
        return


def _write_text_artifact(path: Path, text: str) -> None:
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def pdf_to_html(
    pdf_bytes: bytes,
    *,
    full_document: bool = True,
    on_progress: HtmlProgressCallback | None = None,
    on_page: PageCallback | None = None,
    page_output_dir: Path | None = None,
) -> str:
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")
    if page_output_dir is not None:
        page_output_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={settings.max_upload_bytes}"
        )
    if not pdf_bytes.lstrip().startswith(b"%PDF"):
        raise ValueError("Invalid PDF data")

    total_pages, encoded_pages_iter = iter_pdf_page_png_base64(
        pdf_bytes,
        dpi=settings.lighton_render_dpi,
        max_edge=settings.lighton_render_max_edge,
        max_pages=settings.max_pages,
        max_bytes=settings.max_upload_bytes,
    )
    encoded_pages = list(encoded_pages_iter)
    if len(encoded_pages) != int(total_pages):
        raise RuntimeError("Page count mismatch while rendering PDF pages.")

    pymupdf_texts = extract_pymupdf_page_texts(pdf_bytes)
    if len(pymupdf_texts) != int(total_pages):
        raise RuntimeError("Page count mismatch between OCR rendering and PyMuPDF text layer.")

    _notify_html_progress(
        on_progress,
        PdfToHtmlProgress(
            phase=ProgressPhase.rendering,
            current=0,
            total=int(total_pages),
            message="converting",
        ),
    )

    pages: list[Page] = []
    partial_sections: list[str] = []
    merged_page_texts: list[str] = []

    max_workers = max(1, min(int(settings.lighton_page_concurrency), int(total_pages)))
    futures: dict[int, Future[tuple[Page, str, str]]] = {}

    def _process_page(page_number: int, image_base64: str) -> tuple[Page, str, str]:
        ocr_text = lighton_ocr.ocr_image_base64(image_base64, settings=settings)
        pymupdf_text = pymupdf_texts[page_number - 1]
        merged_text, _merge_stats = merge_ocr_with_pymupdf(
            ocr_text,
            pymupdf_text,
            policy=settings.lighton_merge_policy,
        )
        page = page_from_ocr_markdown(page_number=page_number, markdown=merged_text)
        section_html = render_page_html(page)
        return page, section_html, merged_text

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for page_number, encoded in enumerate(encoded_pages, start=1):
            futures[page_number] = executor.submit(_process_page, page_number, encoded)

        try:
            for page_number in range(1, int(total_pages) + 1):
                page, section_html, merged_text = futures[page_number].result()
                pages.append(page)
                partial_sections.append(section_html)
                merged_page_texts.append(merged_text)

                if on_page is not None:
                    on_page(page_number, section_html)

                _notify_html_progress(
                    on_progress,
                    PdfToHtmlProgress(
                        phase=ProgressPhase.rendering,
                        current=page_number,
                        total=int(total_pages),
                        message="converting",
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            for future in futures.values():
                future.cancel()
            raise RuntimeError(f"LightOn OCR failed: {exc}") from exc

    fragment = render_document_html(Document(pages=tuple(pages)))
    html = wrap_html_document(fragment) if full_document else fragment

    if page_output_dir is not None:
        _write_text_artifact(page_output_dir / "document.html", html)
        _write_text_artifact(page_output_dir / "partial.html", "\n".join(partial_sections))
        _write_text_artifact(page_output_dir / "ocr_merged.txt", "\n\n".join(merged_page_texts))

    _notify_html_progress(
        on_progress,
        PdfToHtmlProgress(
            phase=ProgressPhase.done,
            current=int(total_pages),
            total=int(total_pages),
            message="done",
        ),
    )
    return html

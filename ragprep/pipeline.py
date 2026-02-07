from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import cast

from ragprep.config import Settings, get_settings


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


@dataclass(frozen=True)
class PdfToJsonProgress:
    phase: ProgressPhase
    current: int
    total: int
    message: str | None = None


JsonProgressCallback = Callable[[PdfToJsonProgress], None]
PageCallback = Callable[[int, str], None]


def _notify_progress(on_progress: ProgressCallback | None, update: PdfToMarkdownProgress) -> None:
    if on_progress is None:
        return
    try:
        on_progress(update)
    except Exception:  # noqa: BLE001
        return


def _notify_json_progress(
    on_progress: JsonProgressCallback | None, update: PdfToJsonProgress
) -> None:
    if on_progress is None:
        return
    try:
        on_progress(update)
    except Exception:  # noqa: BLE001
        return


@dataclass(frozen=True)
class PdfToHtmlProgress:
    phase: ProgressPhase
    current: int
    total: int
    message: str | None = None


HtmlProgressCallback = Callable[[PdfToHtmlProgress], None]


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


def _pdf_to_markdown_glm_ocr(
    pdf_bytes: bytes,
    *,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_page: PageCallback | None = None,
) -> str:
    from ragprep.ocr import glm_ocr
    from ragprep.pdf_render import iter_pdf_page_png_base64

    try:
        total_pages, encoded_pages = iter_pdf_page_png_base64(
            pdf_bytes,
            dpi=settings.render_dpi,
            max_edge=settings.render_max_edge,
            max_pages=settings.max_pages,
            max_bytes=settings.max_upload_bytes,
        )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to render PDF for GLM-OCR.") from exc

    _notify_progress(
        on_progress,
        PdfToMarkdownProgress(
            phase=ProgressPhase.rendering,
            current=0,
            total=int(total_pages),
            message="converting",
        ),
    )

    parts: list[str] = []
    for page_index, encoded in enumerate(encoded_pages, start=1):
        try:
            text = glm_ocr.ocr_image_base64(encoded, settings=settings)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"GLM-OCR failed on page {page_index}: {exc}") from exc
        normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
        if on_page is not None:
            on_page(page_index, normalized)
        if normalized:
            parts.append(normalized)
        _notify_progress(
            on_progress,
            PdfToMarkdownProgress(
                phase=ProgressPhase.rendering,
                current=page_index,
                total=int(total_pages),
                message="converting",
            ),
        )

    markdown = "\n\n".join(parts).strip()

    _notify_progress(
        on_progress,
        PdfToMarkdownProgress(
            phase=ProgressPhase.done,
            current=int(total_pages),
            total=int(total_pages),
            message="done",
        ),
    )
    return markdown


def pdf_to_markdown(
    pdf_bytes: bytes,
    *,
    on_progress: ProgressCallback | None = None,
    on_page: PageCallback | None = None,
    page_output_dir: Path | None = None,
) -> str:
    """
    Convert a PDF (bytes) into a Markdown/text string.

    This function is intentionally pure and synchronous; it converts the entire document
    in a single pass via the configured OCR backend.
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

    if settings.pdf_backend != "glm-ocr":
        raise RuntimeError(f"Unsupported PDF backend: {settings.pdf_backend!r}")

    markdown = _pdf_to_markdown_glm_ocr(
        pdf_bytes,
        settings=settings,
        on_progress=on_progress,
        on_page=on_page,
    )

    if page_output_dir is not None:
        _write_text_artifact(page_output_dir / "document.md", markdown)

    return markdown


def _image_to_png_base64(image: object) -> str:
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise ImportError("Pillow is required. Install `pillow`.") from exc

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    import base64

    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pdf_to_html(
    pdf_bytes: bytes,
    *,
    full_document: bool = True,
    on_progress: HtmlProgressCallback | None = None,
    on_page: PageCallback | None = None,
    page_output_dir: Path | None = None,
) -> str:
    """
    Convert a PDF (bytes) into HTML.

    Strategy:
    - Extract text layer spans via PyMuPDF.
    - For `local-fast`, infer layout directly from spans (no image rendering).
    - For `server`/`local-paddle`, run image-based layout analysis.
    - Assign spans to layout regions and render structured HTML.
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

    if not pdf_bytes.lstrip().startswith(b"%PDF"):
        raise ValueError("Invalid PDF data")

    from ragprep.html_render import render_document_html, render_page_html, wrap_html_document
    from ragprep.pdf_text import (
        extract_pymupdf_page_sizes,
        extract_pymupdf_page_spans,
        extract_pymupdf_page_words,
    )
    from ragprep.structure_ir import Document, Page, build_page_blocks, layout_element_from_raw

    spans_by_page = extract_pymupdf_page_spans(pdf_bytes)
    words_by_page = extract_pymupdf_page_words(pdf_bytes)
    page_sizes = extract_pymupdf_page_sizes(pdf_bytes)
    total_pages = int(len(spans_by_page))

    if len(page_sizes) != total_pages:
        raise RuntimeError("Page count mismatch between text extraction and page sizes.")

    layout_mode = (settings.layout_mode or "").strip().lower()
    if layout_mode == "transformers":
        # Backward-compatible alias.
        layout_mode = "local-fast"

    _notify_html_progress(
        on_progress,
        PdfToHtmlProgress(
            phase=ProgressPhase.rendering,
            current=0,
            total=total_pages,
            message="converting",
        ),
    )

    pages: list[Page] = []
    partial_sections: list[str] = []

    def _build_page_from_layout(
        *,
        page_number: int,
        image_width: float,
        image_height: float,
        layout_raw: dict[str, object],
    ) -> tuple[Page, str]:
        raw_elements = layout_raw.get("elements", [])
        if not isinstance(raw_elements, list):
            raise RuntimeError("Layout analysis returned invalid elements.")

        layout_elements = [
            layout_element_from_raw(
                cast(dict[str, object], raw),
                page_index=page_number - 1,
                image_width=image_width,
                image_height=image_height,
            )
            for raw in raw_elements
            if isinstance(raw, dict)
        ]

        page_w, page_h = page_sizes[page_number - 1]
        blocks = build_page_blocks(
            spans=spans_by_page[page_number - 1],
            page_words=words_by_page[page_number - 1],
            page_width=page_w,
            page_height=page_h,
            layout_elements=layout_elements,
        )
        page = Page(page_number=page_number, blocks=blocks)
        return page, render_page_html(page)

    if layout_mode == "local-fast":
        from ragprep.layout.fast_layout import infer_fast_layout_elements

        for page_number, spans in enumerate(spans_by_page, start=1):
            page_w, page_h = page_sizes[page_number - 1]
            fast_elements = infer_fast_layout_elements(spans, page_w, page_h)
            page, section_html = _build_page_from_layout(
                page_number=page_number,
                image_width=page_w,
                image_height=page_h,
                layout_raw={"elements": fast_elements},
            )
            pages.append(page)
            partial_sections.append(section_html)
            if on_page is not None:
                on_page(page_number, section_html)

            _notify_html_progress(
                on_progress,
                PdfToHtmlProgress(
                    phase=ProgressPhase.rendering,
                    current=page_number,
                    total=total_pages,
                    message="converting",
                ),
            )
    else:
        from ragprep.layout.glm_doclayout import analyze_layout_image_base64
        from ragprep.pdf_render import iter_pdf_images, render_pdf_page_image

        adaptive_enabled = bool(settings.layout_render_auto) and layout_mode == "server"
        primary_dpi = settings.layout_render_dpi
        primary_max_edge = settings.layout_render_max_edge
        fallback_dpi = settings.layout_render_dpi
        fallback_max_edge = settings.layout_render_max_edge
        if adaptive_enabled:
            if (
                settings.layout_render_auto_small_dpi == fallback_dpi
                and settings.layout_render_auto_small_max_edge == fallback_max_edge
            ):
                adaptive_enabled = False
            primary_dpi = settings.layout_render_auto_small_dpi
            primary_max_edge = settings.layout_render_auto_small_max_edge

        rendered_total_pages, images = iter_pdf_images(
            pdf_bytes,
            dpi=primary_dpi,
            max_edge=primary_max_edge,
            max_pages=settings.max_pages,
            max_bytes=settings.max_upload_bytes,
        )
        if int(rendered_total_pages) != total_pages:
            raise RuntimeError("Page count mismatch between render and text extraction.")

        def _layout_needs_retry(layout_raw: dict[str, object]) -> bool:
            elements = layout_raw.get("elements")
            return isinstance(elements, list) and len(elements) == 0

        def _analyze_layout_for_page(
            *,
            page_number: int,
            encoded: str,
            image_width: float,
            image_height: float,
        ) -> tuple[dict[str, object], float, float]:
            layout_raw = analyze_layout_image_base64(encoded, settings=settings)
            if not adaptive_enabled or not _layout_needs_retry(layout_raw):
                return layout_raw, image_width, image_height

            large_image = render_pdf_page_image(
                pdf_bytes,
                page_index=page_number - 1,
                dpi=fallback_dpi,
                max_edge=fallback_max_edge,
                max_pages=settings.max_pages,
                max_bytes=settings.max_upload_bytes,
            )
            try:
                encoded_large = _image_to_png_base64(large_image)
                layout_raw_large = analyze_layout_image_base64(encoded_large, settings=settings)
                return layout_raw_large, float(large_image.width), float(large_image.height)
            finally:
                try:
                    large_image.close()
                except Exception:  # noqa: BLE001
                    pass

        layout_workers = 1
        if layout_mode == "server":
            layout_workers = max(1, int(settings.layout_concurrency))

        if layout_workers <= 1:
            for page_number, image in enumerate(images, start=1):
                image_width = float(getattr(image, "width", 0))
                image_height = float(getattr(image, "height", 0))
                if image_width <= 0 or image_height <= 0:
                    raise RuntimeError(
                        "Failed to read rendered image size for layout normalization."
                    )

                try:
                    encoded = _image_to_png_base64(image)
                finally:
                    try:
                        image.close()
                    except Exception:  # noqa: BLE001
                        pass

                layout_raw, used_w, used_h = _analyze_layout_for_page(
                    page_number=page_number,
                    encoded=encoded,
                    image_width=image_width,
                    image_height=image_height,
                )
                page, section_html = _build_page_from_layout(
                    page_number=page_number,
                    image_width=used_w,
                    image_height=used_h,
                    layout_raw=layout_raw,
                )
                pages.append(page)
                partial_sections.append(section_html)
                if on_page is not None:
                    on_page(page_number, section_html)

                _notify_html_progress(
                    on_progress,
                    PdfToHtmlProgress(
                        phase=ProgressPhase.rendering,
                        current=page_number,
                        total=total_pages,
                        message="converting",
                    ),
                )
        else:
            inflight: dict[int, Future[tuple[dict[str, object], float, float]]] = {}
            next_page_to_process = 1
            executor = ThreadPoolExecutor(
                max_workers=min(layout_workers, total_pages),
                thread_name_prefix="ragprep-layout",
            )

            def _process_page(page_number: int) -> None:
                future = inflight[page_number]
                layout_raw, image_width, image_height = future.result()
                page, section_html = _build_page_from_layout(
                    page_number=page_number,
                    image_width=image_width,
                    image_height=image_height,
                    layout_raw=layout_raw,
                )
                pages.append(page)
                partial_sections.append(section_html)
                if on_page is not None:
                    on_page(page_number, section_html)
                _notify_html_progress(
                    on_progress,
                    PdfToHtmlProgress(
                        phase=ProgressPhase.rendering,
                        current=page_number,
                        total=total_pages,
                        message="converting",
                    ),
                )

            try:
                for page_number, image in enumerate(images, start=1):
                    image_width = float(getattr(image, "width", 0))
                    image_height = float(getattr(image, "height", 0))
                    if image_width <= 0 or image_height <= 0:
                        raise RuntimeError(
                            "Failed to read rendered image size for layout normalization."
                        )

                    try:
                        encoded = _image_to_png_base64(image)
                    finally:
                        try:
                            image.close()
                        except Exception:  # noqa: BLE001
                            pass

                    future = executor.submit(
                        _analyze_layout_for_page,
                        page_number=page_number,
                        encoded=encoded,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    inflight[page_number] = future

                    while len(inflight) >= layout_workers and next_page_to_process in inflight:
                        _process_page(next_page_to_process)
                        inflight.pop(next_page_to_process, None)
                        next_page_to_process += 1

                    while (
                        next_page_to_process in inflight
                        and inflight[next_page_to_process].done()
                    ):
                        _process_page(next_page_to_process)
                        inflight.pop(next_page_to_process, None)
                        next_page_to_process += 1

                while next_page_to_process in inflight:
                    _process_page(next_page_to_process)
                    inflight.pop(next_page_to_process, None)
                    next_page_to_process += 1
            finally:
                for future in inflight.values():
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)

    document = Document(pages=tuple(pages))
    fragment = render_document_html(document)
    html = wrap_html_document(fragment) if full_document else fragment

    if page_output_dir is not None:
        _write_text_artifact(page_output_dir / "document.html", html)
        _write_text_artifact(page_output_dir / "partial.html", "\n".join(partial_sections))

    _notify_html_progress(
        on_progress,
        PdfToHtmlProgress(
            phase=ProgressPhase.done,
            current=total_pages,
            total=total_pages,
            message="done",
        ),
    )

    return html


def pdf_to_json(
    pdf_bytes: bytes,
    *,
    on_progress: JsonProgressCallback | None = None,
    on_page: PageCallback | None = None,
    page_output_dir: Path | None = None,
) -> str:
    """
    Backward-compatible wrapper that now returns Markdown only.

    JSON 蜃ｺ蜉帙・蟒・ｭ｢縺励・md 縺ｮ縺ｿ繧堤函謌舌☆繧九・
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

    def _adapt_progress(update: PdfToMarkdownProgress) -> None:
        if on_progress is None:
            return
        _notify_json_progress(
            on_progress,
            PdfToJsonProgress(
                phase=update.phase,
                current=update.current,
                total=update.total,
                message=update.message,
            ),
        )

    return pdf_to_markdown(
        pdf_bytes,
        on_progress=_adapt_progress if on_progress is not None else None,
        on_page=on_page,
        page_output_dir=page_output_dir,
    )


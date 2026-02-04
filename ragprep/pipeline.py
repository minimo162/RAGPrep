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

    Current strategy:
    - Extract text layer spans via PyMuPDF.
    - Run layout analysis per rendered page image (PP-DocLayout-V3 contract).
    - Assign spans to layout regions and render a structured HTML output.
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
    from ragprep.layout.glm_doclayout import analyze_layout_image_base64
    from ragprep.pdf_render import iter_pdf_images
    from ragprep.pdf_text import extract_pymupdf_page_sizes, extract_pymupdf_page_spans
    from ragprep.structure_ir import Document, Page, build_page_blocks, layout_element_from_raw

    spans_by_page = extract_pymupdf_page_spans(pdf_bytes)
    page_sizes = extract_pymupdf_page_sizes(pdf_bytes)

    total_pages, images = iter_pdf_images(
        pdf_bytes,
        dpi=settings.layout_render_dpi,
        max_edge=settings.layout_render_max_edge,
        max_pages=settings.max_pages,
        max_bytes=settings.max_upload_bytes,
    )

    if len(spans_by_page) != int(total_pages) or len(page_sizes) != int(total_pages):
        raise RuntimeError("Page count mismatch between render and text extraction.")

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
            page_width=page_w,
            page_height=page_h,
            layout_elements=layout_elements,
        )
        page = Page(page_number=page_number, blocks=blocks)
        return page, render_page_html(page)

    layout_workers = 1
    if (settings.layout_mode or "").strip().lower() == "server":
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

            layout_raw = analyze_layout_image_base64(encoded, settings=settings)
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
                    total=int(total_pages),
                    message="converting",
                ),
            )
    else:
        inflight: dict[int, tuple[Future[dict[str, object]], float, float]] = {}
        next_page_to_process = 1
        executor = ThreadPoolExecutor(
            max_workers=min(layout_workers, int(total_pages)),
            thread_name_prefix="ragprep-layout",
        )

        def _process_page(page_number: int) -> None:
            future, image_width, image_height = inflight[page_number]
            layout_raw = future.result()
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
                    total=int(total_pages),
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

                future = executor.submit(analyze_layout_image_base64, encoded, settings=settings)
                inflight[page_number] = (future, image_width, image_height)

                while len(inflight) >= layout_workers and next_page_to_process in inflight:
                    _process_page(next_page_to_process)
                    inflight.pop(next_page_to_process, None)
                    next_page_to_process += 1

                while (
                    next_page_to_process in inflight
                    and inflight[next_page_to_process][0].done()
                ):
                    _process_page(next_page_to_process)
                    inflight.pop(next_page_to_process, None)
                    next_page_to_process += 1

            while next_page_to_process in inflight:
                _process_page(next_page_to_process)
                inflight.pop(next_page_to_process, None)
                next_page_to_process += 1
        finally:
            for future, _w, _h in inflight.values():
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
            current=int(total_pages),
            total=int(total_pages),
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

    JSON 出力は廃止し、.md のみを生成する。
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

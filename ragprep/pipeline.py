from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ragprep.config import Settings, get_settings
from ragprep.pymupdf4llm_json import pdf_bytes_to_json
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


@dataclass(frozen=True)
class PdfToJsonProgress:
    phase: ProgressPhase
    current: int
    total: int
    message: str | None = None


JsonProgressCallback = Callable[[PdfToJsonProgress], None]


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


def _write_text_artifact(path: Path, text: str) -> None:
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _pdf_to_markdown_lightonocr(pdf_bytes: bytes, *, settings: Settings) -> str:
    from ragprep.ocr import lightonocr
    from ragprep.pdf_render import iter_pdf_images

    try:
        _total_pages, images = iter_pdf_images(
            pdf_bytes,
            dpi=settings.render_dpi,
            max_edge=settings.render_max_edge,
            max_pages=settings.max_pages,
            max_bytes=settings.max_upload_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to render PDF for LightOnOCR.") from exc

    parts: list[str] = []
    for page_index, image in enumerate(images, start=1):
        try:
            text = lightonocr.ocr_image(image)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LightOnOCR failed on page {page_index}.") from exc
        normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
        if normalized:
            parts.append(normalized)

    return "\n\n".join(parts).strip()


def _pdf_to_json_lightonocr(pdf_bytes: bytes, *, settings: Settings) -> str:
    """
    LightOnOCR JSON schema (minimal):
    {
      "meta": {
        "backend": "lightonocr",
        "page_count": <int>,
        "render_dpi": <int>,
        "render_max_edge": <int>
      },
      "pages": [
        {"page": 1, "markdown": "..."},
        ...
      ]
    }
    """
    import json

    from ragprep.ocr import lightonocr
    from ragprep.pdf_render import iter_pdf_images

    try:
        total_pages, images = iter_pdf_images(
            pdf_bytes,
            dpi=settings.render_dpi,
            max_edge=settings.render_max_edge,
            max_pages=settings.max_pages,
            max_bytes=settings.max_upload_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to render PDF for LightOnOCR.") from exc

    pages: list[dict[str, object]] = []
    for page_index, image in enumerate(images, start=1):
        try:
            text = lightonocr.ocr_image(image)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LightOnOCR failed on page {page_index}.") from exc
        normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
        pages.append({"page": page_index, "markdown": normalized})

    payload = {
        "meta": {
            "backend": "lightonocr",
            "page_count": int(total_pages),
            "render_dpi": int(settings.render_dpi),
            "render_max_edge": int(settings.render_max_edge),
        },
        "pages": pages,
    }
    return json.dumps(payload, ensure_ascii=False)


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

    if settings.pdf_backend == "lightonocr":
        markdown = _pdf_to_markdown_lightonocr(pdf_bytes, settings=settings)
    else:
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


def pdf_to_json(
    pdf_bytes: bytes,
    *,
    on_progress: JsonProgressCallback | None = None,
    page_output_dir: Path | None = None,
) -> str:
    """
    Convert a PDF (bytes) into a JSON string.

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

    _notify_json_progress(
        on_progress,
        PdfToJsonProgress(
            phase=ProgressPhase.rendering,
            current=0,
            total=total_pages,
            message="converting",
        ),
    )

    if settings.pdf_backend == "lightonocr":
        json_output = _pdf_to_json_lightonocr(pdf_bytes, settings=settings)
    else:
        json_output = pdf_bytes_to_json(pdf_bytes)

    if page_output_dir is not None:
        _write_text_artifact(page_output_dir / "document.json", json_output)

    _notify_json_progress(
        on_progress,
        PdfToJsonProgress(
            phase=ProgressPhase.done,
            current=total_pages,
            total=total_pages,
            message="done",
        ),
    )
    return json_output


from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ragprep.config import Settings, get_settings

logger = logging.getLogger(__name__)


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


def _write_text_artifact(path: Path, text: str) -> None:
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _pdf_to_markdown_lightonocr(
    pdf_bytes: bytes,
    *,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_page: PageCallback | None = None,
) -> str:
    from ragprep.diagnostics import record_last_activity, record_last_error, summarize_base64
    from ragprep.ocr import lightonocr
    from ragprep.pdf_render import iter_pdf_page_png_base64

    run_id = uuid.uuid4().hex
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
        record_last_error(
            {
                "run_id": run_id,
                "stage": "render",
                "page_index": None,
                "pdf_bytes_len": len(pdf_bytes),
                "settings": settings.__dict__,
                "pid": os.getpid(),
                "error": str(exc),
            }
        )
        raise RuntimeError("Failed to render PDF for LightOnOCR.") from exc

    record_last_activity(
        {
            "run_id": run_id,
            "stage": "start",
            "page_index": 0,
            "total_pages": int(total_pages),
            "pdf_bytes_len": len(pdf_bytes),
            "settings": settings.__dict__,
            "pid": os.getpid(),
        }
    )

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
    encoded_iter = iter(encoded_pages)
    page_index = 0
    while True:
        try:
            encoded = next(encoded_iter)
        except StopIteration:
            break
        except Exception as exc:  # noqa: BLE001
            failing_page = page_index + 1
            record_last_error(
                {
                    "run_id": run_id,
                    "stage": "render",
                    "page_index": failing_page,
                    "pdf_bytes_len": len(pdf_bytes),
                    "settings": settings.__dict__,
                    "pid": os.getpid(),
                    "error": str(exc),
                }
            )
            raise RuntimeError(
                f"Failed to render PDF page {failing_page} for LightOnOCR: {exc}"
            ) from exc

        page_index += 1
        image_meta = summarize_base64(encoded)
        record_last_activity(
            {
                "run_id": run_id,
                "stage": "ocr",
                "page_index": page_index,
                "total_pages": int(total_pages),
                "pdf_bytes_len": len(pdf_bytes),
                "pid": os.getpid(),
                "settings": settings.__dict__,
                **image_meta,
            }
        )

        try:
            text = lightonocr.ocr_image_base64(encoded)
        except Exception as exc:  # noqa: BLE001
            record_last_error(
                {
                    "run_id": run_id,
                    "stage": "ocr",
                    "page_index": page_index,
                    "total_pages": int(total_pages),
                    "pdf_bytes_len": len(pdf_bytes),
                    "pid": os.getpid(),
                    "settings": settings.__dict__,
                    **image_meta,
                    "error": str(exc),
                }
            )
            raise RuntimeError(f"LightOnOCR failed on page {page_index}: {exc}") from exc
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
    record_last_activity(
        {
            "run_id": run_id,
            "stage": "done",
            "page_index": int(total_pages),
            "total_pages": int(total_pages),
            "pdf_bytes_len": len(pdf_bytes),
            "settings": settings.__dict__,
            "pid": os.getpid(),
        }
    )

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
    in a single pass via LightOnOCR (GGUF).
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

    markdown = _pdf_to_markdown_lightonocr(
        pdf_bytes,
        settings=settings,
        on_progress=on_progress,
        on_page=on_page,
    )

    if page_output_dir is not None:
        _write_text_artifact(page_output_dir / "document.md", markdown)

    return markdown


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

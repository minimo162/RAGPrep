from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable, Iterator
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
    from ragprep.pdf_render import (
        downsample_png_base64,
        iter_pdf_page_png_base64,
        render_pdf_page_png_base64,
    )

    run_id = uuid.uuid4().hex

    def _get_non_negative_int_env(name: str, *, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an int, got: {raw!r}") from exc
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got: {value}")
        return value

    def _render_fallback_candidates(dpi: int, max_edge: int) -> list[tuple[int, int]]:
        min_dpi = 72
        min_edge = 400
        candidates: list[tuple[int, int]] = []

        for factor in (1.0, 0.75, 0.5):
            candidates.append(
                (
                    max(min_dpi, int(dpi * factor)),
                    max(min_edge, int(max_edge * factor)),
                )
            )

        for edge in (1200, 1000, 800, 600):
            if edge < max_edge:
                candidates.append((min(dpi, 300), edge))

        deduped: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _ocr_downsample_edges(current_max_edge: int) -> list[int]:
        candidates = [
            max(400, int(current_max_edge * 0.8)),
            max(400, int(current_max_edge * 0.6)),
            max(400, int(current_max_edge * 0.5)),
            1200,
            1000,
            800,
            600,
        ]
        filtered: list[int] = []
        seen_edges: set[int] = set()
        for edge in candidates:
            if edge >= current_max_edge:
                continue
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            filtered.append(edge)
        return filtered

    ocr_max_image_bytes = _get_non_negative_int_env(
        "RAGPREP_OCR_MAX_IMAGE_BYTES", default=6 * 1024 * 1024
    )

    render_dpi = int(settings.render_dpi)
    render_max_edge = int(settings.render_max_edge)
    effective_dpi = render_dpi
    effective_max_edge = render_max_edge

    def _iter_render(dpi: int, max_edge: int) -> tuple[int, Iterator[str]]:
        return iter_pdf_page_png_base64(
            pdf_bytes,
            dpi=dpi,
            max_edge=max_edge,
            max_pages=settings.max_pages,
            max_bytes=settings.max_upload_bytes,
        )

    def _render_page_with_fallback(page: int, *, dpi: int, max_edge: int) -> tuple[str, int, int]:
        last_exc: Exception | None = None
        for candidate_dpi, candidate_edge in _render_fallback_candidates(dpi, max_edge):
            try:
                _total_pages, encoded = render_pdf_page_png_base64(
                    pdf_bytes,
                    page_index=page,
                    dpi=candidate_dpi,
                    max_edge=candidate_edge,
                    max_pages=settings.max_pages,
                    max_bytes=settings.max_upload_bytes,
                )
                return encoded, candidate_dpi, candidate_edge
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        message = str(last_exc) if last_exc else "unknown error"
        raise RuntimeError(message)

    try:
        total_pages, encoded_pages = _iter_render(effective_dpi, effective_max_edge)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        last_exc: Exception = exc
        render_candidates = _render_fallback_candidates(effective_dpi, effective_max_edge)
        for candidate_dpi, candidate_edge in render_candidates[1:]:
            try:
                total_pages, encoded_pages = _iter_render(candidate_dpi, candidate_edge)
                effective_dpi = candidate_dpi
                effective_max_edge = candidate_edge
                break
            except Exception as inner:  # noqa: BLE001
                last_exc = inner
        else:
            record_last_error(
                {
                    "run_id": run_id,
                    "stage": "render",
                    "page_index": None,
                    "pdf_bytes_len": len(pdf_bytes),
                    "settings": settings.__dict__,
                    "pid": os.getpid(),
                    "render_dpi": effective_dpi,
                    "render_max_edge": effective_max_edge,
                    "error": str(last_exc),
                }
            )
            raise RuntimeError(
                "Failed to render PDF for LightOnOCR. "
                "Hint: try lowering RAGPREP_RENDER_DPI and/or RAGPREP_RENDER_MAX_EDGE."
            ) from last_exc

    record_last_activity(
        {
            "run_id": run_id,
            "stage": "start",
            "page_index": 0,
            "total_pages": int(total_pages),
            "pdf_bytes_len": len(pdf_bytes),
            "settings": settings.__dict__,
            "pid": os.getpid(),
            "render_dpi": effective_dpi,
            "render_max_edge": effective_max_edge,
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
    use_per_page_render = False

    def _downsample_for_payload_limit(
        encoded: str,
        *,
        current_max_edge: int,
        page: int,
    ) -> str:
        if ocr_max_image_bytes <= 0:
            return encoded

        meta = summarize_base64(encoded)
        if int(meta.get("estimated_bytes", 0)) <= ocr_max_image_bytes:
            return encoded

        current = encoded
        for edge in _ocr_downsample_edges(current_max_edge):
            candidate = downsample_png_base64(current, max_edge=edge)
            if candidate == current:
                continue
            candidate_meta = summarize_base64(candidate)
            record_last_activity(
                {
                    "run_id": run_id,
                    "stage": "ocr_payload_downsample",
                    "page_index": page,
                    "total_pages": int(total_pages),
                    "pdf_bytes_len": len(pdf_bytes),
                    "pid": os.getpid(),
                    "settings": settings.__dict__,
                    "render_dpi": effective_dpi,
                    "render_max_edge": current_max_edge,
                    "ocr_max_image_bytes": ocr_max_image_bytes,
                    "ocr_downsample_max_edge": edge,
                    "base64_len_original": int(meta.get("base64_len", 0)),
                    "estimated_bytes_original": int(meta.get("estimated_bytes", 0)),
                    **candidate_meta,
                }
            )
            current = candidate
            meta = candidate_meta
            if int(meta.get("estimated_bytes", 0)) <= ocr_max_image_bytes:
                break
        return current

    def _ocr_with_fallbacks(
        encoded: str,
        *,
        current_max_edge: int,
        page: int,
    ) -> str:
        last_exc: Exception | None = None
        try:
            return lightonocr.ocr_image_base64(encoded)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

        current = encoded
        for edge in _ocr_downsample_edges(current_max_edge):
            candidate = downsample_png_base64(current, max_edge=edge)
            if candidate == current:
                continue
            candidate_meta = summarize_base64(candidate)
            record_last_activity(
                {
                    "run_id": run_id,
                    "stage": "ocr_fallback",
                    "page_index": page,
                    "total_pages": int(total_pages),
                    "pdf_bytes_len": len(pdf_bytes),
                    "pid": os.getpid(),
                    "settings": settings.__dict__,
                    "render_dpi": effective_dpi,
                    "render_max_edge": current_max_edge,
                    "ocr_downsample_max_edge": edge,
                    **candidate_meta,
                }
            )
            current = candidate
            try:
                return lightonocr.ocr_image_base64(current)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OCR failed (unknown error)")

    for page_index in range(1, int(total_pages) + 1):
        try:
            if use_per_page_render:
                raise StopIteration
            encoded = next(encoded_iter)
        except StopIteration:
            use_per_page_render = True
            try:
                encoded, effective_dpi, effective_max_edge = _render_page_with_fallback(
                    page_index,
                    dpi=effective_dpi,
                    max_edge=effective_max_edge,
                )
            except Exception as exc:  # noqa: BLE001
                record_last_error(
                    {
                        "run_id": run_id,
                        "stage": "render",
                        "page_index": page_index,
                        "pdf_bytes_len": len(pdf_bytes),
                        "settings": settings.__dict__,
                        "pid": os.getpid(),
                        "render_dpi": effective_dpi,
                        "render_max_edge": effective_max_edge,
                        "error": str(exc),
                    }
                )
                raise RuntimeError(
                    f"Failed to render PDF page {page_index} for LightOnOCR: {exc}. "
                    "Hint: try lowering RAGPREP_RENDER_DPI and/or RAGPREP_RENDER_MAX_EDGE."
                ) from exc
        except Exception as exc:  # noqa: BLE001
            use_per_page_render = True
            try:
                encoded, effective_dpi, effective_max_edge = _render_page_with_fallback(
                    page_index,
                    dpi=effective_dpi,
                    max_edge=effective_max_edge,
                )
            except Exception as inner:  # noqa: BLE001
                record_last_error(
                    {
                        "run_id": run_id,
                        "stage": "render",
                        "page_index": page_index,
                        "pdf_bytes_len": len(pdf_bytes),
                        "settings": settings.__dict__,
                        "pid": os.getpid(),
                        "render_dpi": effective_dpi,
                        "render_max_edge": effective_max_edge,
                        "error": str(inner),
                        "root_cause": str(exc),
                    }
                )
                raise RuntimeError(
                    f"Failed to render PDF page {page_index} for LightOnOCR: {inner}. "
                    "Hint: try lowering RAGPREP_RENDER_DPI and/or RAGPREP_RENDER_MAX_EDGE."
                ) from inner

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
                "render_dpi": effective_dpi,
                "render_max_edge": effective_max_edge,
                **image_meta,
            }
        )

        encoded_for_ocr = _downsample_for_payload_limit(
            encoded,
            current_max_edge=effective_max_edge,
            page=page_index,
        )
        sent_meta = summarize_base64(encoded_for_ocr)

        try:
            text = _ocr_with_fallbacks(
                encoded_for_ocr,
                current_max_edge=effective_max_edge,
                page=page_index,
            )
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
                    "render_dpi": effective_dpi,
                    "render_max_edge": effective_max_edge,
                    "ocr_max_image_bytes": ocr_max_image_bytes,
                    "base64_len_sent": int(sent_meta.get("base64_len", 0)),
                    "estimated_bytes_sent": int(sent_meta.get("estimated_bytes", 0)),
                    "error": str(exc),
                }
            )
            raise RuntimeError(
                f"LightOnOCR failed on page {page_index}: {exc}\n"
                "\n"
                "Hint: try lowering RAGPREP_RENDER_DPI and/or RAGPREP_RENDER_MAX_EDGE, "
                "or lowering RAGPREP_OCR_MAX_IMAGE_BYTES. "
                "Also verify LIGHTONOCR_MODEL and consider increasing "
                "LIGHTONOCR_REQUEST_TIMEOUT_SECONDS."
            ) from exc

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
            "render_dpi": effective_dpi,
            "render_max_edge": effective_max_edge,
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

from __future__ import annotations

import difflib
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image

from ragprep.config import get_settings
from ragprep.markdown_table import TableBlock, parse_markdown_blocks
from ragprep.ocr.lightonocr import ocr_image
from ragprep.pdf_render import _import_fitz, _pixmap_to_rgb_image, iter_pdf_images
from ragprep.pdf_text import (
    PageKind,
    _extract_image_signals,
    _extract_words,
    _safe_find_tables_score,
    classify_page_kind,
    estimate_table_likelihood,
    normalize_extracted_text,
    score_text_quality,
    tokenize_by_char_class,
)
from ragprep.pymupdf4llm_markdown import pdf_bytes_to_markdown
from ragprep.table_merge import TableMergeStats, merge_markdown_tables_with_pymupdf_words
from ragprep.text_merge import MergeStats, merge_ocr_with_pymupdf

_DEFAULT_SKIP_OCR_MIN_SCORE = 0.85
_DEFAULT_TABLE_OCR_LIKELIHOOD_MIN = 0.45
_DEFAULT_MERGE_MIN_SCORE = 0.55
_DEFAULT_TABLE_MERGE_MIN_SCORE = 0.15
_FORCE_OCR_ALL_PAGES = True


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


def _requires_ocr_for_page(
    page_kind: PageKind,
    *,
    text_quality_score: float,
    table_likelihood: float,
) -> bool:
    if page_kind in (PageKind.table, PageKind.image, PageKind.mixed, PageKind.empty):
        return True
    if table_likelihood >= _DEFAULT_TABLE_OCR_LIKELIHOOD_MIN:
        return True
    return text_quality_score < _DEFAULT_SKIP_OCR_MIN_SCORE


def _render_page_to_image(page: Any, *, fitz: Any, matrix: Any, max_edge: int) -> Image.Image:
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
    rgb = _pixmap_to_rgb_image(pix)

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

    return rgb


def _write_text_artifact(path: Path, text: str) -> None:
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _estimate_replace_counts(a_text: str, b_text: str) -> tuple[int, int]:
    a = normalize_extracted_text(a_text)
    b = normalize_extracted_text(b_text)

    a_tokens = tokenize_by_char_class(a)
    b_tokens = tokenize_by_char_class(b)
    if not a_tokens or not b_tokens:
        return 0, 0

    replaced_tokens_estimated = 0
    replaced_chars_estimated = 0
    matcher = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        replaced_tokens_estimated += max(i2 - i1, j2 - j1)
        replaced_chars_estimated += max(
            len("".join(a_tokens[i1:i2])),
            len("".join(b_tokens[j1:j2])),
        )
    return replaced_tokens_estimated, replaced_chars_estimated


def _short_unified_diff(a_text: str, b_text: str, *, max_lines: int = 60) -> list[str]:
    try:
        lines = list(
            difflib.unified_diff(
                a_text.splitlines(),
                b_text.splitlines(),
                fromfile="ocr",
                tofile="pymupdf",
                lineterm="",
            )
        )
    except Exception:  # noqa: BLE001
        return []
    if len(lines) > max_lines:
        return lines[:max_lines] + ["...(truncated)..."]
    return lines


def _pdf_to_markdown_ocr_legacy(
    pdf_bytes: bytes,
    *,
    on_progress: ProgressCallback | None = None,
    page_output_dir: Path | None = None,
) -> str:
    """
    Convert a PDF (bytes) into a Markdown/text string.

    This function is intentionally pure and synchronous; it orchestrates PDF rendering
    and per-page OCR, and applies small deterministic post-processing.
    """

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    if page_output_dir is not None:
        page_output_dir.mkdir(parents=True, exist_ok=True)

    _notify_progress(
        on_progress,
        PdfToMarkdownProgress(
            phase=ProgressPhase.rendering,
            current=0,
            total=0,
            message="rendering",
        ),
    )

    settings = get_settings()

    doc = None
    fitz = None
    total_pages = 0
    try:
        if len(pdf_bytes) > settings.max_upload_bytes:
            raise ValueError(
                f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={settings.max_upload_bytes}"
            )
        fitz = _import_fitz()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = int(doc.page_count)
        if total_pages > settings.max_pages:
            doc.close()
            raise ValueError(f"PDF has {total_pages} pages, max_pages={settings.max_pages}")
    except Exception:  # noqa: BLE001
        if doc is not None:
            try:
                doc.close()
            except Exception:  # noqa: BLE001
                pass
        doc = None

    page_texts: list[str] = []

    if doc is None:
        total_pages, images = iter_pdf_images(pdf_bytes)
        pad = max(4, len(str(total_pages)))

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

        for page_number, image in enumerate(images, start=1):
            page_prefix = f"page-{page_number:0{pad}d}"

            requires_ocr = True
            ocr_reason = "analysis_unavailable"
            pymupdf_text = ""
            page_kind = "unknown"

            ocr_text = normalize_extracted_text(ocr_image(image)).strip()
            merged_text = ocr_text
            selected_source = "ocr"
            merge_stats: MergeStats | None = None

            progress_message = f"page {page_number}"
            if merged_text:
                page_texts.append(merged_text)

            if page_output_dir is not None:
                ocr_path = page_output_dir / f"{page_prefix}.ocr.md"
                pymupdf_path = page_output_dir / f"{page_prefix}.pymupdf.md"
                merged_path = page_output_dir / f"{page_prefix}.merged.md"
                meta_path = page_output_dir / f"{page_prefix}.meta.json"

                _write_text_artifact(ocr_path, ocr_text)
                _write_text_artifact(pymupdf_path, pymupdf_text)
                _write_text_artifact(merged_path, merged_text)

                meta: dict[str, Any] = {
                    "page_number": page_number,
                    "page_kind": page_kind,
                    "analysis_available": False,
                    "selected_source": selected_source,
                    "ocr_required": bool(requires_ocr),
                    "ocr_skipped": not requires_ocr,
                    "ocr_reason": ocr_reason,
                    "pymupdf": {
                        "score": None,
                        "visible_char_count": None,
                        "replacement_char_ratio": None,
                        "symbol_ratio": None,
                    },
                    "table_likelihood": None,
                    "image_count": None,
                    "image_area_ratio": None,
                    "merge": {
                        "applied": merge_stats is not None,
                        "changed_chars": (
                            int(merge_stats.changed_char_count) if merge_stats is not None else 0
                        ),
                        "changed_tokens": (
                            int(merge_stats.changed_token_count) if merge_stats is not None else 0
                        ),
                        "applied_blocks": (
                            int(merge_stats.applied_block_count) if merge_stats is not None else 0
                        ),
                        "samples": list(merge_stats.samples) if merge_stats is not None else [],
                        "replacements": (
                            [{"before": b, "after": a} for (b, a) in merge_stats.replacements]
                            if merge_stats is not None
                            else []
                        ),
                    },
                    "diff_estimate": {"replaced_tokens": 0, "replaced_chars": 0},
                    "diff_preview": [],
                }
                _write_json_artifact(meta_path, meta)

            _notify_progress(
                on_progress,
                PdfToMarkdownProgress(
                    phase=ProgressPhase.ocr,
                    current=page_number,
                    total=total_pages,
                    message=progress_message,
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

    pad = max(4, len(str(total_pages)))

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

    assert fitz is not None
    dpi = settings.render_dpi
    max_edge = settings.render_max_edge
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with doc:
        for page_index in range(total_pages):
            page_number = page_index + 1
            page_prefix = f"page-{page_number:0{pad}d}"

            page = doc.load_page(page_index)

            raw_text = str(page.get_text("text") or "")
            normalized_text = normalize_extracted_text(raw_text)
            pymupdf_text = normalized_text.strip()
            has_text_layer = bool(pymupdf_text.strip())

            text_quality = score_text_quality(normalized_text)

            words = _extract_words(page)
            table_likelihood = estimate_table_likelihood(words)
            table_likelihood = max(table_likelihood, _safe_find_tables_score(page))

            image_count, image_area_ratio = _extract_image_signals(page)

            page_kind_obj = classify_page_kind(
                has_text_layer=has_text_layer,
                image_area_ratio=image_area_ratio,
                table_likelihood=table_likelihood,
            )

            if _FORCE_OCR_ALL_PAGES:
                requires_ocr = True
                ocr_reason = "forced_all_pages"
            else:
                requires_ocr = _requires_ocr_for_page(
                    page_kind_obj,
                    text_quality_score=text_quality.score,
                    table_likelihood=table_likelihood,
                )

                if page_kind_obj in (
                    PageKind.table,
                    PageKind.image,
                    PageKind.mixed,
                    PageKind.empty,
                ):
                    ocr_reason = f"page_kind={page_kind_obj.value}"
                elif table_likelihood >= _DEFAULT_TABLE_OCR_LIKELIHOOD_MIN:
                    ocr_reason = f"table_likelihood>={_DEFAULT_TABLE_OCR_LIKELIHOOD_MIN}"
                elif text_quality.score < _DEFAULT_SKIP_OCR_MIN_SCORE:
                    ocr_reason = f"text_quality<{_DEFAULT_SKIP_OCR_MIN_SCORE}"
                else:
                    ocr_reason = f"text_quality>={_DEFAULT_SKIP_OCR_MIN_SCORE}"

            ocr_text = ""
            merged_text = ""
            selected_source = "ocr"
            merge_stats = None
            merge_reason: str | None = None
            table_merge_stats: TableMergeStats | None = None
            ocr_markdown_table_detected = False
            ocr_markdown_table_count = 0
            selected_source_reason: str | None = None

            progress_message = f"page {page_number}"
            if requires_ocr:
                image = _render_page_to_image(page, fitz=fitz, matrix=matrix, max_edge=max_edge)
                ocr_text = normalize_extracted_text(ocr_image(image)).strip()
                merged_text = ocr_text
                if ocr_text:
                    blocks = parse_markdown_blocks(ocr_text)
                    ocr_markdown_table_count = sum(1 for b in blocks if isinstance(b, TableBlock))
                    ocr_markdown_table_detected = ocr_markdown_table_count > 0

                if not ocr_markdown_table_detected:
                    table_merge_stats = TableMergeStats(
                        applied=False,
                        changed_cells=0,
                        changed_chars=0,
                        confidence=None,
                        reason="no_table",
                    )
                elif not has_text_layer:
                    table_merge_stats = TableMergeStats(
                        applied=False,
                        changed_cells=0,
                        changed_chars=0,
                        confidence=None,
                        reason="no_text_layer",
                    )
                elif text_quality.score < _DEFAULT_TABLE_MERGE_MIN_SCORE:
                    table_merge_stats = TableMergeStats(
                        applied=False,
                        changed_cells=0,
                        changed_chars=0,
                        confidence=None,
                        reason=f"text_quality<{_DEFAULT_TABLE_MERGE_MIN_SCORE}",
                    )
                else:
                    merge_reason = "ocr_markdown_table_detected"
                    merged_table_text, table_merge_stats = merge_markdown_tables_with_pymupdf_words(
                        ocr_text, words
                    )
                    if table_merge_stats.applied:
                        merged_text = merged_table_text

                if (
                    table_merge_stats is not None
                    and table_merge_stats.reason == "no_table"
                    and has_text_layer
                ):
                    merged_text, merge_stats = merge_ocr_with_pymupdf(ocr_text, normalized_text)
                    if merge_stats.changed_char_count:
                        selected_source = "merged"
                        merge_reason = f"changed_chars={merge_stats.changed_char_count}"
                        progress_message = (
                            f"page {page_number} (merged {merge_stats.changed_char_count})"
                        )
                    else:
                        merge_reason = "attempted_no_changes"

                if merge_reason is None:
                    if not has_text_layer:
                        merge_reason = "no_text_layer"
                    else:
                        merge_reason = f"page_kind={page_kind_obj.value}"
                if selected_source != "merged":
                    selected_source = "ocr"
                if table_merge_stats is not None and table_merge_stats.applied:
                    selected_source_reason = "table_merge_applied"
                elif selected_source == "merged":
                    selected_source_reason = "text_merge_applied"
                else:
                    selected_source_reason = "ocr"
                if merged_text:
                    page_texts.append(merged_text)
            else:
                merged_text = pymupdf_text
                selected_source = "pymupdf"
                selected_source_reason = "ocr_skipped"
                merge_reason = "ocr_skipped"
                if merged_text:
                    page_texts.append(merged_text)
                progress_message = f"page {page_number} (skip ocr)"

            if page_output_dir is not None:
                ocr_path = page_output_dir / f"{page_prefix}.ocr.md"
                pymupdf_path = page_output_dir / f"{page_prefix}.pymupdf.md"
                merged_path = page_output_dir / f"{page_prefix}.merged.md"
                meta_path = page_output_dir / f"{page_prefix}.meta.json"

                _write_text_artifact(ocr_path, ocr_text)
                _write_text_artifact(pymupdf_path, pymupdf_text)
                _write_text_artifact(merged_path, merged_text)

                replaced_tokens_estimated = 0
                replaced_chars_estimated = 0
                diff_preview: list[str] = []
                if ocr_text and pymupdf_text:
                    replaced_tokens_estimated, replaced_chars_estimated = _estimate_replace_counts(
                        ocr_text, pymupdf_text
                    )
                    diff_preview = _short_unified_diff(ocr_text, pymupdf_text)

                table_merge_meta: dict[str, Any] = {
                    "ocr_markdown_table_detected": bool(ocr_markdown_table_detected),
                    "ocr_markdown_table_count": int(ocr_markdown_table_count),
                    "attempted": table_merge_stats is not None,
                    "applied": bool(table_merge_stats.applied) if table_merge_stats else False,
                    "changed_cells": (
                        int(table_merge_stats.changed_cells) if table_merge_stats is not None else 0
                    ),
                    "changed_chars": (
                        int(table_merge_stats.changed_chars) if table_merge_stats is not None else 0
                    ),
                    "confidence": (
                        float(table_merge_stats.confidence)
                        if (
                            table_merge_stats is not None
                            and table_merge_stats.confidence is not None
                        )
                        else None
                    ),
                    "reason": table_merge_stats.reason if table_merge_stats is not None else None,
                    "samples": list(table_merge_stats.samples)
                    if table_merge_stats is not None
                    else [],
                }

                meta = {
                    "page_number": page_number,
                    "page_kind": page_kind_obj.value,
                    "analysis_available": True,
                    "selected_source": selected_source,
                    "ocr_required": bool(requires_ocr),
                    "ocr_skipped": not requires_ocr,
                    "ocr_reason": ocr_reason,
                    "pymupdf": {
                        "score": text_quality.score,
                        "visible_char_count": text_quality.visible_char_count,
                        "replacement_char_ratio": text_quality.replacement_char_ratio,
                        "symbol_ratio": text_quality.symbol_ratio,
                    },
                    "table_likelihood": table_likelihood,
                    "image_count": image_count,
                    "image_area_ratio": image_area_ratio,
                    "selected_source_reason": selected_source_reason,
                    "merge": {
                        "applied": merge_stats is not None,
                        "reason": merge_reason,
                        "used": selected_source == "merged",
                        "changed_chars": (
                            int(merge_stats.changed_char_count) if merge_stats is not None else 0
                        ),
                        "changed_tokens": (
                            int(merge_stats.changed_token_count) if merge_stats is not None else 0
                        ),
                        "applied_blocks": (
                            int(merge_stats.applied_block_count) if merge_stats is not None else 0
                        ),
                        "samples": list(merge_stats.samples) if merge_stats is not None else [],
                        "replacements": (
                            [{"before": b, "after": a} for (b, a) in merge_stats.replacements]
                            if merge_stats is not None
                            else []
                        ),
                    },
                    "table_merge": table_merge_meta,
                    "diff_estimate": {
                        "replaced_tokens": int(replaced_tokens_estimated),
                        "replaced_chars": int(replaced_chars_estimated),
                    },
                    "diff_preview": diff_preview,
                }
                _write_json_artifact(meta_path, meta)

            _notify_progress(
                on_progress,
                PdfToMarkdownProgress(
                    phase=ProgressPhase.ocr,
                    current=page_number,
                    total=total_pages,
                    message=progress_message,
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

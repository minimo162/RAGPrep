from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ragprep.ocr.lightonocr import ocr_image
from ragprep.pdf_render import iter_pdf_images
from ragprep.pdf_text import (
    PageKind,
    analyze_pdf_pages,
    normalize_extracted_text,
    tokenize_by_char_class,
)

_DEFAULT_SKIP_OCR_MIN_SCORE = 0.85
_DEFAULT_TABLE_OCR_LIKELIHOOD_MIN = 0.45
_DEFAULT_MERGE_MIN_SCORE = 0.55
_DEFAULT_MERGE_MAX_CHANGED_RATIO = 0.08

_REPLACEMENT_CHAR = "\ufffd"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://|www\\.", flags=re.IGNORECASE)


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


@dataclass(frozen=True)
class MergeStats:
    changed_char_count: int
    changed_token_count: int
    applied_block_count: int
    samples: tuple[str, ...] = ()
    replacements: tuple[tuple[str, str], ...] = ()


def _is_japanese_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        (0x3040 <= code <= 0x309F)  # hiragana
        or (0x30A0 <= code <= 0x30FF)  # katakana
        or (0xFF66 <= code <= 0xFF9D)  # halfwidth katakana
        or (0x4E00 <= code <= 0x9FFF)  # kanji
    )


def _looks_like_url_or_email(text: str) -> bool:
    return bool(_URL_RE.search(text) or _EMAIL_RE.search(text))


def _token_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    for token in tokens:
        while i < len(text) and text[i].isspace():
            i += 1
        if not text.startswith(token, i):
            pos = text.find(token, i)
            if pos < 0:
                break
            i = pos
        start = i
        end = i + len(token)
        spans.append((start, end))
        i = end
    return spans


def merge_ocr_with_pymupdf(ocr_text: str, pymupdf_text: str) -> tuple[str, MergeStats]:
    """
    Merge OCR output with PyMuPDF text layer as a corrective reference.

    Policy (strict and deterministic):
    - Preserve OCR whitespace/line breaks.
    - Apply only small, local substitutions (e.g., 1-kanji corrections).
    - Prefer PyMuPDF for Japanese characters and for fixing replacement chars.
    - Prefer OCR for digits/latin/symbols, and for URL/email-like spans.
    """

    ocr_normalized = normalize_extracted_text(ocr_text).strip()
    pymupdf_normalized = normalize_extracted_text(pymupdf_text).strip()

    if not ocr_normalized or not pymupdf_normalized:
        return ocr_normalized, MergeStats(0, 0, 0)

    ocr_tokens = tokenize_by_char_class(ocr_normalized)
    pymupdf_tokens = tokenize_by_char_class(pymupdf_normalized)

    if not ocr_tokens or not pymupdf_tokens:
        return ocr_normalized, MergeStats(0, 0, 0)

    spans = _token_spans(ocr_normalized, ocr_tokens)
    if len(spans) != len(ocr_tokens):
        return ocr_normalized, MergeStats(0, 0, 0)

    out_chars = list(ocr_normalized)
    changed_char_count = 0
    changed_token_count = 0
    applied_block_count = 0
    samples: list[str] = []
    replacements: list[tuple[str, str]] = []

    matcher = difflib.SequenceMatcher(a=ocr_tokens, b=pymupdf_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        if i1 >= i2 or j1 >= j2:
            continue

        start = spans[i1][0]
        end = spans[i2 - 1][1]
        template = ocr_normalized[start:end]

        ocr_compact = "".join(ocr_tokens[i1:i2])
        pym_compact = "".join(pymupdf_tokens[j1:j2])
        if len(ocr_compact) != len(pym_compact):
            continue

        diff_count = sum(1 for a, b in zip(ocr_compact, pym_compact, strict=False) if a != b)
        if diff_count == 0:
            continue

        max_changed = max(1, int(len(ocr_compact) * _DEFAULT_MERGE_MAX_CHANGED_RATIO))
        if diff_count > max_changed:
            continue

        if _looks_like_url_or_email(ocr_compact) or _looks_like_url_or_email(pym_compact):
            continue

        non_ws_count = sum(1 for ch in template if not ch.isspace())
        if non_ws_count != len(ocr_compact):
            continue

        merged_compact_chars: list[str] = []
        block_changed_chars = 0
        for o, p in zip(ocr_compact, pym_compact, strict=False):
            if o == p:
                merged_compact_chars.append(o)
                continue
            if o == _REPLACEMENT_CHAR and p != _REPLACEMENT_CHAR:
                merged_compact_chars.append(p)
            elif p == _REPLACEMENT_CHAR:
                merged_compact_chars.append(o)
            elif _is_japanese_char(p) or _is_japanese_char(o):
                merged_compact_chars.append(p)
            else:
                merged_compact_chars.append(o)

            if merged_compact_chars[-1] != o:
                block_changed_chars += 1

        if block_changed_chars == 0:
            continue

        merged_compact = "".join(merged_compact_chars)
        merged_segment_chars: list[str] = []
        compact_i = 0
        for ch in template:
            if ch.isspace():
                merged_segment_chars.append(ch)
                continue
            merged_segment_chars.append(merged_compact[compact_i])
            compact_i += 1
        merged_segment = "".join(merged_segment_chars)

        out_chars[start:end] = list(merged_segment)
        changed_char_count += block_changed_chars
        changed_token_count += i2 - i1
        applied_block_count += 1
        if len(samples) < 3:
            samples.append(f"{ocr_compact[:40]} -> {merged_compact[:40]}")
        if len(replacements) < 20:
            before = ocr_compact if len(ocr_compact) <= 80 else (ocr_compact[:80] + "…")
            after = merged_compact if len(merged_compact) <= 80 else (merged_compact[:80] + "…")
            replacements.append((before, after))

        merged_text = "".join(out_chars).strip()
    return (
        merged_text,
        MergeStats(
            changed_char_count=changed_char_count,
            changed_token_count=changed_token_count,
            applied_block_count=applied_block_count,
            samples=tuple(samples),
            replacements=tuple(replacements),
        ),
    )


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


def pdf_to_markdown(
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

    page_analyses = None
    try:
        page_analyses = analyze_pdf_pages(pdf_bytes, use_find_tables=True)
    except Exception:  # noqa: BLE001
        page_analyses = None

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
    total_pages, images = iter_pdf_images(pdf_bytes)
    pad = max(4, len(str(total_pages)))
    if page_analyses is not None and len(page_analyses) != total_pages:
        page_analyses = None
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
        page_prefix = f"page-{page_number:0{pad}d}"

        analysis = page_analyses[page_number - 1] if page_analyses is not None else None
        requires_ocr = True
        ocr_reason = "analysis_unavailable"
        if analysis is not None:
            requires_ocr = _requires_ocr_for_page(
                analysis.page_kind,
                text_quality_score=analysis.text_quality.score,
                table_likelihood=analysis.table_likelihood,
            )
            if analysis.page_kind in (
                PageKind.table,
                PageKind.image,
                PageKind.mixed,
                PageKind.empty,
            ):
                ocr_reason = f"page_kind={analysis.page_kind.value}"
            elif analysis.table_likelihood >= _DEFAULT_TABLE_OCR_LIKELIHOOD_MIN:
                ocr_reason = f"table_likelihood>={_DEFAULT_TABLE_OCR_LIKELIHOOD_MIN}"
            elif analysis.text_quality.score < _DEFAULT_SKIP_OCR_MIN_SCORE:
                ocr_reason = f"text_quality<{_DEFAULT_SKIP_OCR_MIN_SCORE}"
            else:
                ocr_reason = f"text_quality>={_DEFAULT_SKIP_OCR_MIN_SCORE}"

        pymupdf_text = analysis.normalized_text.strip() if analysis is not None else ""
        ocr_text = ""
        merged_text = ""
        selected_source = "ocr"
        merge_stats: MergeStats | None = None

        progress_message = f"page {page_number}"
        if requires_ocr:
            ocr_text = normalize_extracted_text(ocr_image(image)).strip()
            merged_text = ocr_text
            if analysis is not None and analysis.page_kind == PageKind.text:
                if (
                    analysis.text_quality.score >= _DEFAULT_MERGE_MIN_SCORE
                    and analysis.table_likelihood < _DEFAULT_TABLE_OCR_LIKELIHOOD_MIN
                ):
                    merged_text, merge_stats = merge_ocr_with_pymupdf(
                        ocr_text, analysis.normalized_text
                    )
                    if merge_stats.changed_char_count:
                        selected_source = "merged"
                        progress_message = (
                            f"page {page_number} (merged {merge_stats.changed_char_count})"
                        )
            if selected_source != "merged":
                selected_source = "ocr"
            if merged_text:
                page_texts.append(merged_text)
        else:
            merged_text = pymupdf_text
            selected_source = "pymupdf"
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

            meta: dict[str, Any] = {
                "page_number": page_number,
                "page_kind": analysis.page_kind.value if analysis is not None else "unknown",
                "analysis_available": analysis is not None,
                "selected_source": selected_source,
                "ocr_required": bool(requires_ocr),
                "ocr_skipped": not requires_ocr,
                "ocr_reason": ocr_reason,
                "pymupdf": {
                    "score": analysis.text_quality.score if analysis is not None else None,
                    "visible_char_count": (
                        analysis.text_quality.visible_char_count if analysis is not None else None
                    ),
                    "replacement_char_ratio": (
                        analysis.text_quality.replacement_char_ratio
                        if analysis is not None
                        else None
                    ),
                    "symbol_ratio": analysis.text_quality.symbol_ratio
                    if analysis is not None
                    else None,
                },
                "table_likelihood": analysis.table_likelihood if analysis is not None else None,
                "image_count": analysis.image_count if analysis is not None else None,
                "image_area_ratio": analysis.image_area_ratio if analysis is not None else None,
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

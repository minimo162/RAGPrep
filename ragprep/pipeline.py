from __future__ import annotations

import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

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

    merged_text = "".join(out_chars).strip()
    return (
        merged_text,
        MergeStats(
            changed_char_count=changed_char_count,
            changed_token_count=changed_token_count,
            applied_block_count=applied_block_count,
            samples=tuple(samples),
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


def pdf_to_markdown(pdf_bytes: bytes, *, on_progress: ProgressCallback | None = None) -> str:
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
        analysis = page_analyses[page_number - 1] if page_analyses is not None else None
        requires_ocr = True
        if analysis is not None:
            requires_ocr = _requires_ocr_for_page(
                analysis.page_kind,
                text_quality_score=analysis.text_quality.score,
                table_likelihood=analysis.table_likelihood,
            )

        progress_message = f"page {page_number}"
        if requires_ocr:
            ocr_text = normalize_extracted_text(ocr_image(image)).strip()
            merged_text = ocr_text
            merge_stats = None
            if analysis is not None and analysis.page_kind == PageKind.text:
                if (
                    analysis.text_quality.score >= _DEFAULT_MERGE_MIN_SCORE
                    and analysis.table_likelihood < _DEFAULT_TABLE_OCR_LIKELIHOOD_MIN
                ):
                    merged_text, merge_stats = merge_ocr_with_pymupdf(
                        ocr_text, analysis.normalized_text
                    )
                    if merge_stats.changed_char_count:
                        progress_message = (
                            f"page {page_number} (merged {merge_stats.changed_char_count})"
                        )
            if merged_text:
                page_texts.append(merged_text)
        else:
            text_layer = analysis.normalized_text.strip() if analysis is not None else ""
            if text_layer:
                page_texts.append(text_layer)
            progress_message = f"page {page_number} (skip ocr)"

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

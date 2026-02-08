from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import cast

from ragprep.config import Settings, get_settings
from ragprep.pdf_text import Span, Word


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


_DIGIT_RE = re.compile(r"\d+")
_NAV_LIKE_RE = re.compile(r"(contents|section\s*\d+|(?:\b\d{2}\b(?:\s+\b\d{2}\b){2,}))", re.I)
_PAGE_TOKEN_RE = re.compile(r"(?<!\d)\d{1,3}(?!\d)")
_COMPACT_PAGE_RUN_RE = re.compile(r"(?<!\d)\d{4,7}(?!\d)")
_LEADING_PAGE_TOKEN_RE = re.compile(r"^\s*(?:[pｐ]\.?\s*)?\d{1,3}(?!\d)")
_TOC_ROLE_RE = re.compile(r"\b(?:ceo|cfo|cto|csco|cso|chro|cio)\b|at a glance|メッセージ", re.I)

_TOC_NOISE_KEYWORDS: tuple[str, ...] = (
    "contents",
    "section",
    "価値創造",
    "データセクション",
    "ガバナンス",
    "サステナビリティ",
    "メッセージ",
    "ライトアセット",
    "市場別進捗レビュー",
    "ものづくり革新",
    "統合報告書",
    "ceo",
    "cfo",
    "cto",
    "csco",
    "cso",
    "chro",
    "cio",
    "at a glance",
)

_TOC_TITLE_HINTS: tuple[str, ...] = (
    "価値創造の道筋",
    "価値創造の実践",
    "価値創造の基盤",
    "データセクション",
    "人事戦略",
    "ライトアセット戦略",
    "サステナビリティ推進",
    "コーポレートガバナンス",
    "環境負荷ゼロへの挑戦",
)


def _normalize_margin_text(text: str) -> str:
    return " ".join(str(text).split()).strip().casefold()


def _template_margin_text(text: str) -> str:
    normalized = _normalize_margin_text(text)
    templated = _DIGIT_RE.sub("<n>", normalized)
    return re.sub(r"(?:<n>\s*){2,}", "<n> ", templated).strip()


def _looks_like_nav_header(text: str) -> bool:
    normalized = _normalize_margin_text(text)
    if len(normalized) > 220:
        return False
    if _NAV_LIKE_RE.search(normalized):
        return True
    score = 0
    if "価値創造" in normalized:
        score += 1
    if "データセクション" in normalized:
        score += 1
    if "マツダものづくり革新" in normalized:
        score += 1
    return score >= 2


def _median_positive(values: list[float], *, default: float) -> float:
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and v > 0 and v == v]
    if not cleaned:
        return float(default)
    try:
        return float(statistics.median(cleaned))
    except statistics.StatisticsError:
        return float(default)


def _join_row_text_for_margin(spans: list[Span]) -> str:
    if not spans:
        return ""

    ordered = sorted(
        spans,
        key=lambda s: (
            float(getattr(s, "x0", 0.0)),
            float(getattr(s, "x1", 0.0)),
            str(getattr(s, "text", "")),
        ),
    )
    heights = [
        max(0.0, float(getattr(s, "y1", 0.0)) - float(getattr(s, "y0", 0.0)))
        for s in ordered
    ]
    gap_threshold = max(1.0, _median_positive(heights, default=10.0) * 0.25)

    parts: list[str] = []
    prev = None
    for span in ordered:
        text = str(getattr(span, "text", "")).strip()
        if not text:
            continue
        if prev is not None:
            gap = float(getattr(span, "x0", 0.0)) - float(getattr(prev, "x1", 0.0))
            if gap >= gap_threshold:
                parts.append(" ")
        parts.append(text)
        prev = span
    return "".join(parts).strip()


def _cluster_page_spans_to_rows(spans: list[Span]) -> list[list[Span]]:
    if not spans:
        return []

    ordered = sorted(
        spans,
        key=lambda s: (
            float(getattr(s, "y0", 0.0)),
            float(getattr(s, "x0", 0.0)),
            float(getattr(s, "x1", 0.0)),
        ),
    )
    heights = [
        max(0.0, float(getattr(s, "y1", 0.0)) - float(getattr(s, "y0", 0.0)))
        for s in ordered
    ]
    line_threshold = max(1.5, _median_positive(heights, default=10.0) * 0.65)

    rows: list[list[Span]] = []
    for span in ordered:
        cy = (float(getattr(span, "y0", 0.0)) + float(getattr(span, "y1", 0.0))) / 2.0
        if not rows:
            rows.append([span])
            continue
        last = rows[-1]
        last_center = statistics.mean(
            (float(getattr(s, "y0", 0.0)) + float(getattr(s, "y1", 0.0))) / 2.0 for s in last
        )
        if abs(cy - last_center) <= line_threshold:
            last.append(span)
            continue
        rows.append([span])

    for row in rows:
        row.sort(
            key=lambda s: (
                float(getattr(s, "x0", 0.0)),
                float(getattr(s, "x1", 0.0)),
                str(getattr(s, "text", "")),
            )
        )
    return rows


def _split_row_into_segments_for_toc(row: list[Span], *, page_width: float) -> list[list[Span]]:
    if not row:
        return []
    ordered = sorted(
        row,
        key=lambda s: (
            float(getattr(s, "x0", 0.0)),
            float(getattr(s, "x1", 0.0)),
            str(getattr(s, "text", "")),
        ),
    )
    heights = [
        max(0.0, float(getattr(s, "y1", 0.0)) - float(getattr(s, "y0", 0.0)))
        for s in ordered
    ]
    split_gap = max(8.0, _median_positive(heights, default=10.0) * 1.4, page_width * 0.01)
    segments: list[list[Span]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        gap = float(getattr(cur, "x0", 0.0)) - float(getattr(prev, "x1", 0.0))
        if gap >= split_gap:
            segments.append([cur])
            continue
        segments[-1].append(cur)
    return segments


def _toc_keyword_score(normalized: str) -> int:
    return sum(1 for keyword in _TOC_NOISE_KEYWORDS if keyword in normalized)


def _count_toc_page_tokens(normalized: str) -> int:
    count = len(_PAGE_TOKEN_RE.findall(normalized))
    for run in _COMPACT_PAGE_RUN_RE.findall(normalized):
        count += max(2, len(run) // 2)
    return count


def _looks_like_toc_noise_segment_strict(text: str) -> bool:
    normalized = _normalize_margin_text(text)
    if not normalized:
        return False
    if len(normalized) > 220:
        return False

    page_token_count = _count_toc_page_tokens(normalized)
    compact_run_count = len(_COMPACT_PAGE_RUN_RE.findall(normalized))
    keyword_score = _toc_keyword_score(normalized)
    has_contents_or_section = ("contents" in normalized) or ("section" in normalized)
    has_role_word = bool(_TOC_ROLE_RE.search(normalized))
    has_toc_url = (
        "https://www.mazda.com/ja/" in normalized
        and ("investors" in normalized or "sustainability" in normalized)
    )

    if compact_run_count >= 1 and (keyword_score >= 1 or has_role_word):
        return True
    if compact_run_count >= 1 and has_contents_or_section:
        return True
    if has_contents_or_section and (page_token_count >= 1 or keyword_score >= 1):
        return True
    if page_token_count >= 1 and has_role_word and len(normalized) <= 180:
        return True
    if page_token_count >= 1 and keyword_score >= 2 and len(normalized) <= 160:
        return True
    if page_token_count >= 2 and keyword_score >= 1 and len(normalized) <= 160:
        return True
    if page_token_count >= 3 and len(normalized) <= 110:
        return True
    if has_toc_url and len(normalized) <= 140:
        return True
    return False


def _looks_like_toc_noise_segment_aggressive(text: str) -> bool:
    normalized = _normalize_margin_text(text)
    if not normalized:
        return False
    if len(normalized) > 180:
        return False

    page_token_count = _count_toc_page_tokens(normalized)
    compact_run_count = len(_COMPACT_PAGE_RUN_RE.findall(normalized))
    keyword_score = _toc_keyword_score(normalized)
    has_role_word = bool(_TOC_ROLE_RE.search(normalized))

    if compact_run_count >= 1 and (keyword_score >= 1 or has_role_word):
        return True
    if compact_run_count >= 1 and page_token_count >= 2 and len(normalized) <= 120:
        return True
    if page_token_count >= 2 and (keyword_score >= 1 or has_role_word) and len(normalized) <= 170:
        return True
    if page_token_count >= 1 and keyword_score >= 2 and len(normalized) <= 170:
        return True
    return False


def _looks_like_toc_noise_segment_left_index(text: str) -> bool:
    normalized = _normalize_margin_text(text)
    if not normalized:
        return False
    if len(normalized) > 120:
        return False

    page_token_count = _count_toc_page_tokens(normalized)
    keyword_score = _toc_keyword_score(normalized)
    has_role_word = bool(_TOC_ROLE_RE.search(normalized))
    starts_with_page = bool(_LEADING_PAGE_TOKEN_RE.match(normalized)) or bool(
        _COMPACT_PAGE_RUN_RE.match(normalized)
    )

    if starts_with_page and len(normalized) <= 90:
        return True
    if starts_with_page and keyword_score >= 1:
        return True
    if page_token_count >= 1 and keyword_score >= 1 and len(normalized) <= 90:
        return True
    if has_role_word and len(normalized) <= 80:
        return True
    if keyword_score >= 1 and len(normalized) <= 28:
        return True
    return False


def _looks_like_toc_noise_segment_loose(text: str) -> bool:
    normalized = _normalize_margin_text(text)
    if not normalized:
        return False
    if len(normalized) > 80:
        return False

    if _looks_like_toc_noise_segment_aggressive(text):
        return True
    if any(hint in normalized for hint in _TOC_TITLE_HINTS):
        return True
    if normalized.startswith("section"):
        return True
    if "investors/" in normalized or "sustainability/" in normalized:
        return True
    return False


def _remove_toc_noise_segments(
    spans_by_page: list[list[Span]],
    page_sizes: list[tuple[float, float]],
) -> list[list[Span]]:
    if len(spans_by_page) != len(page_sizes):
        return spans_by_page

    total_pages = len(spans_by_page)
    filtered_pages: list[list[Span]] = []
    for spans, (page_w, page_h) in zip(spans_by_page, page_sizes, strict=True):
        if page_w <= 0 or page_h <= 0 or not spans:
            filtered_pages.append(spans)
            continue

        rows = _cluster_page_spans_to_rows(spans)
        segment_meta: list[tuple[tuple[int, ...], str, str, float, float, bool]] = []
        strict_count = 0
        toc_anchor_count = 0
        left_index_count = 0

        for row in rows:
            segments = _split_row_into_segments_for_toc(row, page_width=page_w)
            for segment in segments:
                if not segment:
                    continue
                text = _join_row_text_for_margin(segment)
                if not text:
                    continue
                y0 = min(float(getattr(s, "y0", 0.0)) for s in segment)
                x0 = min(float(getattr(s, "x0", 0.0)) for s in segment)
                x1 = max(float(getattr(s, "x1", 0.0)) for s in segment)
                top_ratio = y0 / page_h
                x_center_ratio = ((x0 + x1) * 0.5) / page_w
                strict = _looks_like_toc_noise_segment_strict(text)
                normalized = _normalize_margin_text(text)
                if strict:
                    strict_count += 1
                if "contents" in normalized or "section" in normalized:
                    toc_anchor_count += 1
                if x_center_ratio <= 0.40 and _looks_like_toc_noise_segment_left_index(text):
                    left_index_count += 1
                segment_meta.append(
                    (
                        tuple(id(s) for s in segment),
                        text,
                        normalized,
                        top_ratio,
                        x_center_ratio,
                        strict,
                    )
                )

        if strict_count < 3:
            filtered_pages.append(spans)
            continue

        toc_page_candidate = toc_anchor_count >= 2
        if not toc_page_candidate and total_pages == 1:
            toc_page_candidate = strict_count >= 3 and toc_anchor_count >= 1
        if not toc_page_candidate:
            filtered_pages.append(spans)
            continue

        remove_ids: set[int] = set()
        aggressive_mode = strict_count >= 6 or left_index_count >= 6
        for ids, text, _normalized, top_ratio, x_center_ratio, strict in segment_meta:
            if strict:
                remove_ids.update(ids)
                continue
            if top_ratio <= 0.65 and _looks_like_toc_noise_segment_loose(text):
                remove_ids.update(ids)
                continue
            if (
                aggressive_mode
                and top_ratio <= 0.90
                and x_center_ratio <= 0.40
                and _looks_like_toc_noise_segment_left_index(text)
            ):
                remove_ids.update(ids)
                continue
            if (
                aggressive_mode
                and top_ratio <= 0.78
                and x_center_ratio <= 0.34
                and len(_normalized) <= 40
            ):
                remove_ids.update(ids)
                continue
            if (
                aggressive_mode
                and top_ratio <= 0.82
                and x_center_ratio <= 0.58
                and _looks_like_toc_noise_segment_aggressive(text)
            ):
                remove_ids.update(ids)

        if not remove_ids:
            filtered_pages.append(spans)
            continue
        filtered_pages.append([s for s in spans if id(s) not in remove_ids])
    return filtered_pages


def _remove_repeated_margin_rows(
    spans_by_page: list[list[Span]],
    page_sizes: list[tuple[float, float]],
) -> list[list[Span]]:
    total_pages = len(spans_by_page)
    if total_pages < 3:
        return spans_by_page
    if len(page_sizes) != total_pages:
        return spans_by_page

    header_exact_counts: dict[str, int] = {}
    footer_exact_counts: dict[str, int] = {}
    header_tpl_counts: dict[str, int] = {}
    footer_tpl_counts: dict[str, int] = {}
    row_meta: list[list[tuple[str, str, str, tuple[int, ...]]]] = []

    for spans, (_page_w, page_h) in zip(spans_by_page, page_sizes, strict=True):
        rows = _cluster_page_spans_to_rows(spans)
        page_rows: list[tuple[str, str, str, tuple[int, ...]]] = []
        for row in rows:
            if not row or page_h <= 0:
                continue
            y0 = min(float(getattr(s, "y0", 0.0)) for s in row)
            y1 = max(float(getattr(s, "y1", 0.0)) for s in row)

            text = _join_row_text_for_margin(row)
            if not text:
                continue
            top_ratio = y0 / page_h
            bottom_ratio = (page_h - y1) / page_h

            zone: str | None = None
            if top_ratio <= 0.10:
                zone = "header"
            elif bottom_ratio <= 0.10:
                zone = "footer"
            elif top_ratio <= 0.40 and _looks_like_nav_header(text):
                # Repeated TOC/navigation rows often appear below the strict header band.
                zone = "header"
            if zone is None:
                continue

            exact = _normalize_margin_text(text)
            tpl = _template_margin_text(text)
            ids = tuple(id(s) for s in row)
            page_rows.append((zone, exact, tpl, ids))

            if zone == "header":
                header_exact_counts[exact] = header_exact_counts.get(exact, 0) + 1
                header_tpl_counts[tpl] = header_tpl_counts.get(tpl, 0) + 1
            else:
                footer_exact_counts[exact] = footer_exact_counts.get(exact, 0) + 1
                footer_tpl_counts[tpl] = footer_tpl_counts.get(tpl, 0) + 1
        row_meta.append(page_rows)

    min_header_exact_repeat = max(3, int(math.ceil(total_pages * 0.15)))
    min_header_template_repeat = max(3, int(math.ceil(total_pages * 0.20)))
    min_footer_exact_repeat = max(3, int(math.ceil(total_pages * 0.30)))
    min_footer_template_repeat = max(3, int(math.ceil(total_pages * 0.45)))

    remove_header_exact = {
        k
        for k, v in header_exact_counts.items()
        if v >= min_header_exact_repeat or (v >= 3 and _looks_like_nav_header(k))
    }
    remove_footer_exact = {
        k
        for k, v in footer_exact_counts.items()
        if v >= min_footer_exact_repeat
    }
    remove_header_tpl = {
        k
        for k, v in header_tpl_counts.items()
        if v >= min_header_template_repeat or (v >= 3 and _looks_like_nav_header(k))
    }
    remove_footer_tpl = {
        k
        for k, v in footer_tpl_counts.items()
        if v >= min_footer_template_repeat
    }

    filtered: list[list[Span]] = []
    for spans, page_rows in zip(spans_by_page, row_meta, strict=True):
        remove_ids: set[int] = set()
        for zone, exact, tpl, ids in page_rows:
            if zone == "header":
                if exact in remove_header_exact or tpl in remove_header_tpl:
                    remove_ids.update(ids)
            else:
                if exact in remove_footer_exact or tpl in remove_footer_tpl:
                    remove_ids.update(ids)
        if not remove_ids:
            filtered.append(spans)
            continue
        filtered.append([s for s in spans if id(s) not in remove_ids])
    return filtered


def _lighton_lines_to_page_spans(
    *,
    lines: list[dict[str, object]],
    image_width: float,
    image_height: float,
    page_width: float,
    page_height: float,
) -> list[Span]:
    from ragprep.pdf_text import normalize_extracted_text

    if image_width <= 0 or image_height <= 0:
        return []
    if page_width <= 0 or page_height <= 0:
        return []

    sx = page_width / image_width
    sy = page_height / image_height
    spans: list[Span] = []
    for raw in lines:
        bbox_obj = raw.get("bbox")
        if not isinstance(bbox_obj, (list, tuple)) or len(bbox_obj) != 4:
            continue
        try:
            x0 = float(bbox_obj[0]) * sx
            y0 = float(bbox_obj[1]) * sy
            x1 = float(bbox_obj[2]) * sx
            y1 = float(bbox_obj[3]) * sy
        except Exception:  # noqa: BLE001
            continue
        if not (x0 < x1 and y0 < y1):
            continue
        text = normalize_extracted_text(str(raw.get("text") or "")).replace("\n", " ").strip()
        text = " ".join(text.split())
        if not text:
            continue
        spans.append(
            Span(
                x0=max(0.0, min(page_width, x0)),
                y0=max(0.0, min(page_height, y0)),
                x1=max(0.0, min(page_width, x1)),
                y1=max(0.0, min(page_height, y1)),
                text=text,
                size=max(1.0, y1 - y0),
                flags=None,
                font=None,
            )
        )
    spans.sort(key=lambda s: (s.y0, s.x0, s.y1, s.x1, s.text))
    return spans


def _collect_pymupdf_words_for_bbox(
    words: list[Word],
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> list[Word]:
    line_h = max(1.0, y1 - y0)
    margin_x = max(2.0, line_h * 0.35)
    margin_y = max(1.5, line_h * 0.35)
    bx0 = x0 - margin_x
    bx1 = x1 + margin_x
    by0 = y0 - margin_y
    by1 = y1 + margin_y

    hits: list[Word] = []
    for word in words:
        wx0 = float(getattr(word, "x0", 0.0))
        wy0 = float(getattr(word, "y0", 0.0))
        wx1 = float(getattr(word, "x1", 0.0))
        wy1 = float(getattr(word, "y1", 0.0))
        cx = (wx0 + wx1) / 2.0
        cy = (wy0 + wy1) / 2.0
        if bx0 <= cx <= bx1 and by0 <= cy <= by1:
            hits.append(word)

    hits.sort(
        key=lambda w: (
            int(getattr(w, "line_no", 0)),
            float(getattr(w, "y0", 0.0)),
            float(getattr(w, "x0", 0.0)),
            int(getattr(w, "word_no", 0)),
        )
    )
    return hits


def _join_pymupdf_words_for_line(words: list[Word]) -> str:
    if not words:
        return ""
    out: list[str] = []
    prev = None
    for word in words:
        text = str(getattr(word, "text", "")).strip()
        if not text:
            continue
        if prev is not None:
            prev_line = int(getattr(prev, "line_no", -1))
            cur_line = int(getattr(word, "line_no", -1))
            if prev_line != cur_line:
                out.append("\n")
            else:
                out.append(" ")
        out.append(text)
        prev = word
    return "".join(out).strip()


def _correct_line_spans_with_pymupdf_words(
    spans: list[Span],
    words: list[Word],
    *,
    policy: str = "aggressive",
) -> list[Span]:
    from ragprep.text_merge import merge_ocr_with_pymupdf

    corrected: list[Span] = []
    for span in spans:
        span_text = str(getattr(span, "text", ""))
        if not span_text:
            continue
        matches = _collect_pymupdf_words_for_bbox(
            words,
            x0=float(getattr(span, "x0", 0.0)),
            y0=float(getattr(span, "y0", 0.0)),
            x1=float(getattr(span, "x1", 0.0)),
            y1=float(getattr(span, "y1", 0.0)),
        )
        pym_line = _join_pymupdf_words_for_line(matches)
        merged = span_text
        if pym_line:
            merged, _ = merge_ocr_with_pymupdf(span_text, pym_line, policy=policy)
            if not merged:
                merged = span_text
        corrected.append(
            Span(
                x0=float(getattr(span, "x0", 0.0)),
                y0=float(getattr(span, "y0", 0.0)),
                x1=float(getattr(span, "x1", 0.0)),
                y1=float(getattr(span, "y1", 0.0)),
                text=str(merged).strip(),
                size=float(getattr(span, "size", 0.0)) if getattr(span, "size", None) else None,
                flags=getattr(span, "flags", None),
                font=getattr(span, "font", None),
            )
        )
    corrected.sort(key=lambda s: (s.y0, s.x0, s.y1, s.x1, s.text))
    return corrected


def _render_plain_text_from_spans(spans: list[Span]) -> str:
    if not spans:
        return ""
    rows = _cluster_page_spans_to_rows(spans)
    parts: list[str] = []
    for row in rows:
        text = _join_row_text_for_margin(row)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _pdf_to_markdown_lighton_ocr(
    pdf_bytes: bytes,
    *,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_page: PageCallback | None = None,
) -> str:
    from ragprep.ocr import lighton_ocr
    from ragprep.pdf_render import iter_pdf_images
    from ragprep.pdf_text import extract_pymupdf_page_sizes, extract_pymupdf_page_words

    try:
        page_sizes = extract_pymupdf_page_sizes(pdf_bytes)
        words_by_page = extract_pymupdf_page_words(pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to extract PyMuPDF page references for LightOn OCR.") from exc

    try:
        total_pages, images = iter_pdf_images(
            pdf_bytes,
            dpi=settings.render_dpi,
            max_edge=settings.render_max_edge,
            max_pages=settings.max_pages,
            max_bytes=settings.max_upload_bytes,
        )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to render PDF for LightOn OCR.") from exc

    if len(page_sizes) != int(total_pages) or len(words_by_page) != int(total_pages):
        raise RuntimeError("Page count mismatch between render and PyMuPDF references.")

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
    for page_index, image in enumerate(images, start=1):
        image_width = float(getattr(image, "width", 0.0))
        image_height = float(getattr(image, "height", 0.0))
        try:
            encoded = _image_to_png_base64(image)
        finally:
            try:
                image.close()
            except Exception:  # noqa: BLE001
                pass

        try:
            lighton_raw = lighton_ocr.analyze_ocr_layout_image_base64(encoded, settings=settings)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LightOn OCR failed on page {page_index}: {exc}") from exc

        raw_lines = lighton_raw.get("lines")
        if not isinstance(raw_lines, list):
            raise RuntimeError(f"LightOn OCR returned invalid lines on page {page_index}.")

        page_w, page_h = page_sizes[page_index - 1]
        line_spans = _lighton_lines_to_page_spans(
            lines=[cast(dict[str, object], r) for r in raw_lines if isinstance(r, dict)],
            image_width=image_width,
            image_height=image_height,
            page_width=page_w,
            page_height=page_h,
        )
        corrected = _correct_line_spans_with_pymupdf_words(
            line_spans,
            words_by_page[page_index - 1],
            policy="aggressive",
        )
        normalized = _render_plain_text_from_spans(corrected)
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

    backend = (settings.ocr_backend or settings.pdf_backend or "").strip().lower()
    if backend == "lighton-ocr":
        markdown = _pdf_to_markdown_lighton_ocr(
            pdf_bytes,
            settings=settings,
            on_progress=on_progress,
            on_page=on_page,
        )
    elif backend == "glm-ocr":
        markdown = _pdf_to_markdown_glm_ocr(
            pdf_bytes,
            settings=settings,
            on_progress=on_progress,
            on_page=on_page,
        )
    else:
        raise RuntimeError(f"Unsupported OCR backend: {backend!r}")

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

    pymupdf_spans_by_page = extract_pymupdf_page_spans(pdf_bytes)
    pymupdf_words_by_page = extract_pymupdf_page_words(pdf_bytes)
    page_sizes = extract_pymupdf_page_sizes(pdf_bytes)
    active_spans_by_page = pymupdf_spans_by_page
    total_pages = int(len(active_spans_by_page))

    if len(page_sizes) != total_pages or len(pymupdf_words_by_page) != total_pages:
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
            spans=active_spans_by_page[page_number - 1],
            page_width=page_w,
            page_height=page_h,
            layout_elements=layout_elements,
        )
        page = Page(page_number=page_number, blocks=blocks)
        return page, render_page_html(page)

    if layout_mode == "local-fast":
        from ragprep.layout.fast_layout import infer_fast_layout_elements

        active_spans_by_page = _remove_repeated_margin_rows(active_spans_by_page, page_sizes)
        active_spans_by_page = _remove_toc_noise_segments(active_spans_by_page, page_sizes)

        for page_number, spans in enumerate(active_spans_by_page, start=1):
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
    elif layout_mode == "lighton":
        from ragprep.ocr import lighton_ocr
        from ragprep.pdf_render import iter_pdf_images

        rendered_total_pages, images = iter_pdf_images(
            pdf_bytes,
            dpi=settings.layout_render_dpi,
            max_edge=settings.layout_render_max_edge,
            max_pages=settings.max_pages,
            max_bytes=settings.max_upload_bytes,
        )
        if int(rendered_total_pages) != total_pages:
            raise RuntimeError("Page count mismatch between render and text extraction.")

        active_spans_by_page = [[] for _ in range(total_pages)]

        for page_number, image in enumerate(images, start=1):
            image_width = float(getattr(image, "width", 0))
            image_height = float(getattr(image, "height", 0))
            if image_width <= 0 or image_height <= 0:
                raise RuntimeError("Failed to read rendered image size for LightOn normalization.")

            try:
                encoded = _image_to_png_base64(image)
            finally:
                try:
                    image.close()
                except Exception:  # noqa: BLE001
                    pass

            try:
                lighton_raw = lighton_ocr.analyze_ocr_layout_image_base64(
                    encoded,
                    settings=settings,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"LightOn OCR failed on page {page_number}: {exc}") from exc

            raw_lines = lighton_raw.get("lines")
            if not isinstance(raw_lines, list):
                raise RuntimeError(f"LightOn OCR returned invalid lines on page {page_number}.")

            page_w, page_h = page_sizes[page_number - 1]
            line_spans = _lighton_lines_to_page_spans(
                lines=[cast(dict[str, object], r) for r in raw_lines if isinstance(r, dict)],
                image_width=image_width,
                image_height=image_height,
                page_width=page_w,
                page_height=page_h,
            )
            active_spans_by_page[page_number - 1] = _correct_line_spans_with_pymupdf_words(
                line_spans,
                pymupdf_words_by_page[page_number - 1],
                policy="aggressive",
            )

            page, section_html = _build_page_from_layout(
                page_number=page_number,
                image_width=image_width,
                image_height=image_height,
                layout_raw=lighton_raw,
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


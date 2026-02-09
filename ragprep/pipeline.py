from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import cast

from ragprep.config import Settings, get_settings
from ragprep.pdf_text import Span


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
    if backend == "glm-ocr":
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
    - For `local-paddle`, run image-based layout analysis.
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
    )
    from ragprep.structure_ir import Document, Page, build_page_blocks, layout_element_from_raw

    pymupdf_spans_by_page = extract_pymupdf_page_spans(pdf_bytes)
    page_sizes = extract_pymupdf_page_sizes(pdf_bytes)
    active_spans_by_page = pymupdf_spans_by_page
    total_pages = int(len(active_spans_by_page))

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
    else:
        from ragprep.layout.glm_doclayout import analyze_layout_image_base64
        from ragprep.pdf_render import iter_pdf_images, render_pdf_page_image

        if layout_mode != "local-paddle":
            raise RuntimeError(f"Unsupported layout mode: {layout_mode!r}")

        adaptive_enabled = bool(settings.layout_render_auto)
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
            else:
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


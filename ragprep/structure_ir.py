from __future__ import annotations

import statistics
from dataclasses import dataclass

from ragprep.pdf_text import Span
from ragprep.table_grid import build_table_grid


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class LayoutElement:
    page_index: int
    bbox: BBox
    label: str
    score: float | None = None


@dataclass(frozen=True)
class Heading:
    level: int
    text: str


@dataclass(frozen=True)
class Paragraph:
    text: str


@dataclass(frozen=True)
class Table:
    text: str
    grid: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True)
class Figure:
    alt: str


@dataclass(frozen=True)
class Unknown:
    text: str


Block = Heading | Paragraph | Table | Figure | Unknown


@dataclass(frozen=True)
class Page:
    page_number: int
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Document:
    pages: tuple[Page, ...]


def normalize_bbox(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    width: float,
    height: float,
) -> BBox:
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be > 0")
    return BBox(
        x0=max(0.0, min(1.0, x0 / width)),
        y0=max(0.0, min(1.0, y0 / height)),
        x1=max(0.0, min(1.0, x1 / width)),
        y1=max(0.0, min(1.0, y1 / height)),
    )


def layout_element_from_raw(
    raw: dict[str, object],
    *,
    page_index: int,
    image_width: float,
    image_height: float,
) -> LayoutElement:
    bbox_obj = raw.get("bbox")
    if not isinstance(bbox_obj, tuple) or len(bbox_obj) != 4:
        raise ValueError("raw.bbox must be a tuple[4] from glm_doclayout")
    x0, y0, x1, y1 = (
        float(bbox_obj[0]),
        float(bbox_obj[1]),
        float(bbox_obj[2]),
        float(bbox_obj[3]),
    )
    label_obj = raw.get("label")
    if not isinstance(label_obj, str) or not label_obj.strip():
        raise ValueError("raw.label must be a non-empty string")
    score_obj = raw.get("score")
    score = float(score_obj) if isinstance(score_obj, (int, float)) else None
    return LayoutElement(
        page_index=page_index,
        bbox=normalize_bbox(x0, y0, x1, y1, width=image_width, height=image_height),
        label=label_obj.strip(),
        score=score,
    )


def build_page_blocks(
    *,
    spans: list[Span],
    page_width: float,
    page_height: float,
    layout_elements: list[LayoutElement],
) -> tuple[Block, ...]:
    """
    Build structured blocks for a single page.

    Assumptions:
    - `layout_elements` are in normalized coordinates [0..1] in page space.
    - `spans` bboxes are in page coordinates; we normalize span centers for assignment.
    """

    if page_width <= 0 or page_height <= 0:
        raise ValueError("page_width/page_height must be > 0")

    elements_sorted = _order_layout_elements_for_reading(layout_elements)

    assignments: dict[int, list[Span]] = {i: [] for i in range(len(elements_sorted))}
    unassigned: list[Span] = []

    # Prefer smallest region that contains the span center.
    for span in spans:
        cx = ((span.x0 + span.x1) / 2.0) / page_width
        cy = ((span.y0 + span.y1) / 2.0) / page_height

        best_index: int | None = None
        best_area = 0.0
        for i, elt in enumerate(elements_sorted):
            if not elt.bbox.contains_point(cx, cy):
                continue
            area = elt.bbox.area
            if best_index is None or area < best_area:
                best_index = i
                best_area = area

        if best_index is None:
            unassigned.append(span)
        else:
            assignments[best_index].append(span)

    span_sizes = [
        float(s.size)
        for s in spans
        if isinstance(s.size, (int, float)) and s.size and s.size > 0
    ]
    page_median_size = _median_or_default(span_sizes, default=0.0) if span_sizes else 0.0

    blocks: list[Block] = []
    for i, elt in enumerate(elements_sorted):
        collected = assignments[i]
        if not collected:
            continue
        text = _join_spans_text(collected)
        if not text:
            continue
        blocks.append(
            _block_from_label(
                elt.label,
                text,
                spans=collected,
                page_median_span_size=page_median_size,
            )
        )

    if unassigned:
        text = _join_spans_text(unassigned)
        if text:
            blocks.append(Paragraph(text=text))

    return tuple(blocks)


def build_document(
    *,
    spans_by_page: list[list[Span]],
    page_sizes: list[tuple[float, float]],
    layout_by_page: list[list[LayoutElement]],
) -> Document:
    if len(spans_by_page) != len(page_sizes) or len(spans_by_page) != len(layout_by_page):
        raise ValueError("spans_by_page, page_sizes, layout_by_page length mismatch")

    pages: list[Page] = []
    for page_index, (spans, (w, h), layout_elements) in enumerate(
        zip(spans_by_page, page_sizes, layout_by_page, strict=True)
    ):
        blocks = build_page_blocks(
            spans=spans,
            page_width=w,
            page_height=h,
            layout_elements=layout_elements,
        )
        pages.append(Page(page_number=page_index + 1, blocks=blocks))

    return Document(pages=tuple(pages))


def _order_layout_elements_for_reading(elements: list[LayoutElement]) -> list[LayoutElement]:
    """
    Order layout regions for a human reading order.

    Heuristic:
    - If there are clear left/right column regions, output all left-column blocks top→bottom,
      then all right-column blocks top→bottom.
    - Full-width regions above the columns (e.g., title) come first; full-width below come last.
    - Otherwise fall back to (y0, x0) ordering.
    """

    if not elements:
        return []

    fallback = sorted(elements, key=lambda e: (e.bbox.y0, e.bbox.x0, e.bbox.y1, e.bbox.x1, e.label))

    full_width: list[LayoutElement] = []
    left: list[LayoutElement] = []
    right: list[LayoutElement] = []
    middle: list[LayoutElement] = []

    for e in fallback:
        width = max(0.0, e.bbox.x1 - e.bbox.x0)
        if e.bbox.x0 <= 0.15 and e.bbox.x1 >= 0.85 and width >= 0.70:
            full_width.append(e)
        elif e.bbox.x1 <= 0.52:
            left.append(e)
        elif e.bbox.x0 >= 0.48:
            right.append(e)
        else:
            middle.append(e)

    # Only treat as two-column when both sides have multiple regions.
    if len(left) < 2 or len(right) < 2:
        return fallback

    columns_start_y0 = min((e.bbox.y0 for e in left + right), default=0.0)
    columns_end_y1 = max((e.bbox.y1 for e in left + right), default=1.0)

    top_full = [e for e in full_width if e.bbox.y1 <= columns_start_y0 + 1e-6]
    bottom_full = [
        e
        for e in full_width
        if e not in top_full and e.bbox.y0 >= columns_end_y1 - 1e-6
    ]
    mid_full = [e for e in full_width if e not in top_full and e not in bottom_full]

    # Keep anything ambiguous in the fallback position by y0.
    ordered: list[LayoutElement] = []
    ordered.extend(sorted(top_full, key=lambda e: (e.bbox.y0, e.bbox.x0)))
    ordered.extend(sorted(mid_full, key=lambda e: (e.bbox.y0, e.bbox.x0)))
    ordered.extend(sorted(left, key=lambda e: (e.bbox.y0, e.bbox.x0)))
    ordered.extend(sorted(right, key=lambda e: (e.bbox.y0, e.bbox.x0)))
    ordered.extend(sorted(middle, key=lambda e: (e.bbox.y0, e.bbox.x0)))
    ordered.extend(sorted(bottom_full, key=lambda e: (e.bbox.y0, e.bbox.x0)))

    # Dedupe while preserving order.
    seen: set[int] = set()
    out: list[LayoutElement] = []
    for e in ordered:
        key = id(e)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out if len(out) == len(fallback) else fallback


def _join_spans_text(spans: list[Span]) -> str:
    if not spans:
        return ""
    ordered = sorted(spans, key=lambda s: (s.y0, s.x0, s.y1, s.x1, s.text))
    heights = [max(0.0, s.y1 - s.y0) for s in ordered]
    line_bin = max(2.0, _median_or_default(heights, default=10.0) * 0.75)

    lines: dict[int, list[Span]] = {}
    for s in ordered:
        cy = (s.y0 + s.y1) / 2.0
        key = int(round(cy / line_bin)) if line_bin > 0 else 0
        lines.setdefault(key, []).append(s)

    line_keys = sorted(lines.keys())

    line_items: list[tuple[float, float, str]] = []
    for key in line_keys:
        spans_in_line = lines[key]
        rendered = _join_spans_in_line(spans_in_line)
        if not rendered:
            continue
        y0 = min(s.y0 for s in spans_in_line)
        y1 = max(s.y1 for s in spans_in_line)
        line_items.append((float(y0), float(y1), rendered))

    if not line_items:
        return ""

    # Join wrapped lines with spaces; keep paragraph breaks when there is a large vertical gap.
    heights = [max(0.0, y1 - y0) for (y0, y1, _t) in line_items]
    median_h = _median_or_default(heights, default=10.0)
    paragraph_gap = max(6.0, median_h * 1.8)

    out: list[str] = [line_items[0][2]]
    for (_prev_y0, prev_y1, _prev_text), (y0, _y1, text) in zip(
        line_items, line_items[1:], strict=False
    ):
        gap = float(y0 - prev_y1)
        if gap >= paragraph_gap:
            out.append("\n")
            out.append(text)
            continue

        # Wrapped line: prefer space-join; handle simple hyphenation.
        if out and out[-1].endswith("-") and text and text[:1].isalpha():
            out[-1] = out[-1][:-1]
            out.append(text)
        else:
            out.append(" ")
            out.append(text)

    return "".join(out).strip()


def _median_or_default(values: list[float], *, default: float) -> float:
    cleaned = [v for v in values if v > 0 and v == v]  # filter non-positive and NaN
    if not cleaned:
        return float(default)
    try:
        return float(statistics.median(cleaned))
    except statistics.StatisticsError:
        return float(default)


def _join_spans_in_line(spans: list[Span]) -> str:
    if not spans:
        return ""
    ordered = sorted(spans, key=lambda s: (s.x0, s.x1, s.text))
    heights = [max(0.0, s.y1 - s.y0) for s in ordered]
    median_h = _median_or_default(heights, default=10.0)
    gap_threshold = max(1.0, median_h * 0.25)

    out: list[str] = []
    prev: Span | None = None
    for s in ordered:
        if not s.text:
            continue
        if prev is not None:
            gap = s.x0 - prev.x1
            if gap >= gap_threshold:
                out.append(" ")
        out.append(s.text)
        prev = s
    return "".join(out).strip()


def _heading_level_from_spans(*, spans: list[Span], page_median_span_size: float) -> int:
    if page_median_span_size <= 0:
        return 1
    sizes = [
        float(s.size)
        for s in spans
        if isinstance(s.size, (int, float)) and s.size and s.size > 0
    ]
    if not sizes:
        return 1
    median = _median_or_default(sizes, default=page_median_span_size)
    ratio = median / max(1e-6, float(page_median_span_size))
    if ratio >= 1.60:
        return 1
    if ratio >= 1.35:
        return 2
    if ratio >= 1.20:
        return 3
    return 4


def _block_from_label(
    label: str,
    text: str,
    *,
    spans: list[Span],
    page_median_span_size: float,
) -> Block:
    normalized = (label or "").strip().lower()

    heading_labels = {"title", "heading", "header", "paragraph_title", "section_title"}
    if normalized in heading_labels:
        level = _heading_level_from_spans(spans=spans, page_median_span_size=page_median_span_size)
        if normalized == "title":
            level = 1
        return Heading(level=level, text=text)

    if normalized in {"text", "paragraph"}:
        # Best-effort: promote large-font short regions to headings when the layout label is
        # ambiguous.
        if (
            page_median_span_size > 0
            and _heading_level_from_spans(
                spans=spans,
                page_median_span_size=page_median_span_size,
            )
            <= 2
            and len(text) <= 80
        ):
            level = _heading_level_from_spans(
                spans=spans,
                page_median_span_size=page_median_span_size,
            )
            return Heading(level=max(2, int(level)), text=text)
        return Paragraph(text=text)

    if normalized == "table":
        return _table_from_spans(text=text, spans=spans)
    if normalized in {"figure", "image"}:
        return Figure(alt=text)
    return Unknown(text=text)


def _table_from_spans(*, text: str, spans: list[Span]) -> Table:
    # Best-effort: infer a grid from span positions. If uncertain, fall back to plain text.
    from ragprep.pdf_text import Word

    words = [
        Word(
            x0=float(s.x0),
            y0=float(s.y0),
            x1=float(s.x1),
            y1=float(s.y1),
            text=str(s.text),
            block_no=0,
            line_no=0,
            word_no=0,
        )
        for s in spans
        if s.text
    ]

    best_rows: tuple[tuple[str, ...], ...] | None = None
    best_empty_ratio: float | None = None
    best_conf = -1.0
    best_k = 0
    for k in range(2, 7):
        result = build_table_grid(words, column_count=k)
        if not result.ok or result.grid is None:
            continue
        rows = result.grid.rows
        total = max(1, len(rows) * k)
        empty = sum(1 for r in rows for c in r[:k] if not str(c).strip())
        empty_ratio = empty / total
        conf = float(result.confidence)

        if best_empty_ratio is None or empty_ratio < best_empty_ratio - 1e-9:
            best_empty_ratio = empty_ratio
            best_conf = conf
            best_k = k
            best_rows = rows
            continue

        if best_empty_ratio is not None and abs(empty_ratio - best_empty_ratio) <= 1e-9:
            if k > best_k or (k == best_k and conf > best_conf):
                best_empty_ratio = empty_ratio
                best_conf = conf
                best_k = k
                best_rows = rows

    return Table(text=text, grid=best_rows)

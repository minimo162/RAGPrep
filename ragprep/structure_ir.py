from __future__ import annotations

import statistics
from dataclasses import dataclass

from ragprep.pdf_text import Span, Word
from ragprep.table_grid import TableCell, build_table_grid


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
    order: int | None = None


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
    cells: tuple[TableCell, ...] | None = None


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
    if not isinstance(bbox_obj, (list, tuple)) or len(bbox_obj) != 4:
        raise ValueError("raw.bbox must be a list[4] from layout backend")
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
    order_obj = raw.get("order")
    order: int | None = None
    if isinstance(order_obj, int) and not isinstance(order_obj, bool):
        order = int(order_obj)
    elif isinstance(order_obj, float) and order_obj.is_integer():
        order = int(order_obj)
    if order is not None and order < 0:
        raise ValueError("raw.order must be >= 0")
    return LayoutElement(
        page_index=page_index,
        bbox=normalize_bbox(x0, y0, x1, y1, width=image_width, height=image_height),
        label=label_obj.strip(),
        score=score,
        order=order,
    )


def build_page_blocks(
    *,
    spans: list[Span],
    page_words: list[Word] | None = None,
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

    elements_for_assignment = sorted(
        layout_elements,
        key=lambda e: (
            e.bbox.area,
            e.bbox.y0,
            e.bbox.x0,
            e.bbox.y1,
            e.bbox.x1,
            e.label,
        ),
    )
    element_to_assignment_index = {id(e): i for i, e in enumerate(elements_for_assignment)}
    ordered_for_reading = _order_layout_elements_for_reading(layout_elements)

    assignments: dict[int, list[Span]] = {i: [] for i in range(len(elements_for_assignment))}
    unassigned: list[Span] = []

    # Prefer smallest region that contains the span center.
    for span in spans:
        cx = ((span.x0 + span.x1) / 2.0) / page_width
        cy = ((span.y0 + span.y1) / 2.0) / page_height

        best_index: int | None = None
        best_area = 0.0
        for i, elt in enumerate(elements_for_assignment):
            if not elt.bbox.contains_point(cx, cy):
                continue
            area = elt.bbox.area
            if best_index is None or area < best_area:
                best_index = i
                best_area = area

        if best_index is None:
            span_x0 = float(span.x0) / page_width
            span_y0 = float(span.y0) / page_height
            span_x1 = float(span.x1) / page_width
            span_y1 = float(span.y1) / page_height
            span_area = max(1e-6, max(0.0, span_x1 - span_x0) * max(0.0, span_y1 - span_y0))

            best_overlap_ratio = 0.0
            best_overlap_area = 0.0
            best_overlap_index: int | None = None
            best_overlap_target_area = 0.0
            for i, elt in enumerate(elements_for_assignment):
                overlap_w = max(
                    0.0,
                    min(span_x1, float(elt.bbox.x1)) - max(span_x0, float(elt.bbox.x0)),
                )
                overlap_h = max(
                    0.0,
                    min(span_y1, float(elt.bbox.y1)) - max(span_y0, float(elt.bbox.y0)),
                )
                overlap_area = overlap_w * overlap_h
                if overlap_area <= 0:
                    continue
                overlap_ratio = overlap_area / span_area
                target_area = float(elt.bbox.area)
                if (
                    best_overlap_index is None
                    or overlap_ratio > best_overlap_ratio + 1e-9
                    or (
                        abs(overlap_ratio - best_overlap_ratio) <= 1e-9
                        and (
                            overlap_area > best_overlap_area + 1e-9
                            or (
                                abs(overlap_area - best_overlap_area) <= 1e-9
                                and target_area < best_overlap_target_area
                            )
                        )
                    )
                ):
                    best_overlap_ratio = overlap_ratio
                    best_overlap_area = overlap_area
                    best_overlap_index = i
                    best_overlap_target_area = target_area

            if best_overlap_index is not None and best_overlap_ratio >= 0.05:
                assignments[best_overlap_index].append(span)
            else:
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
    for elt in ordered_for_reading:
        assignment_index = element_to_assignment_index.get(id(elt))
        if assignment_index is None:
            continue
        collected = assignments[assignment_index]
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
                page_words=page_words,
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
    page_words_by_page: list[list[Word]] | None = None,
    page_sizes: list[tuple[float, float]],
    layout_by_page: list[list[LayoutElement]],
) -> Document:
    if len(spans_by_page) != len(page_sizes) or len(spans_by_page) != len(layout_by_page):
        raise ValueError("spans_by_page, page_sizes, layout_by_page length mismatch")
    if page_words_by_page is not None and len(page_words_by_page) != len(spans_by_page):
        raise ValueError("spans_by_page and page_words_by_page length mismatch")

    pages: list[Page] = []
    for page_index, (spans, (w, h), layout_elements) in enumerate(
        zip(spans_by_page, page_sizes, layout_by_page, strict=True)
    ):
        blocks = build_page_blocks(
            spans=spans,
            page_words=page_words_by_page[page_index] if page_words_by_page is not None else None,
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

    with_order = [e for e in elements if e.order is not None]
    without_order = [e for e in elements if e.order is None]
    if not with_order:
        return _xy_cut_order(elements, max_depth=10)

    ordered_explicit = sorted(
        with_order,
        key=lambda e: (int(e.order or 0), e.bbox.y0, e.bbox.x0, e.bbox.y1, e.bbox.x1, e.label),
    )
    ordered_rest = _xy_cut_order(without_order, max_depth=10) if without_order else []
    return ordered_explicit + ordered_rest


def _xy_cut_order(elements: list[LayoutElement], *, max_depth: int) -> list[LayoutElement]:
    fallback = sorted(
        elements,
        key=lambda e: (e.bbox.y0, e.bbox.x0, e.bbox.y1, e.bbox.x1, e.label),
    )
    return _xy_cut_order_inner(fallback, depth=0, max_depth=max_depth)


def _xy_cut_order_inner(
    elements: list[LayoutElement],
    *,
    depth: int,
    max_depth: int,
) -> list[LayoutElement]:
    if len(elements) <= 1:
        return list(elements)
    if depth >= max_depth:
        return _topo_order(elements)

    split_y = _best_gap_split(elements, axis="y")
    split_x = _best_gap_split(elements, axis="x")

    min_gap = 0.03
    has_y = split_y is not None and split_y.gap >= min_gap
    has_x = split_x is not None and split_x.gap >= min_gap

    if not has_y and not has_x:
        return _topo_order(elements)

    if has_y and not has_x:
        column_split = _try_column_major_split(elements)
        if column_split is not None:
            left, right = column_split
            left_ordered = _xy_cut_order_inner(left, depth=depth + 1, max_depth=max_depth)
            right_ordered = _xy_cut_order_inner(right, depth=depth + 1, max_depth=max_depth)
            return left_ordered + right_ordered

    chosen = split_y
    if has_x and (
        not has_y
        or (
            split_x is not None
            and split_y is not None
            and split_x.gap > split_y.gap
        )
    ):
        chosen = split_x

    if chosen is None:
        return _topo_order(elements)

    a, b = chosen.groups
    if not a or not b:
        return _topo_order(elements)

    first = _xy_cut_order_inner(a, depth=depth + 1, max_depth=max_depth)
    second = _xy_cut_order_inner(b, depth=depth + 1, max_depth=max_depth)
    return first + second


def _try_column_major_split(
    elements: list[LayoutElement],
) -> tuple[list[LayoutElement], list[LayoutElement]] | None:
    if len(elements) < 4:
        return None

    widths = [max(0.0, float(e.bbox.x1) - float(e.bbox.x0)) for e in elements]
    if any(width > 0.75 for width in widths):
        return None

    with_centers = sorted(
        [
            (
                (float(e.bbox.x0) + float(e.bbox.x1)) / 2.0,
                e,
            )
            for e in elements
        ],
        key=lambda item: (item[0], item[1].bbox.y0, item[1].bbox.x0),
    )

    split_at = len(with_centers) // 2
    left_pairs = with_centers[:split_at]
    right_pairs = with_centers[split_at:]
    if len(left_pairs) < 2 or len(right_pairs) < 2:
        return None

    left_centers = [c for c, _ in left_pairs]
    right_centers = [c for c, _ in right_pairs]
    left_center = float(statistics.median(left_centers))
    right_center = float(statistics.median(right_centers))
    center_gap = right_center - left_center
    width_median = _median_or_default(widths, default=0.0)
    if center_gap < max(0.15, width_median * 0.55):
        return None

    left = [e for _, e in left_pairs]
    right = [e for _, e in right_pairs]
    if max(left_centers) >= min(right_centers):
        return None

    return left, right


@dataclass(frozen=True)
class _GapSplit:
    axis: str
    gap: float
    groups: tuple[list[LayoutElement], list[LayoutElement]]


def _best_gap_split(elements: list[LayoutElement], *, axis: str) -> _GapSplit | None:
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if len(elements) <= 1:
        return None

    if axis == "x":
        ordered = sorted(
            elements,
            key=lambda e: (e.bbox.x0, e.bbox.x1, e.bbox.y0, e.bbox.y1, e.label),
        )
        starts = [e.bbox.x0 for e in ordered]
        ends = [e.bbox.x1 for e in ordered]
    else:
        ordered = sorted(
            elements,
            key=lambda e: (e.bbox.y0, e.bbox.y1, e.bbox.x0, e.bbox.x1, e.label),
        )
        starts = [e.bbox.y0 for e in ordered]
        ends = [e.bbox.y1 for e in ordered]

    best_gap = 0.0
    best_i: int | None = None
    running_end = float(ends[0])
    for i in range(len(ordered) - 1):
        running_end = max(running_end, float(ends[i]))
        gap = float(starts[i + 1]) - running_end
        if gap > best_gap:
            best_gap = gap
            best_i = i

    if best_i is None or best_gap <= 0:
        return None

    left = ordered[: best_i + 1]
    right = ordered[best_i + 1 :]
    return _GapSplit(axis=axis, gap=float(best_gap), groups=(list(left), list(right)))


def _topo_order(elements: list[LayoutElement]) -> list[LayoutElement]:
    if len(elements) <= 1:
        return list(elements)

    nodes = list(elements)

    def tie_key(e: LayoutElement) -> tuple[float, float, float, float, str]:
        return (e.bbox.y0, e.bbox.x0, e.bbox.y1, e.bbox.x1, e.label)

    n = len(nodes)
    edges: list[set[int]] = [set() for _ in range(n)]
    indeg = [0] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _precedes(nodes[i], nodes[j]):
                if j not in edges[i]:
                    edges[i].add(j)

    for i in range(n):
        for j in edges[i]:
            indeg[j] += 1

    ready = sorted([i for i in range(n) if indeg[i] == 0], key=lambda idx: tie_key(nodes[idx]))
    out: list[int] = []
    while ready:
        idx = ready.pop(0)
        out.append(idx)
        for j in sorted(edges[idx]):
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
        ready.sort(key=lambda k: tie_key(nodes[k]))

    if len(out) != n:
        return sorted(nodes, key=tie_key)

    return [nodes[i] for i in out]


def _precedes(a: LayoutElement, b: LayoutElement) -> bool:
    # Use overlap-aware precedence rules to create a partial order.
    x_overlap = max(0.0, min(a.bbox.x1, b.bbox.x1) - max(a.bbox.x0, b.bbox.x0))
    y_overlap = max(0.0, min(a.bbox.y1, b.bbox.y1) - max(a.bbox.y0, b.bbox.y0))
    a_w = max(1e-6, a.bbox.x1 - a.bbox.x0)
    b_w = max(1e-6, b.bbox.x1 - b.bbox.x0)
    a_h = max(1e-6, a.bbox.y1 - a.bbox.y0)
    b_h = max(1e-6, b.bbox.y1 - b.bbox.y0)

    x_overlap_ratio = x_overlap / min(a_w, b_w)
    y_overlap_ratio = y_overlap / min(a_h, b_h)

    eps = 0.005

    # Stacked blocks (same column): vertical precedence.
    if x_overlap_ratio >= 0.15 and a.bbox.y1 <= b.bbox.y0 + eps:
        return True

    # Side-by-side blocks (same row band): left-to-right precedence.
    if y_overlap_ratio >= 0.15 and a.bbox.x1 <= b.bbox.x0 + eps:
        return True

    return False


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
    page_words: list[Word] | None,
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
        return _table_from_spans(text=text, spans=spans, page_words=page_words)
    if normalized in {"figure", "image"}:
        return Figure(alt=text)
    return Unknown(text=text)


def _table_from_spans(
    *,
    text: str,
    spans: list[Span],
    page_words: list[Word] | None,
) -> Table:
    # Best-effort: infer a grid from span positions. If uncertain, fall back to plain text.
    words = page_words if page_words else [
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
    best_cells: tuple[TableCell, ...] | None = None
    best_collision_ratio: float | None = None
    best_empty_ratio: float | None = None
    best_conf = -1.0
    best_k = 0
    for k in _candidate_table_column_counts(words):
        result = build_table_grid(words, column_count=k)
        if not result.ok or result.grid is None:
            continue
        rows = result.grid.rows
        total = max(1, len(rows) * k)
        empty = sum(1 for r in rows for c in r[:k] if not str(c).strip())
        empty_ratio = empty / total
        conf = float(result.confidence)
        collision_ratio = float(result.grid.collision_count) / max(
            1.0,
            float(result.grid.group_count),
        )

        if best_collision_ratio is None or collision_ratio < best_collision_ratio - 1e-9:
            best_collision_ratio = collision_ratio
            best_empty_ratio = empty_ratio
            best_conf = conf
            best_k = k
            best_rows = rows
            best_cells = result.grid.cells
            continue

        if best_collision_ratio is not None and abs(collision_ratio - best_collision_ratio) <= 1e-9:
            if best_empty_ratio is None or empty_ratio < best_empty_ratio - 1e-9:
                best_collision_ratio = collision_ratio
                best_empty_ratio = empty_ratio
                best_conf = conf
                best_k = k
                best_rows = rows
                best_cells = result.grid.cells
                continue
            if best_empty_ratio is not None and abs(empty_ratio - best_empty_ratio) <= 1e-9:
                if conf > best_conf + 1e-9 or (
                    abs(conf - best_conf) <= 1e-9 and k > best_k
                ):
                    best_collision_ratio = collision_ratio
                    best_empty_ratio = empty_ratio
                    best_conf = conf
                    best_k = k
                    best_rows = rows
                    best_cells = result.grid.cells

    return Table(text=text, grid=best_rows, cells=best_cells)


def _candidate_table_column_counts(words: list[Word]) -> tuple[int, ...]:
    if not words:
        return (2,)

    heights = [max(0.0, float(w.y1) - float(w.y0)) for w in words]
    row_bin = max(4.0, _median_or_default(heights, default=10.0) * 0.75)
    row_counts: dict[int, int] = {}
    for w in words:
        y_center = (float(w.y0) + float(w.y1)) / 2.0
        key = int(round(y_center / row_bin)) if row_bin > 0 else 0
        row_counts[key] = row_counts.get(key, 0) + 1

    max_columns = max(2, min(6, max(row_counts.values(), default=2)))
    return tuple(range(max_columns, 1, -1))

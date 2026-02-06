from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from ragprep.pdf_text import Span, Word
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
class TableCell:
    row: int
    col: int
    text: str
    colspan: int = 1
    rowspan: int = 1


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

    # Assign spans to layout regions with graceful fallback for slightly off detector boxes.
    for span in spans:
        best_index = _match_span_to_layout_element(
            span,
            page_width=page_width,
            page_height=page_height,
            elements=elements_for_assignment,
        )
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
        table_words: list[Word] | None = None
        if (elt.label or "").strip().lower() == "table":
            table_words = _collect_words_for_layout_element(
                page_words=page_words,
                element_bbox=elt.bbox,
                page_width=page_width,
                page_height=page_height,
            )
        blocks.append(
            _block_from_label(
                elt.label,
                text,
                spans=collected,
                table_words=table_words,
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
    words_by_page: list[list[Word]] | None = None,
    page_sizes: list[tuple[float, float]],
    layout_by_page: list[list[LayoutElement]],
) -> Document:
    if len(spans_by_page) != len(page_sizes) or len(spans_by_page) != len(layout_by_page):
        raise ValueError("spans_by_page, page_sizes, layout_by_page length mismatch")
    if words_by_page is not None and len(spans_by_page) != len(words_by_page):
        raise ValueError("spans_by_page and words_by_page length mismatch")

    pages: list[Page] = []
    for page_index, (spans, (w, h), layout_elements) in enumerate(
        zip(spans_by_page, page_sizes, layout_by_page, strict=True)
    ):
        blocks = build_page_blocks(
            spans=spans,
            page_words=words_by_page[page_index] if words_by_page is not None else None,
            page_width=w,
            page_height=h,
            layout_elements=layout_elements,
        )
        pages.append(Page(page_number=page_index + 1, blocks=blocks))

    return Document(pages=tuple(pages))


def _match_span_to_layout_element(
    span: Span,
    *,
    page_width: float,
    page_height: float,
    elements: list[LayoutElement],
) -> int | None:
    if not elements:
        return None

    span_bbox = normalize_bbox(
        float(span.x0),
        float(span.y0),
        float(span.x1),
        float(span.y1),
        width=page_width,
        height=page_height,
    )
    cx = (span_bbox.x0 + span_bbox.x1) / 2.0
    cy = (span_bbox.y0 + span_bbox.y1) / 2.0

    center_match = _find_smallest_center_match(cx=cx, cy=cy, elements=elements)
    if center_match is not None:
        return center_match

    overlap_match = _find_best_overlap_match(span_bbox=span_bbox, elements=elements)
    if overlap_match is not None:
        return overlap_match

    nearest_index, nearest_distance = _nearest_layout_element(cx=cx, cy=cy, elements=elements)
    if nearest_index is None:
        return None
    if nearest_distance > 0.045:
        return None
    return nearest_index


def _find_smallest_center_match(
    *,
    cx: float,
    cy: float,
    elements: list[LayoutElement],
) -> int | None:
    best_index: int | None = None
    best_area = 0.0
    for i, elt in enumerate(elements):
        if not elt.bbox.contains_point(cx, cy):
            continue
        area = elt.bbox.area
        if best_index is None or area < best_area:
            best_index = i
            best_area = area
    return best_index


def _find_best_overlap_match(
    *,
    span_bbox: BBox,
    elements: list[LayoutElement],
) -> int | None:
    best_index: int | None = None
    best_overlap_ratio = 0.0
    best_area = 0.0
    span_area = max(1e-9, span_bbox.area)

    for i, elt in enumerate(elements):
        overlap = _bbox_intersection_area(span_bbox, elt.bbox)
        if overlap <= 0:
            continue
        overlap_ratio = overlap / span_area
        if (
            best_index is None
            or overlap_ratio > best_overlap_ratio + 1e-9
            or (
                abs(overlap_ratio - best_overlap_ratio) <= 1e-9
                and elt.bbox.area < best_area
            )
        ):
            best_index = i
            best_overlap_ratio = overlap_ratio
            best_area = elt.bbox.area

    if best_index is None:
        return None
    if best_overlap_ratio < 0.02:
        return None
    return best_index


def _nearest_layout_element(
    *,
    cx: float,
    cy: float,
    elements: list[LayoutElement],
) -> tuple[int | None, float]:
    if not elements:
        return None, float("inf")

    best_index: int | None = None
    best_distance = float("inf")
    for i, elt in enumerate(elements):
        distance = _point_to_bbox_distance(cx=cx, cy=cy, bbox=elt.bbox)
        if distance < best_distance:
            best_index = i
            best_distance = distance
    return best_index, best_distance


def _point_to_bbox_distance(*, cx: float, cy: float, bbox: BBox) -> float:
    if cx < bbox.x0:
        dx = bbox.x0 - cx
    elif cx > bbox.x1:
        dx = cx - bbox.x1
    else:
        dx = 0.0

    if cy < bbox.y0:
        dy = bbox.y0 - cy
    elif cy > bbox.y1:
        dy = cy - bbox.y1
    else:
        dy = 0.0

    return math.hypot(dx, dy)


def _bbox_intersection_area(a: BBox, b: BBox) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _collect_words_for_layout_element(
    *,
    page_words: list[Word] | None,
    element_bbox: BBox,
    page_width: float,
    page_height: float,
) -> list[Word] | None:
    if not page_words:
        return None

    by_center: list[Word] = []
    by_overlap: list[tuple[float, Word]] = []
    for word in page_words:
        norm = normalize_bbox(
            float(word.x0),
            float(word.y0),
            float(word.x1),
            float(word.y1),
            width=page_width,
            height=page_height,
        )
        cx = (norm.x0 + norm.x1) / 2.0
        cy = (norm.y0 + norm.y1) / 2.0
        if element_bbox.contains_point(cx, cy):
            by_center.append(word)
            continue

        overlap = _bbox_intersection_area(norm, element_bbox)
        if overlap <= 0:
            continue
        ratio = overlap / max(1e-9, norm.area)
        if ratio >= 0.02:
            by_overlap.append((ratio, word))

    if by_center:
        return sorted(by_center, key=lambda w: (w.y0, w.x0, w.y1, w.x1, w.word_no))
    if by_overlap:
        words = [
            w
            for (_r, w) in sorted(
                by_overlap,
                key=lambda item: (-item[0], item[1].y0, item[1].x0),
            )
        ]
        return sorted(words, key=lambda w: (w.y0, w.x0, w.y1, w.x1, w.word_no))
    return None


def _order_layout_elements_for_reading(elements: list[LayoutElement]) -> list[LayoutElement]:
    """
    Order layout regions for a human reading order.

    Heuristic:
    - If there are clear left/right column regions, output all left-column blocks top→bottom,
      then all right-column blocks top→bottom.
    - Full-width regions above the columns (e.g., title) come first; full-width below come last.
    - Otherwise fall back to (y0, x0) ordering.
    """

    if _should_use_band_graph_order(elements):
        return _band_graph_order(elements)
    return _xy_cut_order(elements, max_depth=10)


def _should_use_band_graph_order(elements: list[LayoutElement]) -> bool:
    if len(elements) < 4:
        return False
    if _column_center_split(elements) is not None:
        return False
    if _prefer_column_major_tie_break(elements):
        return False

    split_x = _best_gap_split(elements, axis="x")
    split_y = _best_gap_split(elements, axis="y")
    x_gap = float(split_x.gap) if split_x is not None else 0.0
    y_gap = float(split_y.gap) if split_y is not None else 0.0

    # Strong column/page partitioning should stay on XY-cut; otherwise use band+graph.
    if x_gap >= 0.08 or y_gap >= 0.08:
        return False
    return True


@dataclass(frozen=True)
class _Band:
    index: int
    y0: float
    y1: float
    element_indices: tuple[int, ...]


def _band_graph_order(elements: list[LayoutElement]) -> list[LayoutElement]:
    if len(elements) <= 1:
        return list(elements)

    nodes = list(elements)
    bands = _build_vertical_bands(nodes)
    band_by_node: dict[int, int] = {}
    for band in bands:
        for node_index in band.element_indices:
            band_by_node[node_index] = band.index

    n = len(nodes)
    edges: list[set[int]] = [set() for _ in range(n)]
    indeg = [0] * n
    eps = 0.01

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bi = band_by_node.get(i, 0)
            bj = band_by_node.get(j, 0)
            if bi < bj and nodes[i].bbox.y0 <= nodes[j].bbox.y0 + eps:
                edges[i].add(j)
            elif bi == bj and nodes[i].bbox.x1 <= nodes[j].bbox.x0 + eps:
                edges[i].add(j)

            if _precedes(nodes[i], nodes[j]):
                edges[i].add(j)

    for i in range(n):
        for j in edges[i]:
            indeg[j] += 1

    def tie_key(idx: int) -> tuple[float, float, float, float, float, str]:
        b = float(band_by_node.get(idx, 0))
        e = nodes[idx]
        return (b, e.bbox.y0, e.bbox.x0, e.bbox.y1, e.bbox.x1, e.label)

    ready = sorted([i for i in range(n) if indeg[i] == 0], key=tie_key)
    out: list[int] = []
    while ready:
        idx = ready.pop(0)
        out.append(idx)
        for j in sorted(edges[idx]):
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
        ready.sort(key=tie_key)

    if len(out) != n:
        return sorted(nodes, key=lambda e: (e.bbox.y0, e.bbox.x0, e.bbox.y1, e.bbox.x1, e.label))

    return [nodes[i] for i in out]


def _build_vertical_bands(elements: list[LayoutElement]) -> list[_Band]:
    if not elements:
        return []
    order = sorted(range(len(elements)), key=lambda i: (elements[i].bbox.y0, elements[i].bbox.x0))
    bands_raw: list[list[int]] = []
    band_y0: list[float] = []
    band_y1: list[float] = []

    for idx in order:
        e = elements[idx]
        if not bands_raw:
            bands_raw.append([idx])
            band_y0.append(float(e.bbox.y0))
            band_y1.append(float(e.bbox.y1))
            continue

        last = len(bands_raw) - 1
        by0 = band_y0[last]
        by1 = band_y1[last]
        overlap = max(0.0, min(by1, e.bbox.y1) - max(by0, e.bbox.y0))
        e_h = max(1e-6, e.bbox.y1 - e.bbox.y0)
        b_h = max(1e-6, by1 - by0)
        overlap_ratio = overlap / min(e_h, b_h)
        gap = float(e.bbox.y0 - by1)

        if overlap_ratio >= 0.20 or gap <= 0.02:
            bands_raw[last].append(idx)
            band_y0[last] = min(by0, float(e.bbox.y0))
            band_y1[last] = max(by1, float(e.bbox.y1))
            continue

        bands_raw.append([idx])
        band_y0.append(float(e.bbox.y0))
        band_y1.append(float(e.bbox.y1))

    return [
        _Band(
            index=i,
            y0=band_y0[i],
            y1=band_y1[i],
            element_indices=tuple(
                sorted(
                    bands_raw[i],
                    key=lambda n: (elements[n].bbox.x0, elements[n].bbox.y0),
                )
            ),
        )
        for i in range(len(bands_raw))
    ]


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

    center_split = _column_center_split(elements)
    if center_split is not None:
        left_group, right_group = center_split
        first = _xy_cut_order_inner(left_group, depth=depth + 1, max_depth=max_depth)
        second = _xy_cut_order_inner(right_group, depth=depth + 1, max_depth=max_depth)
        return first + second

    split_y = _best_gap_split(elements, axis="y")
    split_x = _best_gap_split(elements, axis="x")

    min_gap = 0.03
    has_y = split_y is not None and split_y.gap >= min_gap
    has_x = split_x is not None and split_x.gap >= min_gap

    if not has_y and not has_x:
        return _topo_order(elements)

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


def _column_center_split(
    elements: list[LayoutElement],
) -> tuple[list[LayoutElement], list[LayoutElement]] | None:
    if len(elements) < 4:
        return None

    widths = [max(0.0, e.bbox.x1 - e.bbox.x0) for e in elements]
    if any(w >= 0.8 for w in widths):
        # Full-width blocks should be handled by gap-based splitting.
        return None

    centers = sorted((e.bbox.x0 + e.bbox.x1) / 2.0 for e in elements)
    mid = len(centers) // 2
    if len(centers) % 2 == 0:
        split = (centers[mid - 1] + centers[mid]) / 2.0
    else:
        split = centers[mid]

    left = [e for e in elements if ((e.bbox.x0 + e.bbox.x1) / 2.0) <= split]
    right = [e for e in elements if ((e.bbox.x0 + e.bbox.x1) / 2.0) > split]
    if len(left) < 2 or len(right) < 2:
        return None

    left_centers = sorted((e.bbox.x0 + e.bbox.x1) / 2.0 for e in left)
    right_centers = sorted((e.bbox.x0 + e.bbox.x1) / 2.0 for e in right)
    separation = right_centers[0] - left_centers[-1]
    if separation <= 0.05:
        return None

    left_y0 = min(e.bbox.y0 for e in left)
    left_y1 = max(e.bbox.y1 for e in left)
    right_y0 = min(e.bbox.y0 for e in right)
    right_y1 = max(e.bbox.y1 for e in right)
    overlap = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
    left_span = max(1e-6, left_y1 - left_y0)
    right_span = max(1e-6, right_y1 - right_y0)
    overlap_ratio = overlap / min(left_span, right_span)
    if overlap_ratio < 0.5:
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
    prefer_column_major = _prefer_column_major_tie_break(nodes)

    def tie_key(e: LayoutElement) -> tuple[float, float, float, float, str]:
        if prefer_column_major:
            return (e.bbox.x0, e.bbox.y0, e.bbox.y1, e.bbox.x1, e.label)
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


def _prefer_column_major_tie_break(elements: list[LayoutElement]) -> bool:
    if len(elements) < 4:
        return False

    centers = sorted((e.bbox.x0 + e.bbox.x1) / 2.0 for e in elements)
    mid = len(centers) // 2
    if len(centers) % 2 == 0:
        split = (centers[mid - 1] + centers[mid]) / 2.0
    else:
        split = centers[mid]
    left = [e for e in elements if ((e.bbox.x0 + e.bbox.x1) / 2.0) <= split]
    right = [e for e in elements if ((e.bbox.x0 + e.bbox.x1) / 2.0) > split]
    if len(left) < 2 or len(right) < 2:
        return False

    def _y_bounds(group: list[LayoutElement]) -> tuple[float, float]:
        return min(e.bbox.y0 for e in group), max(e.bbox.y1 for e in group)

    left_y0, left_y1 = _y_bounds(left)
    right_y0, right_y1 = _y_bounds(right)
    overlap = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
    left_span = max(1e-6, left_y1 - left_y0)
    right_span = max(1e-6, right_y1 - right_y0)
    overlap_ratio = overlap / min(left_span, right_span)

    # Prefer column-major tie-break when both sides span similar vertical bands.
    return overlap_ratio >= 0.5


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
    table_words: list[Word] | None,
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
        return _table_from_spans(text=text, spans=spans, words=table_words)
    if normalized in {"figure", "image"}:
        return Figure(alt=text)
    return Unknown(text=text)


def _table_from_spans(
    *,
    text: str,
    spans: list[Span],
    words: list[Word] | None = None,
) -> Table:
    # Best-effort: infer a grid from span positions. If uncertain, fall back to plain text.
    candidate_words: list[Word]
    if words:
        candidate_words = [w for w in words if str(w.text).strip()]
    else:
        candidate_words = [
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

    candidates = _candidate_table_column_counts(candidate_words)
    if not candidates:
        candidates = tuple(range(2, 7))
    column_target = _estimate_table_column_target(candidate_words)

    best_rows: tuple[tuple[str, ...], ...] | None = None
    best_cells: tuple[TableCell, ...] | None = None
    best_score = -1.0
    best_filled = -1
    best_conf = -1.0
    best_k = 0
    for k in candidates:
        result = build_table_grid(candidate_words, column_count=k)
        if not result.ok or result.grid is None:
            continue
        rows = result.grid.rows
        cells = result.grid.cells
        total = max(1, len(rows) * k)
        filled = sum(1 for r in rows for c in r[:k] if str(c).strip())
        fill_ratio = filled / total
        sparse_rows = 0
        for row in rows:
            non_empty = sum(1 for cell in row[:k] if str(cell).strip())
            if non_empty <= 1:
                sparse_rows += 1
        sparse_ratio = (sparse_rows / max(1, len(rows))) if rows else 1.0
        conf = float(result.confidence)
        target_alignment = 0.0
        if column_target is not None:
            target_alignment = max(
                0.0,
                1.0 - (abs(k - column_target) / max(1.0, float(column_target - 1))),
            )
        score = (
            (fill_ratio * 0.70)
            + (conf * 0.30)
            - (sparse_ratio * 0.15)
            + (target_alignment * 0.25)
        )

        if (
            score > best_score + 1e-9
            or (
                abs(score - best_score) <= 1e-9
                and (filled > best_filled or (filled == best_filled and conf > best_conf))
            )
            or (
                abs(score - best_score) <= 1e-9
                and filled == best_filled
                and abs(conf - best_conf) <= 1e-9
                and k > best_k
            )
        ):
            best_score = score
            best_filled = filled
            best_conf = conf
            best_k = k
            best_rows = rows
            best_cells = tuple(
                TableCell(
                    row=int(c.row),
                    col=int(c.col),
                    text=str(c.text),
                    colspan=max(1, int(c.colspan)),
                    rowspan=max(1, int(c.rowspan)),
                )
                for c in cells
            )

    return Table(text=text, grid=best_rows, cells=best_cells)


def _candidate_table_column_counts(words: list[Word]) -> tuple[int, ...]:
    observed_counts = _observed_column_group_counts(words)
    if not observed_counts:
        return (2, 3, 4, 5, 6)

    median_count = int(round(_median_or_default([float(c) for c in observed_counts], default=2.0)))
    candidates: set[int] = set()
    for count in observed_counts:
        if 2 <= count <= 8:
            candidates.add(count)

    for count in (
        min(observed_counts),
        median_count - 1,
        median_count,
        median_count + 1,
        max(observed_counts),
    ):
        if 2 <= count <= 8:
            candidates.add(int(count))

    if not candidates:
        return (2, 3, 4, 5, 6)

    return tuple(sorted(candidates))


def _estimate_table_column_target(words: list[Word]) -> int | None:
    observed_counts = _observed_column_group_counts(words)
    if not observed_counts:
        return None
    frequencies: dict[int, int] = {}
    for count in observed_counts:
        frequencies[count] = frequencies.get(count, 0) + 1

    best_count = max(
        frequencies.items(),
        key=lambda item: (item[1], item[0]),
    )[0]
    if best_count < 2:
        return None
    return min(8, int(best_count))


def _observed_column_group_counts(words: list[Word]) -> list[int]:
    if not words:
        return []

    heights = [max(0.0, w.y1 - w.y0) for w in words]
    widths = [max(0.0, w.x1 - w.x0) for w in words]
    median_h = _median_or_default(heights, default=10.0)
    median_w = _median_or_default(widths, default=20.0)

    row_bin = max(4.0, median_h * 0.75)
    rows: dict[int, list[Word]] = {}
    for w in words:
        cy = (w.y0 + w.y1) / 2.0
        row_key = int(round(cy / row_bin)) if row_bin > 0 else 0
        rows.setdefault(row_key, []).append(w)

    if not rows:
        return []

    gap_threshold = max(10.0, median_h * 1.15, median_w * 0.60)
    observed_counts: list[int] = []
    for row_words in rows.values():
        count = _count_groups_by_x_gap(row_words, gap_threshold=gap_threshold)
        if count >= 2:
            observed_counts.append(count)

    return observed_counts


def _count_groups_by_x_gap(words: list[Word], *, gap_threshold: float) -> int:
    if not words:
        return 0
    ordered = sorted(words, key=lambda w: (w.x0, w.x1, w.text))
    groups = 1
    prev = ordered[0]
    for word in ordered[1:]:
        gap = float(word.x0 - prev.x1)
        if gap > gap_threshold:
            groups += 1
        prev = word
    return groups

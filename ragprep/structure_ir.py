from __future__ import annotations

import statistics
from dataclasses import dataclass

from ragprep.pdf_text import Span


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

    elements_sorted = sorted(
        layout_elements,
        key=lambda e: (e.bbox.y0, e.bbox.x0, e.bbox.y1, e.bbox.x1, e.label),
    )

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

    blocks: list[Block] = []
    for i, elt in enumerate(elements_sorted):
        collected = assignments[i]
        if not collected:
            continue
        text = _join_spans_text(collected)
        if not text:
            continue
        blocks.append(_block_from_label(elt.label, text))

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
    rendered_lines: list[str] = []
    for key in line_keys:
        rendered = _join_spans_in_line(lines[key])
        if rendered:
            rendered_lines.append(rendered)

    return "\n".join(rendered_lines).strip()


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


def _block_from_label(label: str, text: str) -> Block:
    normalized = (label or "").strip().lower()
    if normalized in {"title", "heading"}:
        return Heading(level=1, text=text)
    if normalized in {"text", "paragraph"}:
        return Paragraph(text=text)
    if normalized == "table":
        return Table(text=text)
    if normalized in {"figure", "image"}:
        return Figure(alt=text)
    return Unknown(text=text)

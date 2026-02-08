from __future__ import annotations

import statistics
from dataclasses import dataclass

from ragprep.pdf_text import Span


@dataclass(frozen=True)
class _SegmentFeature:
    spans: tuple[Span, ...]
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    median_size: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


@dataclass(frozen=True)
class _RowFeature:
    spans: tuple[Span, ...]
    segments: tuple[_SegmentFeature, ...]
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    median_size: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cell_count(self) -> int:
        return len(self.segments)


@dataclass(frozen=True)
class _LineFeature:
    spans: tuple[Span, ...]
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    median_size: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


def infer_fast_layout_elements(
    spans: list[Span],
    page_width: float,
    page_height: float,
) -> list[dict[str, object]]:
    if page_width <= 0 or page_height <= 0:
        raise ValueError("page_width/page_height must be > 0")

    cleaned = [s for s in spans if isinstance(s.text, str) and s.text.strip()]
    if not cleaned:
        return []

    ordered = sorted(cleaned, key=lambda s: (float(s.y0), float(s.x0), float(s.x1)))
    rows = _cluster_spans_to_rows(ordered)
    if not rows:
        return []

    row_features = [_build_row_feature(row, page_width=page_width) for row in rows]
    if not row_features:
        return []

    page_median_size = _median(
        [f.median_size for f in row_features if f.median_size > 0],
        default=10.0,
    )

    table_groups = _detect_table_groups(row_features, page_width=page_width)
    table_row_indexes = {row_idx for group in table_groups for row_idx in group}
    column_split_x = _estimate_two_column_split(
        row_features,
        table_row_indexes=table_row_indexes,
        page_width=page_width,
    )

    text_lines: list[_LineFeature] = []
    for row_idx, row in enumerate(row_features):
        if row_idx in table_row_indexes:
            continue
        text_lines.extend(
            _split_row_into_text_lines(
                row,
                page_width=page_width,
                column_split_x=column_split_x,
            )
        )

    labels = _label_lines(
        text_lines,
        page_width=page_width,
        page_height=page_height,
        page_median_size=page_median_size,
    )
    merged = _merge_text_lines(
        text_lines,
        labels,
        page_width=page_width,
        column_split_x=column_split_x,
    )

    elements: list[dict[str, object]] = []
    for group in table_groups:
        rows_in_group = [row_features[i] for i in group]
        table_feature = _merge_rows(rows_in_group)
        if not table_feature.text:
            continue
        x0 = max(0.0, min(float(page_width), table_feature.x0))
        y0 = max(0.0, min(float(page_height), table_feature.y0))
        x1 = max(0.0, min(float(page_width), table_feature.x1))
        y1 = max(0.0, min(float(page_height), table_feature.y1))
        if not (x0 < x1 and y0 < y1):
            continue
        elements.append(
            {
                "page_index": 0,
                "bbox": (x0, y0, x1, y1),
                "label": "table",
                "score": None,
                "_order": (y0, x0, 0),
            }
        )

    for idx, (feature, label) in enumerate(merged):
        x0 = max(0.0, min(float(page_width), feature.x0))
        y0 = max(0.0, min(float(page_height), feature.y0))
        x1 = max(0.0, min(float(page_width), feature.x1))
        y1 = max(0.0, min(float(page_height), feature.y1))
        if not (x0 < x1 and y0 < y1):
            continue
        elements.append(
            {
                "page_index": 0,
                "bbox": (x0, y0, x1, y1),
                "label": label,
                "score": None,
                "_order": (y0, x0, 1 + idx),
            }
        )

    elements.sort(key=_element_order_key)
    for e in elements:
        e.pop("_order", None)
    return elements


def _cluster_spans_to_rows(spans: list[Span]) -> list[list[Span]]:
    heights = [max(0.0, float(s.y1) - float(s.y0)) for s in spans]
    median_h = _median([h for h in heights if h > 0], default=10.0)
    line_threshold = max(1.5, median_h * 0.65)

    rows: list[list[Span]] = []
    for span in spans:
        cy = (float(span.y0) + float(span.y1)) / 2.0
        if not rows:
            rows.append([span])
            continue

        last = rows[-1]
        last_center = statistics.mean((float(s.y0) + float(s.y1)) / 2.0 for s in last)
        if abs(cy - last_center) <= line_threshold:
            last.append(span)
            continue
        rows.append([span])

    for row in rows:
        row.sort(key=lambda s: (float(s.x0), float(s.x1), str(s.text)))
    return rows


def _build_row_feature(row: list[Span], *, page_width: float) -> _RowFeature:
    x0 = min(float(s.x0) for s in row)
    y0 = min(float(s.y0) for s in row)
    x1 = max(float(s.x1) for s in row)
    y1 = max(float(s.y1) for s in row)

    sizes = [float(s.size) for s in row if isinstance(s.size, (int, float)) and float(s.size) > 0]
    median_size = _median(sizes, default=max(1.0, y1 - y0))

    heights = [max(0.0, float(s.y1) - float(s.y0)) for s in row]
    median_h = _median([h for h in heights if h > 0], default=10.0)
    segment_gap = max(8.0, median_h * 1.4, page_width * 0.01)
    segment_rows = _split_row_by_gap(row, split_gap=segment_gap)
    segments = tuple(_build_segment_feature(seg) for seg in segment_rows)

    text = " ".join(seg.text for seg in segments if seg.text).strip()
    return _RowFeature(
        spans=tuple(row),
        segments=segments,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        median_size=median_size,
    )


def _split_row_by_gap(row: list[Span], *, split_gap: float) -> list[list[Span]]:
    if not row:
        return []
    ordered = sorted(row, key=lambda s: (float(s.x0), float(s.x1), str(s.text)))
    groups: list[list[Span]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        gap = float(cur.x0) - float(prev.x1)
        if gap >= split_gap:
            groups.append([cur])
            continue
        groups[-1].append(cur)
    return groups


def _build_segment_feature(segment: list[Span]) -> _SegmentFeature:
    x0 = min(float(s.x0) for s in segment)
    y0 = min(float(s.y0) for s in segment)
    x1 = max(float(s.x1) for s in segment)
    y1 = max(float(s.y1) for s in segment)
    text = _join_line_text(segment)
    sizes = [
        float(s.size)
        for s in segment
        if isinstance(s.size, (int, float)) and float(s.size) > 0
    ]
    median_size = _median(sizes, default=max(1.0, y1 - y0))
    return _SegmentFeature(
        spans=tuple(segment),
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        median_size=median_size,
    )


def _line_from_segments(segments: list[_SegmentFeature]) -> _LineFeature:
    x0 = min(seg.x0 for seg in segments)
    y0 = min(seg.y0 for seg in segments)
    x1 = max(seg.x1 for seg in segments)
    y1 = max(seg.y1 for seg in segments)
    text = " ".join(seg.text for seg in segments if seg.text).strip()
    return _LineFeature(
        spans=tuple(span for seg in segments for span in seg.spans),
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        median_size=_median([seg.median_size for seg in segments], default=max(1.0, y1 - y0)),
    )


def _join_line_text(line: list[Span]) -> str:
    heights = [max(0.0, float(s.y1) - float(s.y0)) for s in line]
    median_h = _median([h for h in heights if h > 0], default=10.0)
    gap_threshold = max(1.0, median_h * 0.25)

    parts: list[str] = []
    prev: Span | None = None
    for span in line:
        text = str(span.text).strip()
        if not text:
            continue
        if prev is not None:
            gap = float(span.x0) - float(prev.x1)
            if gap >= gap_threshold:
                parts.append(" ")
        parts.append(text)
        prev = span
    return "".join(parts).strip()


def _detect_table_groups(rows: list[_RowFeature], *, page_width: float) -> list[tuple[int, ...]]:
    if not rows:
        return []

    row_heights = [r.height for r in rows if r.height > 0]
    row_gap_threshold = max(8.0, _median(row_heights, default=10.0) * 1.8)

    groups: list[tuple[int, ...]] = []
    i = 0
    while i < len(rows):
        current = rows[i]
        if not _is_table_row_candidate(current, page_width=page_width):
            i += 1
            continue

        group = [i]
        j = i + 1
        while j < len(rows):
            prev_idx = group[-1]
            prev = rows[prev_idx]
            nxt = rows[j]
            gap = float(nxt.y0 - prev.y1)
            if gap > row_gap_threshold:
                break
            if not _is_table_row_candidate(nxt, page_width=page_width):
                break
            if not _rows_are_aligned(prev, nxt, page_width=page_width):
                break
            group.append(j)
            j += 1

        if len(group) >= 2 or _is_single_row_table_candidate(current, page_width=page_width):
            groups.append(tuple(group))
            i = j
            continue
        i += 1
    return groups


def _is_table_row_candidate(row: _RowFeature, *, page_width: float) -> bool:
    if row.cell_count < 3:
        return False
    if row.width / max(1e-6, page_width) < 0.30:
        return False

    lengths = [len(seg.text) for seg in row.segments if seg.text]
    if not lengths:
        return False
    if max(lengths) > 80:
        return False

    short_cells = sum(1 for n in lengths if n <= 24)
    if short_cells < max(2, int(len(lengths) * 0.6)):
        return False

    total = max(1, sum(lengths))
    if max(lengths) / total > 0.70 and len(lengths) >= 3:
        return False
    return True


def _is_single_row_table_candidate(row: _RowFeature, *, page_width: float) -> bool:
    if row.cell_count < 4:
        return False
    if row.width / max(1e-6, page_width) < 0.45:
        return False
    lengths = [len(seg.text) for seg in row.segments if seg.text]
    if not lengths:
        return False
    if max(lengths) > 24:
        return False
    return True


def _rows_are_aligned(left: _RowFeature, right: _RowFeature, *, page_width: float) -> bool:
    left_starts = [seg.x0 / max(1e-6, page_width) for seg in left.segments]
    right_starts = [seg.x0 / max(1e-6, page_width) for seg in right.segments]
    if abs(len(left_starts) - len(right_starts)) > 1:
        return False

    tolerance = max(0.015, 14.0 / max(1e-6, page_width))
    i = 0
    j = 0
    matched = 0
    while i < len(left_starts) and j < len(right_starts):
        diff = left_starts[i] - right_starts[j]
        if abs(diff) <= tolerance:
            matched += 1
            i += 1
            j += 1
            continue
        if diff < 0:
            i += 1
            continue
        j += 1

    required = max(2, min(len(left_starts), len(right_starts)))
    return matched >= required


def _split_row_into_text_lines(
    row: _RowFeature,
    *,
    page_width: float,
    column_split_x: float | None,
) -> list[_LineFeature]:
    if row.cell_count <= 1:
        return [_line_from_segments(list(row.segments))]

    if column_split_x is not None:
        left_segments: list[_SegmentFeature] = []
        right_segments: list[_SegmentFeature] = []
        for seg in row.segments:
            center_x = (seg.x0 + seg.x1) / 2.0
            if center_x <= column_split_x:
                left_segments.append(seg)
            else:
                right_segments.append(seg)
        if left_segments and right_segments:
            left_is_column = _looks_like_column_chunk(left_segments, page_width=page_width)
            right_is_column = _looks_like_column_chunk(right_segments, page_width=page_width)
            if left_is_column and right_is_column:
                left_len = sum(len(seg.text) for seg in left_segments)
                right_len = sum(len(seg.text) for seg in right_segments)
                if left_len >= 12 and right_len >= 12:
                    return [
                        _line_from_segments(left_segments),
                        _line_from_segments(right_segments),
                    ]

    gaps: list[float] = []
    for prev, cur in zip(row.segments, row.segments[1:], strict=False):
        gaps.append(float(cur.x0 - prev.x1))
    if not gaps:
        return [_line_from_segments(list(row.segments))]

    split_threshold = max(56.0, page_width * 0.12)
    best_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    best_gap = gaps[best_idx]
    if best_gap < split_threshold:
        return [_line_from_segments(list(row.segments))]

    left = list(row.segments[: best_idx + 1])
    right = list(row.segments[best_idx + 1 :])
    if not left or not right:
        return [_line_from_segments(list(row.segments))]
    if not _looks_like_column_chunk(left, page_width=page_width):
        return [_line_from_segments(list(row.segments))]
    if not _looks_like_column_chunk(right, page_width=page_width):
        return [_line_from_segments(list(row.segments))]
    return [_line_from_segments(left), _line_from_segments(right)]


def _looks_like_column_chunk(segments: list[_SegmentFeature], *, page_width: float) -> bool:
    if not segments:
        return False
    chunk_x0 = min(seg.x0 for seg in segments)
    chunk_x1 = max(seg.x1 for seg in segments)
    width_ratio = max(0.0, chunk_x1 - chunk_x0) / max(1e-6, page_width)
    text_len = sum(len(seg.text) for seg in segments)
    return width_ratio <= 0.70 and text_len >= 6


def _estimate_two_column_split(
    rows: list[_RowFeature],
    *,
    table_row_indexes: set[int],
    page_width: float,
) -> float | None:
    centers: list[float] = []
    for row_idx, row in enumerate(rows):
        if row_idx in table_row_indexes:
            continue
        for seg in row.segments:
            if len(seg.text) < 6:
                continue
            if seg.width / max(1e-6, page_width) < 0.08:
                continue
            centers.append((seg.x0 + seg.x1) / 2.0)

    if len(centers) < 20:
        return None

    ordered = sorted(float(c) for c in centers)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    if not gaps:
        return None
    best_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    best_gap = gaps[best_idx]
    if best_gap < max(40.0, page_width * 0.10):
        return None

    split_x = (ordered[best_idx] + ordered[best_idx + 1]) / 2.0
    if split_x < page_width * 0.25 or split_x > page_width * 0.75:
        return None

    left_count = sum(1 for c in ordered if c <= split_x)
    right_count = len(ordered) - left_count
    min_side = max(6, int(len(ordered) * 0.20))
    if left_count < min_side or right_count < min_side:
        return None
    return split_x


def _label_lines(
    lines: list[_LineFeature],
    *,
    page_width: float,
    page_height: float,
    page_median_size: float,
) -> list[str]:
    labels = ["text"] * len(lines)
    if not lines:
        return labels

    title_index = _pick_title_index(
        lines,
        page_height=page_height,
        page_median_size=page_median_size,
    )
    for idx, line in enumerate(lines):
        text_len = len(line.text)
        size_ratio = line.median_size / max(1e-6, page_median_size)
        width_ratio = line.width / max(1e-6, page_width)
        is_short = text_len <= 120

        if idx == title_index:
            labels[idx] = "title"
            continue

        is_heading = is_short and size_ratio >= 1.20 and width_ratio <= 0.95
        if is_heading:
            labels[idx] = "heading"
            continue

        labels[idx] = "text"
    return labels


def _pick_title_index(
    lines: list[_LineFeature],
    *,
    page_height: float,
    page_median_size: float,
) -> int | None:
    if len(lines) == 1:
        only = lines[0]
        if only.y0 / max(1e-6, page_height) <= 0.20 and len(only.text) <= 120:
            return 0

    candidates: list[int] = []
    for idx, line in enumerate(lines):
        if line.y0 / max(1e-6, page_height) > 0.20:
            continue
        if len(line.text) > 120:
            continue
        candidates.append(idx)

    if not candidates:
        return None

    best_idx = max(
        candidates,
        key=lambda i: (
            lines[i].median_size / max(1e-6, page_median_size),
            -len(lines[i].text),
            -lines[i].width,
        ),
    )
    best_ratio = lines[best_idx].median_size / max(1e-6, page_median_size)
    if best_ratio < 1.25:
        top_ratio = lines[best_idx].y0 / max(1e-6, page_height)
        if top_ratio > 0.08 or len(lines[best_idx].text) > 80:
            return None
    return best_idx


def _merge_text_lines(
    lines: list[_LineFeature],
    labels: list[str],
    *,
    page_width: float,
    column_split_x: float | None,
) -> list[tuple[_LineFeature, str]]:
    if not lines:
        return []

    heights = [line.height for line in lines if line.height > 0]
    median_h = _median(heights, default=10.0)
    paragraph_gap = max(6.0, median_h * 1.9)

    merged: list[tuple[_LineFeature, str]] = []
    current: _LineFeature | None = None
    current_side: int | None = None
    for line, label in zip(lines, labels, strict=True):
        if label != "text":
            if current is not None:
                merged.append((current, "text"))
                current = None
                current_side = None
            merged.append((line, label))
            continue

        if current is None:
            current = line
            current_side = (
                _line_column_side(line, split_x=column_split_x)
                if column_split_x is not None
                else None
            )
            continue

        gap = line.y0 - current.y1
        if gap >= paragraph_gap:
            merged.append((current, "text"))
            current = line
            current_side = (
                _line_column_side(line, split_x=column_split_x)
                if column_split_x is not None
                else None
            )
            continue

        if column_split_x is not None:
            next_side = _line_column_side(line, split_x=column_split_x)
            if current_side is None:
                current_side = _line_column_side(current, split_x=column_split_x)
            if current_side != 0 and next_side != 0 and current_side != next_side:
                merged.append((current, "text"))
                current = line
                current_side = next_side
                continue
            if current_side == 0 and next_side != 0:
                merged.append((current, "text"))
                current = line
                current_side = next_side
                continue
            if current_side != 0 and next_side == 0:
                merged.append((current, "text"))
                current = line
                current_side = next_side
                continue

        overlap_ratio = _x_overlap_ratio(current, line)
        left_delta = abs(line.x0 - current.x0)
        if overlap_ratio < 0.18 and left_delta > max(10.0, page_width * 0.04):
            merged.append((current, "text"))
            current = line
            current_side = (
                _line_column_side(line, split_x=column_split_x)
                if column_split_x is not None
                else None
            )
            continue

        merged_text = _merge_text(current.text, line.text)
        current = _LineFeature(
            spans=current.spans + line.spans,
            x0=min(current.x0, line.x0),
            y0=min(current.y0, line.y0),
            x1=max(current.x1, line.x1),
            y1=max(current.y1, line.y1),
            text=merged_text,
            median_size=_median(
                [current.median_size, line.median_size],
                default=current.median_size,
            ),
        )
        if column_split_x is not None:
            next_side = _line_column_side(line, split_x=column_split_x)
            if current_side is None:
                current_side = next_side
            elif current_side == 0 and next_side != 0:
                current_side = next_side

    if current is not None:
        merged.append((current, "text"))
    return merged


def _x_overlap_ratio(left: _LineFeature, right: _LineFeature) -> float:
    overlap = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    min_w = max(1e-6, min(left.width, right.width))
    return overlap / min_w


def _line_column_side(line: _LineFeature, *, split_x: float) -> int:
    if line.x1 <= split_x:
        return -1
    if line.x0 >= split_x:
        return 1
    return 0


def _merge_rows(rows: list[_RowFeature]) -> _LineFeature:
    if not rows:
        raise ValueError("rows must not be empty")
    x0 = min(r.x0 for r in rows)
    y0 = min(r.y0 for r in rows)
    x1 = max(r.x1 for r in rows)
    y1 = max(r.y1 for r in rows)
    text = "\n".join(r.text for r in rows if r.text).strip()
    return _LineFeature(
        spans=tuple(span for row in rows for span in row.spans),
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        median_size=_median([r.median_size for r in rows], default=max(1.0, y1 - y0)),
    )


def _merge_text(left: str, right: str) -> str:
    left_clean = left.strip()
    right_clean = right.strip()
    if not left_clean:
        return right_clean
    if not right_clean:
        return left_clean
    if left_clean.endswith("-") and right_clean[:1].isalpha():
        return left_clean[:-1] + right_clean
    return left_clean + " " + right_clean


def _element_order_key(element: dict[str, object]) -> tuple[float, float, int]:
    raw = element.get("_order")
    if isinstance(raw, tuple) and len(raw) == 3:
        return (float(raw[0]), float(raw[1]), int(raw[2]))
    return (0.0, 0.0, 0)


def _median(values: list[float], *, default: float) -> float:
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and v > 0 and v == v]
    if not cleaned:
        return float(default)
    try:
        return float(statistics.median(cleaned))
    except statistics.StatisticsError:
        return float(default)

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ragprep.pdf_text import Span


@dataclass(frozen=True)
class _LineFeature:
    spans: tuple[Span, ...]
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    median_size: float
    gap_count: int

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
    lines = _cluster_spans_to_lines(ordered)
    if not lines:
        return []

    line_features = [_build_line_feature(line) for line in lines]
    if not line_features:
        return []

    page_median_size = _median(
        [f.median_size for f in line_features if f.median_size > 0],
        default=10.0,
    )

    labels = _label_lines(
        line_features,
        page_width=page_width,
        page_height=page_height,
        page_median_size=page_median_size,
    )
    merged = _merge_text_lines(line_features, labels)

    elements: list[dict[str, object]] = []
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
                "_row": idx,
            }
        )

    elements.sort(key=lambda e: (float(e["bbox"][1]), float(e["bbox"][0]), str(e["label"])))
    for e in elements:
        e.pop("_row", None)
    return elements


def _cluster_spans_to_lines(spans: list[Span]) -> list[list[Span]]:
    heights = [max(0.0, float(s.y1) - float(s.y0)) for s in spans]
    median_h = _median([h for h in heights if h > 0], default=10.0)
    line_threshold = max(1.5, median_h * 0.65)

    lines: list[list[Span]] = []
    for span in spans:
        cy = (float(span.y0) + float(span.y1)) / 2.0
        if not lines:
            lines.append([span])
            continue

        last = lines[-1]
        last_center = statistics.mean((float(s.y0) + float(s.y1)) / 2.0 for s in last)
        if abs(cy - last_center) <= line_threshold:
            last.append(span)
            continue
        lines.append([span])

    for line in lines:
        line.sort(key=lambda s: (float(s.x0), float(s.x1), str(s.text)))
    return lines


def _build_line_feature(line: list[Span]) -> _LineFeature:
    x0 = min(float(s.x0) for s in line)
    y0 = min(float(s.y0) for s in line)
    x1 = max(float(s.x1) for s in line)
    y1 = max(float(s.y1) for s in line)
    text = _join_line_text(line)

    sizes = [float(s.size) for s in line if isinstance(s.size, (int, float)) and float(s.size) > 0]
    median_size = _median(sizes, default=max(1.0, y1 - y0))

    heights = [max(0.0, float(s.y1) - float(s.y0)) for s in line]
    median_h = _median([h for h in heights if h > 0], default=10.0)
    gap_threshold = max(2.0, median_h * 0.75)
    gap_count = 0
    for prev, cur in zip(line, line[1:], strict=False):
        gap = float(cur.x0) - float(prev.x1)
        if gap >= gap_threshold:
            gap_count += 1

    return _LineFeature(
        spans=tuple(line),
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        median_size=median_size,
        gap_count=gap_count,
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

    for idx, line in enumerate(lines):
        text_len = len(line.text)
        size_ratio = line.median_size / max(1e-6, page_median_size)
        width_ratio = line.width / max(1e-6, page_width)
        is_short = text_len <= 90

        is_table_candidate = (
            line.gap_count >= 2 and text_len >= 8 and width_ratio >= 0.35 and size_ratio <= 1.20
        )
        if is_table_candidate:
            labels[idx] = "table"
            continue

        top_ratio = line.y0 / max(1e-6, page_height)
        is_top_line = top_ratio <= 0.12

        is_title = idx == 0 and is_short and (
            size_ratio >= 1.35 or (is_top_line and width_ratio <= 0.70 and text_len <= 60)
        )
        if is_title:
            labels[idx] = "title"
            continue

        is_heading = size_ratio >= 1.25 and is_short
        if is_heading:
            labels[idx] = "heading"
            continue

        labels[idx] = "text"
    return labels


def _merge_text_lines(
    lines: list[_LineFeature], labels: list[str]
) -> list[tuple[_LineFeature, str]]:
    if not lines:
        return []

    heights = [line.height for line in lines if line.height > 0]
    median_h = _median(heights, default=10.0)
    paragraph_gap = max(6.0, median_h * 1.9)

    merged: list[tuple[_LineFeature, str]] = []
    current: _LineFeature | None = None
    for line, label in zip(lines, labels, strict=True):
        if label != "text":
            if current is not None:
                merged.append((current, "text"))
                current = None
            merged.append((line, label))
            continue

        if current is None:
            current = line
            continue

        gap = line.y0 - current.y1
        if gap >= paragraph_gap:
            merged.append((current, "text"))
            current = line
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
            gap_count=current.gap_count + line.gap_count,
        )

    if current is not None:
        merged.append((current, "text"))
    return merged


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


def _median(values: list[float], *, default: float) -> float:
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and v > 0 and v == v]
    if not cleaned:
        return float(default)
    try:
        return float(statistics.median(cleaned))
    except statistics.StatisticsError:
        return float(default)



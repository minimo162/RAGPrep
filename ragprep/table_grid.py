from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from dataclasses import dataclass

from ragprep.pdf_text import Word


@dataclass(frozen=True)
class TableGrid:
    column_count: int
    column_centers: tuple[float, ...]
    column_cutoffs: tuple[float, ...]
    row_bin: float
    row_keys: tuple[int, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class TableGridResult:
    grid: TableGrid | None
    confidence: float
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.grid is not None


def build_table_grid(words: list[Word], *, column_count: int) -> TableGridResult:
    """
    Build a simple table grid from PyMuPDF `words` (with coordinates).

    Design goals:
    - Deterministic (no randomness, no external deps).
    - Conservative: if columns cannot be separated confidently, return failure.
    - Column count is provided externally (from OCR table structure).
    """

    if column_count <= 0:
        return TableGridResult(grid=None, confidence=0.0, reason="invalid_column_count")
    if not words:
        return TableGridResult(grid=None, confidence=0.0, reason="no_words")

    median_height = _median_or_default([max(0.0, w.y1 - w.y0) for w in words], default=10.0)
    median_width = _median_or_default([max(0.0, w.x1 - w.x0) for w in words], default=20.0)

    row_bin = max(4.0, median_height * 0.75)
    rows_by_key = _group_words_by_row(words, row_bin=row_bin)
    if not rows_by_key:
        return TableGridResult(grid=None, confidence=0.0, reason="no_rows")

    sorted_row_keys = tuple(sorted(rows_by_key.keys()))
    sorted_rows = [sorted(rows_by_key[k], key=lambda w: w.x0) for k in sorted_row_keys]

    gap_threshold = max(12.0, median_height * 1.2)
    anchors = _extract_column_anchors(sorted_rows, gap_threshold=gap_threshold)
    if len(anchors) < column_count:
        return TableGridResult(grid=None, confidence=0.0, reason="insufficient_anchors")

    centers_list = _kmeans_1d(anchors, k=column_count, max_iter=20)
    if centers_list is None:
        return TableGridResult(grid=None, confidence=0.0, reason="kmeans_failed")

    centers = tuple(sorted(centers_list))
    min_sep = _min_adjacent_separation(centers)
    min_sep_threshold = max(8.0, median_width * 1.5)
    if min_sep < min_sep_threshold:
        return TableGridResult(
            grid=None,
            confidence=min(1.0, min_sep / max(1.0, min_sep_threshold)),
            reason="columns_not_separated",
        )

    cutoffs = tuple((centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1))
    space_gap = max(2.0, median_height * 0.2)

    grid_rows: list[tuple[str, ...]] = []
    for row_words in sorted_rows:
        cols: list[list[Word]] = [[] for _ in range(column_count)]
        for w in row_words:
            x_center = (w.x0 + w.x1) / 2.0
            col = bisect_right(cutoffs, x_center)
            cols[col].append(w)
        row_cells = tuple(_join_words(cell_words, space_gap=space_gap) for cell_words in cols)
        grid_rows.append(row_cells)

    confidence = min(1.0, min_sep / max(1.0, median_width * 3.0))
    return TableGridResult(
        grid=TableGrid(
            column_count=column_count,
            column_centers=centers,
            column_cutoffs=cutoffs,
            row_bin=row_bin,
            row_keys=sorted_row_keys,
            rows=tuple(grid_rows),
        ),
        confidence=confidence,
        reason=None,
    )


def _median_or_default(values: list[float], *, default: float) -> float:
    cleaned = [v for v in values if math.isfinite(v)]
    if not cleaned:
        return float(default)
    try:
        return float(statistics.median(cleaned))
    except statistics.StatisticsError:
        return float(default)


def _group_words_by_row(words: list[Word], *, row_bin: float) -> dict[int, list[Word]]:
    rows: dict[int, list[Word]] = {}
    for w in words:
        y_center = (w.y0 + w.y1) / 2.0
        key = int(round(y_center / row_bin)) if row_bin > 0 else 0
        rows.setdefault(key, []).append(w)
    return rows


def _extract_column_anchors(rows: list[list[Word]], *, gap_threshold: float) -> list[float]:
    anchors: list[float] = []
    for row_words in rows:
        groups = _group_words_by_x_gap(row_words, gap_threshold=gap_threshold)
        anchors.extend([g[0].x0 for g in groups if g])
    return anchors


def _group_words_by_x_gap(words: list[Word], *, gap_threshold: float) -> list[list[Word]]:
    if not words:
        return []

    groups: list[list[Word]] = []
    current: list[Word] = [words[0]]
    prev = words[0]
    for w in words[1:]:
        gap = w.x0 - prev.x1
        if gap <= gap_threshold:
            current.append(w)
        else:
            groups.append(current)
            current = [w]
        prev = w
    groups.append(current)
    return groups


def _kmeans_1d(values: list[float], *, k: int, max_iter: int) -> list[float] | None:
    if k <= 0:
        return None
    points = [v for v in values if math.isfinite(v)]
    if len(points) < k:
        return None

    points.sort()
    n = len(points)

    centers: list[float] = []
    for i in range(k):
        q = (i + 0.5) / k
        idx = int(round(q * (n - 1)))
        centers.append(points[idx])

    for _ in range(max_iter):
        clusters: list[list[float]] = [[] for _ in range(k)]
        for x in points:
            nearest = min(range(k), key=lambda j: (abs(x - centers[j]), j))
            clusters[nearest].append(x)

        moved = 0.0
        for j, cluster in enumerate(clusters):
            if cluster:
                new_center = sum(cluster) / len(cluster)
                moved = max(moved, abs(new_center - centers[j]))
                centers[j] = new_center
                continue

            # Deterministic empty-cluster recovery: pick the farthest point from any center.
            best_point = None
            best_dist = -1.0
            for x in points:
                dist = min(abs(x - c) for c in centers)
                if dist > best_dist:
                    best_dist = dist
                    best_point = x
            if best_point is None:
                return None
            moved = max(moved, abs(best_point - centers[j]))
            centers[j] = best_point

        if moved < 1e-3:
            break

    return centers


def _min_adjacent_separation(sorted_centers: tuple[float, ...]) -> float:
    if len(sorted_centers) < 2:
        return 0.0
    seps = [sorted_centers[i + 1] - sorted_centers[i] for i in range(len(sorted_centers) - 1)]
    return min(seps)


def _join_words(words: list[Word], *, space_gap: float) -> str:
    if not words:
        return ""
    sorted_words = sorted(words, key=lambda w: w.x0)

    out = [sorted_words[0].text]
    prev = sorted_words[0]
    for w in sorted_words[1:]:
        gap = w.x0 - prev.x1
        if gap >= space_gap:
            out.append(" ")
        out.append(w.text)
        prev = w
    return "".join(out).strip()

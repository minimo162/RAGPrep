from __future__ import annotations

from typing import cast

import pytest

from ragprep.pdf_text import _extract_words
from ragprep.table_grid import build_table_grid


def _make_pdf_bytes_with_aligned_table(rows: int = 5) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()

    y = 72
    for i in range(rows):
        page.insert_text((72, y), f"Item{i}")
        page.insert_text((200, y), f"{i}")
        page.insert_text((320, y), f"{i * 10}")
        y += 12

    return cast(bytes, doc.tobytes())


def _make_pdf_bytes_with_multiword_cell() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()

    y = 72
    page.insert_text((72, y), "Hello world")
    page.insert_text((200, y), "1")
    page.insert_text((320, y), "2")

    return cast(bytes, doc.tobytes())


def _make_pdf_bytes_single_column(rows: int = 8) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()

    y = 72
    for i in range(rows):
        page.insert_text((72, y), f"Only{i}")
        y += 12

    return cast(bytes, doc.tobytes())


def test_build_table_grid_builds_cells_for_aligned_columns() -> None:
    import fitz

    pdf_bytes = _make_pdf_bytes_with_aligned_table(rows=6)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    words = _extract_words(page)

    result = build_table_grid(words, column_count=3)
    assert result.ok, result.reason
    assert result.grid is not None

    assert len(result.grid.rows) == 6
    assert result.grid.rows[0] == ("Item0", "0", "0")
    assert result.grid.rows[5] == ("Item5", "5", "50")


def test_build_table_grid_joins_multiword_cells_with_spaces() -> None:
    import fitz

    pdf_bytes = _make_pdf_bytes_with_multiword_cell()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    words = _extract_words(page)

    result = build_table_grid(words, column_count=3)
    assert result.ok, result.reason
    assert result.grid is not None
    assert result.grid.rows[0][0] == "Hello world"


def test_build_table_grid_fails_when_columns_not_separable() -> None:
    import fitz

    pdf_bytes = _make_pdf_bytes_single_column()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    words = _extract_words(page)

    result = build_table_grid(words, column_count=3)
    assert result.ok is False
    assert result.grid is None
    assert result.reason in {"columns_not_separated", "kmeans_failed", "insufficient_anchors"}
    assert result.confidence == pytest.approx(0.0, abs=1e-9) or 0.0 <= result.confidence <= 1.0

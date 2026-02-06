from __future__ import annotations

from typing import cast

import pytest

from ragprep.pdf_text import Word, _extract_words
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


def _make_tight_five_column_words(rows: int = 4) -> list[Word]:
    words: list[Word] = []
    x_starts = (10.0, 65.0, 120.0, 175.0, 230.0)
    for row_index in range(rows):
        y0 = 10.0 + (row_index * 20.0)
        y1 = y0 + 10.0
        for col_index, x0 in enumerate(x_starts):
            words.append(
                Word(
                    x0=x0,
                    y0=y0,
                    x1=x0 + 24.0,
                    y1=y1,
                    text=f"R{row_index}C{col_index}",
                    block_no=0,
                    line_no=row_index,
                    word_no=col_index,
                )
            )
    return words


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


def test_build_table_grid_emits_colspan_for_wide_header_cell() -> None:
    words = [
        Word(x0=10, y0=10, x1=210, y1=20, text="MergedHeader", block_no=0, line_no=0, word_no=0),
        Word(x0=310, y0=10, x1=350, y1=20, text="C", block_no=0, line_no=0, word_no=1),
        Word(x0=10, y0=30, x1=40, y1=40, text="A1", block_no=0, line_no=1, word_no=0),
        Word(x0=150, y0=30, x1=180, y1=40, text="B1", block_no=0, line_no=1, word_no=1),
        Word(x0=310, y0=30, x1=340, y1=40, text="C1", block_no=0, line_no=1, word_no=2),
        Word(x0=10, y0=50, x1=40, y1=60, text="A2", block_no=0, line_no=2, word_no=0),
        Word(x0=150, y0=50, x1=180, y1=60, text="B2", block_no=0, line_no=2, word_no=1),
        Word(x0=310, y0=50, x1=340, y1=60, text="C2", block_no=0, line_no=2, word_no=2),
    ]

    result = build_table_grid(words, column_count=3)
    assert result.ok, result.reason
    assert result.grid is not None
    assert result.grid.rows[0][0] == "MergedHeader"
    header_cells = [c for c in result.grid.cells if c.row == 0 and c.col == 0]
    assert header_cells
    assert header_cells[0].colspan >= 2


def test_build_table_grid_keeps_missing_middle_cell_as_empty() -> None:
    words = [
        Word(x0=10, y0=10, x1=40, y1=20, text="A", block_no=0, line_no=0, word_no=0),
        Word(x0=150, y0=10, x1=180, y1=20, text="B", block_no=0, line_no=0, word_no=1),
        Word(x0=310, y0=10, x1=340, y1=20, text="C", block_no=0, line_no=0, word_no=2),
        Word(x0=10, y0=30, x1=40, y1=40, text="D", block_no=0, line_no=1, word_no=0),
        Word(x0=310, y0=30, x1=340, y1=40, text="F", block_no=0, line_no=1, word_no=1),
    ]

    result = build_table_grid(words, column_count=3)
    assert result.ok, result.reason
    assert result.grid is not None
    assert result.grid.rows[0] == ("A", "B", "C")
    assert result.grid.rows[1] == ("D", "", "F")


def test_build_table_grid_detects_tight_five_columns() -> None:
    words = _make_tight_five_column_words(rows=4)

    result = build_table_grid(words, column_count=5)
    assert result.ok, result.reason
    assert result.grid is not None
    assert result.grid.rows[0] == ("R0C0", "R0C1", "R0C2", "R0C3", "R0C4")
    assert result.grid.collision_count == 0
    assert result.grid.group_count == 20


def test_build_table_grid_preserves_text_when_collapsing_to_two_columns() -> None:
    words = _make_tight_five_column_words(rows=4)

    result = build_table_grid(words, column_count=2)
    assert result.ok, result.reason
    assert result.grid is not None
    assert result.grid.collision_count > 0

    flattened = " ".join(" ".join(row) for row in result.grid.rows)
    for row_index in range(4):
        for col_index in range(5):
            assert f"R{row_index}C{col_index}" in flattened
    assert "/" in flattened

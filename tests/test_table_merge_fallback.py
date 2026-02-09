from __future__ import annotations

from ragprep.pdf_text import Word
from ragprep.table_merge import merge_markdown_tables_with_pymupdf_words


def test_table_merge_uses_aggressive_fallback_for_numeric_cell() -> None:
    words = [
        Word(x0=10, y0=10, x1=60, y1=20, text="Item", block_no=0, line_no=0, word_no=0),
        Word(x0=120, y0=10, x1=170, y1=20, text="Value", block_no=0, line_no=0, word_no=1),
        Word(x0=10, y0=25, x1=70, y1=35, text="Sales", block_no=0, line_no=1, word_no=0),
        Word(x0=120, y0=25, x1=160, y1=35, text="100", block_no=0, line_no=1, word_no=1),
    ]
    ocr_md = "\n".join(
        [
            "| Item | Value |",
            "|---|---|",
            "| Sales | 10O |",
        ]
    )

    merged, stats = merge_markdown_tables_with_pymupdf_words(ocr_md, words)
    assert stats.applied is True
    assert stats.changed_cells >= 1
    assert "10O" not in merged
    assert "| Sales | 100 |" in merged

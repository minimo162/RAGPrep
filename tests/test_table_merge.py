from __future__ import annotations

from ragprep.pdf_text import Word
from ragprep.table_merge import merge_markdown_tables_with_pymupdf_words


def _make_table_words() -> list[Word]:
    return [
        Word(x0=72, y0=70, x1=110, y1=82, text="都市", block_no=0, line_no=0, word_no=0),
        Word(x0=200, y0=70, x1=220, y1=82, text="数", block_no=0, line_no=0, word_no=1),
        Word(x0=320, y0=70, x1=360, y1=82, text="備考", block_no=0, line_no=0, word_no=2),
        Word(x0=72, y0=84, x1=110, y1=96, text="大阪", block_no=0, line_no=1, word_no=0),
        Word(x0=200, y0=84, x1=210, y1=96, text="1", block_no=0, line_no=1, word_no=1),
        Word(x0=320, y0=84, x1=340, y1=96, text="OK", block_no=0, line_no=1, word_no=2),
    ]


def test_merge_markdown_tables_with_pymupdf_words_corrects_single_cell() -> None:
    words = _make_table_words()
    ocr_md = "\n".join(
        [
            "| 都市 | 数 | 備考 |",
            "|---|---|---|",
            "| 大坂 | 1 | OK |",
        ]
    )

    merged, stats = merge_markdown_tables_with_pymupdf_words(ocr_md, words)
    assert stats.applied is True
    assert stats.changed_cells == 1
    assert stats.changed_chars >= 1
    assert "大阪" in merged
    assert "大坂" not in merged


def test_merge_markdown_tables_with_pymupdf_words_falls_back_on_row_mismatch() -> None:
    words = _make_table_words()
    ocr_md = "\n".join(
        [
            "| 都市 | 数 | 備考 |",
            "|---|---|---|",
            "| 大坂 | 1 | OK |",
            "| 東京 | 2 | NG |",
        ]
    )

    merged, stats = merge_markdown_tables_with_pymupdf_words(ocr_md, words)
    assert merged == ocr_md
    assert stats.applied is False
    assert stats.reason in {
        "row_count_mismatch",
        "grid_low_confidence",
        "grid_failed:insufficient_anchors",
    }

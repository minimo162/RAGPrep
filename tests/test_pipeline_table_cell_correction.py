from __future__ import annotations

from collections.abc import Iterator

import pytest

from ragprep.html_render import render_page_html
from ragprep.pdf_text import Word
from ragprep.pipeline import (
    _correct_table_blocks_locally_with_pymupdf,
    _correct_table_cell_with_pymupdf_text_candidates,
    pdf_to_html,
)
from ragprep.structure_ir import Page, Table
from ragprep.table_grid import TableCell


def test_pdf_to_html_corrects_html_table_headers_and_item_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_PROFILE", "balanced")

    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 1, iter(["P1"])

    words = [
        Word(x0=10, y0=10, x1=52, y1=20, text="Item", block_no=0, line_no=0, word_no=0),
        Word(x0=120, y0=10, x1=174, y1=20, text="Value", block_no=0, line_no=0, word_no=1),
        Word(x0=10, y0=24, x1=64, y1=34, text="Sales", block_no=0, line_no=1, word_no=0),
        Word(x0=120, y0=24, x1=152, y1=34, text="100", block_no=0, line_no=1, word_no=1),
    ]

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["Item Value\nSales 100"],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [words])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "\n".join(
            [
                "<table>",
                "<tr><th>Iten</th><th>Valne</th></tr>",
                "<tr><td>SaIes</td><td>10O</td></tr>",
                "</table>",
            ]
        ),
    )

    html = pdf_to_html(b"%PDF", full_document=False)

    assert "<th>Item</th>" in html
    assert "<th>Value</th>" in html
    assert "<td>Sales</td>" in html
    assert "<td>100</td>" in html
    assert "Iten" not in html
    assert "Valne" not in html
    assert "SaIes" not in html
    assert "10O" not in html


def test_correct_table_blocks_syncs_cells_with_updated_grid() -> None:
    original = "潜在在読読商戸1株当たり中間純利益"
    corrected = "潜在株式調整後1株当たり中間純利益"

    page = Page(
        page_number=1,
        blocks=(
            Table(
                text=f"1株当たり中間純利益\t{original}",
                grid=(("1株当たり中間純利益", original),),
                cells=(
                    TableCell(row=0, col=0, text="1株当たり中間純利益"),
                    TableCell(row=0, col=1, text=original),
                ),
            ),
        ),
    )

    updated = _correct_table_blocks_locally_with_pymupdf(
        page=page,
        pymupdf_text="潜在株式調整後 １株当たり中間純利益",
        pymupdf_words=[],
    )

    assert isinstance(updated.blocks[0], Table)
    table = updated.blocks[0]
    assert table.grid is not None
    assert table.grid[0][1] == corrected
    assert table.cells is not None
    assert any(cell.row == 0 and cell.col == 1 and cell.text == corrected for cell in table.cells)

    html = render_page_html(updated)
    assert corrected in html
    assert original not in html


def test_table_cell_correction_allows_artifact_fix_with_longer_reference_candidate() -> None:
    corrected = _correct_table_cell_with_pymupdf_text_candidates(
        source_cell="環境規制間述引当金",
        text_candidates=["環境規制関連引当金の増減額（△は減少）"],
    )
    assert corrected == "環境規制関連引当金"


def test_table_cell_correction_normalizes_known_header_term_without_reference() -> None:
    corrected = _correct_table_cell_with_pymupdf_text_candidates(
        source_cell="前四期発表予想比",
        text_candidates=[],
    )
    assert corrected == "前回発表予想比"

    corrected_short = _correct_table_cell_with_pymupdf_text_candidates(
        source_cell="前四発表予想比",
        text_candidates=[],
    )
    assert corrected_short == "前回発表予想比"


def test_correct_table_blocks_trims_fully_empty_trailing_column() -> None:
    page = Page(
        page_number=1,
        blocks=(
            Table(
                text="通期\t前期比\t\n売上高\t49,000\t",
                grid=(
                    ("通期", "前期比", ""),
                    ("売上高", "49,000", ""),
                ),
                cells=None,
            ),
        ),
    )

    updated = _correct_table_blocks_locally_with_pymupdf(
        page=page,
        pymupdf_text="通期 前期比\n売上高 49,000",
        pymupdf_words=[],
    )

    assert isinstance(updated.blocks[0], Table)
    table = updated.blocks[0]
    assert table.grid is not None
    assert len(table.grid[0]) == 2
    assert len(table.grid[1]) == 2


def test_correct_table_blocks_repairs_forecast_header_alignment_and_keeps_fx_rowspan() -> None:
    page = Page(
        page_number=1,
        blocks=(
            Table(
                text="",
                grid=(
                    ("通期", "前期比", "前回発表予想比", "", ""),
                    ("増減額", "増減率", "", "", ""),
                    ("売上高", "49,000", "△2.4%", "0", "0.0%"),
                    ("為替レート （円）", "ＵＳドル", "147", "△5", "+2"),
                    ("", "ユーロ", "171", "+7", "+2"),
                ),
                cells=(
                    TableCell(row=0, col=0, text="通期", rowspan=2),
                    TableCell(row=0, col=1, text="前期比", rowspan=2),
                    TableCell(row=0, col=2, text="前回発表予想比", colspan=2),
                    TableCell(row=0, col=4, text=""),
                    TableCell(row=1, col=0, text="増減額"),
                    TableCell(row=1, col=1, text="増減率"),
                    TableCell(row=1, col=2, text=""),
                    TableCell(row=3, col=0, text="為替レート （円）", rowspan=2),
                ),
            ),
        ),
    )

    updated = _correct_table_blocks_locally_with_pymupdf(
        page=page,
        pymupdf_text="通期 前期比 前回発表予想比 増減額 増減率 売上高 為替レート ＵＳドル ユーロ",
        pymupdf_words=[],
    )

    assert isinstance(updated.blocks[0], Table)
    table = updated.blocks[0]
    assert table.grid is not None
    assert table.grid[0] == ("", "通期", "前期比", "前回発表予想比", "")
    assert table.grid[1] == ("", "", "", "増減額", "増減率")
    assert table.cells is not None
    assert any(cell.row == 0 and cell.col == 0 and cell.rowspan == 2 for cell in table.cells)
    assert any(cell.row == 0 and cell.col == 3 and cell.colspan == 2 for cell in table.cells)
    assert any(
        cell.row == 3 and cell.col == 0 and cell.rowspan == 2 and cell.text == "為替レート （円）"
        for cell in table.cells
    )

    html = render_page_html(updated)
    assert '<th rowspan="2"></th>' in html
    assert '<th colspan="2">前回発表予想比</th>' in html
    assert '<td rowspan="2">為替レート （円）</td>' in html


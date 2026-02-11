from __future__ import annotations

from ragprep.html_render import render_page_html
from ragprep.ocr_html import page_from_ocr_markdown
from ragprep.structure_ir import Table


def test_page_from_ocr_markdown_preserves_rowspan_alignment() -> None:
    markdown = "\n".join(
        [
            "<table>",
            "<tr><th>項目</th><th>通期</th><th>前期比</th><th>前回発表予想比</th></tr>",
            "<tr><td rowspan=\"2\">為替レート （円）</td><td>ＵＳドル</td><td>147</td><td>△5</td></tr>",
            "<tr><td>ユーロ</td><td>171</td><td>+7</td></tr>",
            "</table>",
        ]
    )

    page = page_from_ocr_markdown(page_number=1, markdown=markdown)
    assert len(page.blocks) == 1
    assert isinstance(page.blocks[0], Table)

    block = page.blocks[0]
    assert block.grid is not None
    assert block.grid[0] == ("項目", "通期", "前期比", "前回発表予想比")
    assert block.grid[1] == ("為替レート （円）", "ＵＳドル", "147", "△5")
    assert block.grid[2] == ("", "ユーロ", "171", "+7")

    assert block.cells is not None
    assert any(
        cell.row == 1 and cell.col == 0 and cell.rowspan == 2 and cell.text == "為替レート （円）"
        for cell in block.cells
    )

    html = render_page_html(page)
    assert '<td rowspan="2">為替レート （円）</td>' in html
    assert "<td>ユーロ</td>" in html
    assert "<td>171</td>" in html

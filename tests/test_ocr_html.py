from __future__ import annotations

from ragprep.html_render import render_page_html
from ragprep.ocr_html import page_from_ocr_markdown
from ragprep.structure_ir import Paragraph, Table


def test_page_from_ocr_markdown_parses_html_table_block() -> None:
    markdown = "\n".join(
        [
            "<table>",
            "<thead>",
            "<tr><th></th><th>売上高</th><th>営業利益</th></tr>",
            "</thead>",
            "<tbody>",
            "<tr><td>2026年3月期中間期</td><td>2,238,463 △6.5</td><td>△53,879 -</td></tr>",
            "</tbody>",
            "</table>",
        ]
    )

    page = page_from_ocr_markdown(page_number=1, markdown=markdown)
    assert len(page.blocks) == 1
    block = page.blocks[0]
    assert isinstance(block, Table)
    assert block.grid is not None
    assert block.grid[0][1] == "売上高"
    assert block.grid[1][0] == "2026年3月期中間期"

    html = render_page_html(page)
    assert '<table data-kind="table">' in html
    assert "<thead>" in html
    assert "<th>売上高</th>" in html
    assert "<td>2,238,463 △6.5</td>" in html


def test_page_from_ocr_markdown_parses_escaped_html_table_block() -> None:
    markdown = (
        "&lt;table&gt;&lt;thead&gt;&lt;tr&gt;"
        "&lt;th&gt;項目&lt;/th&gt;&lt;th&gt;値&lt;/th&gt;"
        "&lt;/tr&gt;&lt;/thead&gt;&lt;tbody&gt;&lt;tr&gt;"
        "&lt;td&gt;売上高&lt;/td&gt;&lt;td&gt;2,238,463&lt;/td&gt;"
        "&lt;/tr&gt;&lt;/tbody&gt;&lt;/table&gt;"
    )

    page = page_from_ocr_markdown(page_number=1, markdown=markdown)
    assert len(page.blocks) == 1
    block = page.blocks[0]
    assert isinstance(block, Table)
    assert block.grid is not None
    assert block.grid[0] == ("項目", "値")
    assert block.grid[1] == ("売上高", "2,238,463")

    html = render_page_html(page)
    assert '<table data-kind="table">' in html
    assert "<th>項目</th>" in html
    assert "<td>2,238,463</td>" in html


def test_page_from_ocr_markdown_parses_incomplete_html_table_block() -> None:
    markdown = "\n".join(
        [
            "<table>",
            "<thead><tr><th>項目</th><th>値</th></tr></thead>",
            "<tbody>",
            "<tr><td>売上高</td><td>2,238,463</td></tr>",
            "<tr><td>営業利益</td><td>△53,879</td></tr>",
        ]
    )

    page = page_from_ocr_markdown(page_number=1, markdown=markdown)
    assert len(page.blocks) == 1
    block = page.blocks[0]
    assert isinstance(block, Table)
    assert block.grid is not None
    assert block.grid[0] == ("項目", "値")
    assert block.grid[1] == ("売上高", "2,238,463")
    assert block.grid[2] == ("営業利益", "△53,879")

    html = render_page_html(page)
    assert '<table data-kind="table">' in html
    assert "<th>項目</th>" in html
    assert "<td>△53,879</td>" in html


def test_page_from_ocr_markdown_keeps_paragraphs_around_html_table() -> None:
    markdown = "\n".join(
        [
            "前文です",
            "<table>",
            "<tr><th>A</th><th>B</th></tr>",
            "<tr><td>1</td><td>2</td></tr>",
            "</table>",
            "後文です",
        ]
    )

    page = page_from_ocr_markdown(page_number=1, markdown=markdown)
    assert len(page.blocks) == 3
    assert isinstance(page.blocks[0], Paragraph)
    assert isinstance(page.blocks[1], Table)
    assert isinstance(page.blocks[2], Paragraph)

    html = render_page_html(page)
    assert "<p>前文です</p>" in html
    assert "<p>後文です</p>" in html
    assert '<table data-kind="table">' in html


def test_page_from_ocr_markdown_drops_dangling_table_fragment_paragraph() -> None:
    markdown = "\n".join(
        [
            "&lt;table&gt;",
            "  &lt;thead&gt;",
            "    &lt;tr&gt;",
        ]
    )

    page = page_from_ocr_markdown(page_number=1, markdown=markdown)
    assert page.blocks == ()

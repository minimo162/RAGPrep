from __future__ import annotations

from ragprep.markdown_table import (
    TableAlignment,
    TableBlock,
    TextBlock,
    parse_markdown_blocks,
    render_markdown_blocks,
)


def test_parse_markdown_blocks_splits_text_and_table() -> None:
    src = "\n".join(
        [
            "intro",
            "",
            "| A | B |",
            "|---|---|",
            "| 1 | 2 |",
            "",
            "tail",
        ]
    )

    blocks = parse_markdown_blocks(src)
    assert len(blocks) == 3
    assert isinstance(blocks[0], TextBlock)
    assert isinstance(blocks[1], TableBlock)
    assert isinstance(blocks[2], TextBlock)

    table = blocks[1].table
    assert table.header == ("A", "B")
    assert table.align == (TableAlignment.none, TableAlignment.none)
    assert table.rows == (("1", "2"),)


def test_render_round_trip_preserves_table_structure() -> None:
    src = "\n".join(
        [
            "before",
            "| A | B |",
            "|:---|---:|",
            "| x | y |",
            "after",
        ]
    )

    blocks1 = parse_markdown_blocks(src)
    rendered = render_markdown_blocks(blocks1)
    blocks2 = parse_markdown_blocks(rendered)

    assert blocks2 == blocks1


def test_parse_supports_rows_without_edge_pipes() -> None:
    src = "\n".join(
        [
            "A | B",
            "---|---",
            "1 | 2",
        ]
    )

    blocks = parse_markdown_blocks(src)
    assert len(blocks) == 1
    assert isinstance(blocks[0], TableBlock)
    assert blocks[0].table.header == ("A", "B")
    assert blocks[0].table.rows == (("1", "2"),)


def test_parse_ignores_non_table_pipes() -> None:
    src = "\n".join(["this | is | not", "a table"])
    blocks = parse_markdown_blocks(src)
    assert blocks == [TextBlock(lines=("this | is | not", "a table"))]


def test_render_escapes_pipes_inside_cells() -> None:
    src = "\n".join(
        [
            "| col |",
            "|---|",
            r"| a\|b |",
        ]
    )

    blocks = parse_markdown_blocks(src)
    rendered = render_markdown_blocks(blocks)
    blocks2 = parse_markdown_blocks(rendered)
    assert blocks2 == blocks

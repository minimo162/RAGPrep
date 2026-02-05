from __future__ import annotations

import pytest

from ragprep.pdf_text import Span
from ragprep.structure_ir import BBox, Heading, LayoutElement, Paragraph, Table, build_page_blocks


def test_build_page_blocks_assigns_spans_to_layout_regions() -> None:
    spans = [
        Span(x0=10, y0=10, x1=20, y1=20, text="Hello"),
        Span(x0=30, y0=10, x1=40, y1=20, text="World"),
        Span(x0=10, y0=120, x1=20, y1=130, text="Body"),
    ]
    layout_elements = [
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.0, 1.0, 0.1), label="title", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.1, 1.0, 0.3), label="text", score=0.9),
    ]

    blocks = build_page_blocks(
        spans=spans,
        page_width=1000.0,
        page_height=1000.0,
        layout_elements=layout_elements,
    )

    assert len(blocks) == 2
    assert isinstance(blocks[0], Heading)
    assert blocks[0].text == "Hello World"
    assert isinstance(blocks[1], Paragraph)
    assert blocks[1].text == "Body"


def test_build_page_blocks_preserves_line_breaks_in_spans_join() -> None:
    spans = [
        Span(x0=10, y0=10, x1=20, y1=20, text="Line1"),
        Span(x0=10, y0=30, x1=20, y1=40, text="Line2"),
    ]
    layout_elements = [
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.0, 1.0, 1.0), label="text", score=0.9),
    ]

    blocks = build_page_blocks(
        spans=spans,
        page_width=1000.0,
        page_height=1000.0,
        layout_elements=layout_elements,
    )
    assert len(blocks) == 1
    assert isinstance(blocks[0], Paragraph)
    assert blocks[0].text == "Line1 Line2"


def test_build_page_blocks_orders_two_columns_left_then_right() -> None:
    spans = [
        Span(x0=50, y0=100, x1=60, y1=110, text="L-top"),
        Span(x0=50, y0=300, x1=60, y1=310, text="L-bottom"),
        Span(x0=600, y0=100, x1=610, y1=110, text="R-top"),
        Span(x0=600, y0=300, x1=610, y1=310, text="R-bottom"),
    ]
    layout_elements = [
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.05, 0.45, 0.20), label="text", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.55, 0.05, 1.0, 0.20), label="text", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.25, 0.45, 0.40), label="text", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.55, 0.25, 1.0, 0.40), label="text", score=0.9),
    ]

    blocks = build_page_blocks(
        spans=spans,
        page_width=1000.0,
        page_height=1000.0,
        layout_elements=layout_elements,
    )

    assert [b.text for b in blocks if isinstance(b, Paragraph)] == [
        "L-top",
        "L-bottom",
        "R-top",
        "R-bottom",
    ]


def test_build_page_blocks_orders_header_then_columns_via_xy_cut() -> None:
    spans = [
        Span(x0=10, y0=10, x1=80, y1=20, text="HEADER"),
        Span(x0=50, y0=200, x1=60, y1=210, text="L1"),
        Span(x0=600, y0=200, x1=610, y1=210, text="R1"),
    ]
    layout_elements = [
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.0, 1.0, 0.10), label="heading", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.20, 0.45, 0.95), label="text", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.55, 0.20, 1.0, 0.95), label="text", score=0.9),
    ]

    blocks = build_page_blocks(
        spans=spans,
        page_width=1000.0,
        page_height=1000.0,
        layout_elements=layout_elements,
    )

    assert isinstance(blocks[0], Heading)
    assert blocks[0].text == "HEADER"
    assert [b.text for b in blocks[1:] if isinstance(b, Paragraph)] == ["L1", "R1"]


def test_build_page_blocks_orders_three_columns_left_to_right() -> None:
    spans = [
        Span(x0=100, y0=200, x1=110, y1=210, text="C1"),
        Span(x0=500, y0=200, x1=510, y1=210, text="C2"),
        Span(x0=900, y0=200, x1=910, y1=210, text="C3"),
    ]
    layout_elements = [
        LayoutElement(page_index=0, bbox=BBox(0.00, 0.10, 0.33, 0.95), label="text", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.34, 0.10, 0.66, 0.95), label="text", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.67, 0.10, 1.00, 0.95), label="text", score=0.9),
    ]

    blocks = build_page_blocks(
        spans=spans,
        page_width=1000.0,
        page_height=1000.0,
        layout_elements=layout_elements,
    )

    assert [b.text for b in blocks if isinstance(b, Paragraph)] == ["C1", "C2", "C3"]


def test_build_page_blocks_uses_topo_order_when_no_clear_gaps() -> None:
    spans = [
        Span(x0=50, y0=200, x1=60, y1=210, text="LEFT"),
        Span(x0=600, y0=200, x1=610, y1=210, text="RIGHT"),
    ]
    # No clear whitespace gap split (overlapping x ranges slightly), but same-row band should
    # still order left-to-right via topo precedence.
    layout_elements = [
        LayoutElement(page_index=0, bbox=BBox(0.00, 0.10, 0.55, 0.30), label="text", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.45, 0.10, 1.00, 0.30), label="text", score=0.9),
    ]

    blocks = build_page_blocks(
        spans=spans,
        page_width=1000.0,
        page_height=1000.0,
        layout_elements=layout_elements,
    )

    assert [b.text for b in blocks if isinstance(b, Paragraph)] == ["LEFT", "RIGHT"]


def test_build_page_blocks_sets_heading_level_from_span_sizes() -> None:
    spans = [
        Span(x0=10, y0=10, x1=20, y1=20, text="BigTitle", size=20.0),
        Span(x0=10, y0=200, x1=20, y1=210, text="Body1", size=10.0),
        Span(x0=30, y0=200, x1=40, y1=210, text="Body2", size=10.0),
        Span(x0=50, y0=200, x1=60, y1=210, text="Body3", size=10.0),
        Span(x0=70, y0=200, x1=80, y1=210, text="Body4", size=10.0),
        Span(x0=90, y0=200, x1=100, y1=210, text="Body5", size=10.0),
    ]
    layout_elements = [
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.0, 1.0, 0.15), label="heading", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.15, 1.0, 1.0), label="text", score=0.9),
    ]

    blocks = build_page_blocks(
        spans=spans,
        page_width=1000.0,
        page_height=1000.0,
        layout_elements=layout_elements,
    )

    assert isinstance(blocks[0], Heading)
    assert blocks[0].level == 1
    assert blocks[0].text == "BigTitle"


def test_build_page_blocks_promotes_large_font_short_text_to_heading() -> None:
    spans = [
        Span(x0=10, y0=10, x1=20, y1=20, text="Section", size=18.0),
        Span(x0=10, y0=200, x1=20, y1=210, text="Body", size=10.0),
        Span(x0=30, y0=200, x1=40, y1=210, text="Text", size=10.0),
        Span(x0=50, y0=200, x1=60, y1=210, text="More", size=10.0),
    ]
    layout_elements = [
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.0, 1.0, 0.15), label="text", score=0.9),
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.15, 1.0, 1.0), label="text", score=0.9),
    ]

    blocks = build_page_blocks(
        spans=spans,
        page_width=1000.0,
        page_height=1000.0,
        layout_elements=layout_elements,
    )

    assert isinstance(blocks[0], Heading)
    assert blocks[0].level == 2
    assert blocks[0].text == "Section"


def test_build_page_blocks_extracts_best_effort_table_grid() -> None:
    spans = [
        Span(x0=10, y0=10, x1=50, y1=20, text="A"),
        Span(x0=200, y0=10, x1=240, y1=20, text="B"),
        Span(x0=400, y0=10, x1=440, y1=20, text="C"),
        Span(x0=10, y0=30, x1=50, y1=40, text="D"),
        Span(x0=200, y0=30, x1=240, y1=40, text="E"),
        Span(x0=400, y0=30, x1=440, y1=40, text="F"),
    ]
    layout_elements = [
        LayoutElement(page_index=0, bbox=BBox(0.0, 0.0, 1.0, 1.0), label="table", score=0.9),
    ]

    blocks = build_page_blocks(
        spans=spans,
        page_width=1000.0,
        page_height=1000.0,
        layout_elements=layout_elements,
    )

    assert len(blocks) == 1
    assert isinstance(blocks[0], Table)
    assert blocks[0].grid == (("A", "B", "C"), ("D", "E", "F"))


def test_build_page_blocks_rejects_invalid_page_size() -> None:
    with pytest.raises(ValueError, match="page_width/page_height"):
        build_page_blocks(
            spans=[],
            page_width=0.0,
            page_height=1.0,
            layout_elements=[],
        )

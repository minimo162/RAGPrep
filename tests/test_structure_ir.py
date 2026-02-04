from __future__ import annotations

import pytest

from ragprep.pdf_text import Span
from ragprep.structure_ir import BBox, Heading, LayoutElement, Paragraph, build_page_blocks


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


def test_build_page_blocks_rejects_invalid_page_size() -> None:
    with pytest.raises(ValueError, match="page_width/page_height"):
        build_page_blocks(
            spans=[],
            page_width=0.0,
            page_height=1.0,
            layout_elements=[],
        )

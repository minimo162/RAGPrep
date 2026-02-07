from __future__ import annotations

from ragprep.layout.fast_layout import infer_fast_layout_elements
from ragprep.pdf_text import Span


def test_infer_fast_layout_elements_detects_title_text_and_table() -> None:
    spans = [
        Span(x0=100, y0=20, x1=360, y1=60, text="Document Title", size=24),
        Span(x0=80, y0=160, x1=200, y1=186, text="This", size=12),
        Span(x0=206, y0=160, x1=300, y1=186, text="is", size=12),
        Span(x0=306, y0=160, x1=460, y1=186, text="paragraph", size=12),
        Span(x0=80, y0=300, x1=160, y1=326, text="A1", size=12),
        Span(x0=320, y0=300, x1=400, y1=326, text="B1", size=12),
        Span(x0=560, y0=300, x1=640, y1=326, text="C1", size=12),
    ]

    elements = infer_fast_layout_elements(spans, page_width=1000, page_height=1400)
    labels = [str(e["label"]) for e in elements]

    assert "title" in labels
    assert "text" in labels
    assert "table" in labels


def test_infer_fast_layout_elements_returns_empty_for_no_text() -> None:
    spans = [
        Span(x0=0, y0=0, x1=10, y1=10, text="  "),
    ]
    assert infer_fast_layout_elements(spans, page_width=100, page_height=100) == []


def test_infer_fast_layout_elements_requires_positive_page_size() -> None:
    spans = [Span(x0=0, y0=0, x1=10, y1=10, text="x")]

    try:
        infer_fast_layout_elements(spans, page_width=0, page_height=100)
    except ValueError as exc:
        assert "page_width/page_height must be > 0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

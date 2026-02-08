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
        Span(x0=80, y0=334, x1=160, y1=360, text="A2", size=12),
        Span(x0=320, y0=334, x1=400, y1=360, text="B2", size=12),
        Span(x0=560, y0=334, x1=640, y1=360, text="C2", size=12),
    ]

    elements = infer_fast_layout_elements(spans, page_width=1000, page_height=1400)
    labels = [str(e["label"]) for e in elements]

    assert "title" in labels
    assert "text" in labels
    assert "table" in labels


def test_infer_fast_layout_elements_splits_two_columns_without_table() -> None:
    spans = [
        Span(x0=60, y0=200, x1=360, y1=226, text="Left column first line", size=11),
        Span(x0=60, y0=230, x1=360, y1=256, text="Left column second line", size=11),
        Span(x0=540, y0=200, x1=900, y1=226, text="Right column first line", size=11),
        Span(x0=540, y0=230, x1=900, y1=256, text="Right column second line", size=11),
    ]
    elements = infer_fast_layout_elements(spans, page_width=1000, page_height=1400)
    labels = [str(e["label"]) for e in elements]
    assert "table" not in labels

    text_boxes: list[tuple[float, float, float, float]] = []
    for element in elements:
        if str(element.get("label")) != "text":
            continue
        bbox = element.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        values: list[float] = []
        for value in bbox:
            if not isinstance(value, (int, float)):
                values = []
                break
            values.append(float(value))
        if len(values) != 4:
            continue
        text_boxes.append((values[0], values[1], values[2], values[3]))
    assert len(text_boxes) >= 2
    assert any(float(b[0]) < 300 for b in text_boxes)
    assert any(float(b[0]) > 400 for b in text_boxes)


def test_infer_fast_layout_elements_does_not_mark_single_sparse_row_as_table() -> None:
    spans = [
        Span(x0=90, y0=120, x1=180, y1=146, text="A", size=12),
        Span(x0=350, y0=120, x1=430, y1=146, text="B", size=12),
        Span(x0=620, y0=120, x1=700, y1=146, text="C", size=12),
        Span(x0=90, y0=190, x1=560, y1=216, text="This is normal sentence content.", size=12),
    ]
    elements = infer_fast_layout_elements(spans, page_width=1000, page_height=1400)
    labels = [str(e["label"]) for e in elements]
    assert "table" not in labels


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

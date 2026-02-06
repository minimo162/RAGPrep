from __future__ import annotations

from ragprep.html_render import render_document_html, wrap_html_document
from ragprep.structure_ir import Document, Heading, Page, Paragraph, Table, TableCell, Unknown


def test_render_document_html_escapes_text_and_wraps_pages() -> None:
    doc = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(
                    Heading(level=1, text="<Title>"),
                    Paragraph(text='A & B "C"\nD'),
                    Unknown(text="X>Y"),
                ),
            ),
        )
    )

    html = render_document_html(doc)
    assert '<section data-page="1">' in html
    assert "&lt;Title&gt;" in html
    assert "A &amp; B &quot;C&quot;" in html
    assert "<br" in html
    assert "X&gt;Y" in html


def test_wrap_html_document_produces_full_document() -> None:
    fragment = "<div>ok</div>"
    html = wrap_html_document(fragment, title='T & "Q"')
    assert html.startswith("<!doctype html>")
    assert "<meta charset" in html
    assert "<div>ok</div>" in html
    assert "<title>T &amp; &quot;Q&quot;</title>" in html


def test_render_document_html_renders_structured_table_grid() -> None:
    doc = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(
                    Table(text="fallback", grid=(("A", "B"), ("C", "D"))),
                ),
            ),
        )
    )

    html = render_document_html(doc)
    assert '<table data-kind="table">' in html
    assert "<td>A</td>" in html
    assert "<td>D</td>" in html


def test_render_document_html_renders_table_colspan_when_cells_present() -> None:
    doc = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(
                    Table(
                        text="fallback",
                        grid=(("Header", "", "C"), ("A", "B", "C1")),
                        cells=(
                            TableCell(row=0, col=0, text="Header", colspan=2, rowspan=1),
                            TableCell(row=0, col=2, text="C", colspan=1, rowspan=1),
                            TableCell(row=1, col=0, text="A", colspan=1, rowspan=1),
                            TableCell(row=1, col=1, text="B", colspan=1, rowspan=1),
                            TableCell(row=1, col=2, text="C1", colspan=1, rowspan=1),
                        ),
                    ),
                ),
            ),
        )
    )

    html = render_document_html(doc)
    assert '<td colspan="2">Header</td>' in html


from __future__ import annotations

from html import escape

from ragprep.structure_ir import (
    Block,
    Document,
    Figure,
    Heading,
    Page,
    Paragraph,
    Table,
    Unknown,
)


def render_document_html(document: Document) -> str:
    """
    Render a Document IR to a safe HTML fragment.

    Output is deterministic and escapes all user/model-provided text.
    """

    parts: list[str] = ['<article class="ragprep-document">']
    for page in document.pages:
        parts.append(_render_page(page))
    parts.append("</article>")
    return "\n".join(parts)


def render_page_html(page: Page) -> str:
    """
    Render a single Page IR to HTML.

    Returns a `<section data-page="...">…</section>` fragment.
    """

    return _render_page(page)


def wrap_html_document(fragment_html: str, *, title: str = "RAGPrep") -> str:
    """
    Wrap a fragment in a minimal standalone HTML document.
    """

    safe_title = escape(title, quote=True)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "  <head>",
            '    <meta charset="utf-8" />',
            '    <meta name="viewport" content="width=device-width, initial-scale=1" />',
            f"    <title>{safe_title}</title>",
            "  </head>",
            "  <body>",
            fragment_html,
            "  </body>",
            "</html>",
        ]
    )


def _render_page(page: Page) -> str:
    parts: list[str] = [f'<section data-page="{int(page.page_number)}">']
    for block in page.blocks:
        parts.append(_render_block(block))
    parts.append("</section>")
    return "\n".join(parts)


def _render_block(block: Block) -> str:
    if isinstance(block, Heading):
        level = max(1, min(6, int(block.level)))
        text = _escape_with_breaks(block.text)
        return f"<h{level}>{text}</h{level}>"
    if isinstance(block, Paragraph):
        text = _escape_with_breaks(block.text)
        return f"<p>{text}</p>"
    if isinstance(block, Table):
        cells = getattr(block, "cells", None)
        if block.grid:
            if cells:
                return _render_table_cells(block.grid, cells)
            return _render_table_grid(block.grid)
        text = _escape_with_breaks(block.text)
        return f'<pre data-kind="table">{text}</pre>'
    if isinstance(block, Figure):
        alt = _escape_with_breaks(block.alt)
        return f"<figure><figcaption>{alt}</figcaption></figure>"
    if isinstance(block, Unknown):
        text = _escape_with_breaks(block.text)
        return f"<p>{text}</p>"
    raise TypeError(f"Unsupported block type: {type(block)!r}")


def _escape_with_breaks(text: str) -> str:
    escaped = escape(text, quote=True)
    return escaped.replace("\n", "<br />\n")


def _render_table_grid(grid: tuple[tuple[str, ...], ...]) -> str:
    rows = list(grid)
    if not rows:
        return '<pre data-kind="table"></pre>'

    col_count = max((len(r) for r in rows), default=0)
    if col_count <= 0:
        return '<pre data-kind="table"></pre>'

    parts: list[str] = ['<table data-kind="table"><tbody>']
    for r in rows:
        parts.append("<tr>")
        padded = tuple(r) + ("",) * max(0, col_count - len(r))
        for cell in padded:
            parts.append(f"<td>{_escape_with_breaks(str(cell))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _render_table_cells(
    grid: tuple[tuple[str, ...], ...],
    cells: tuple[object, ...],
) -> str:
    rows = list(grid)
    if not rows:
        return '<pre data-kind="table"></pre>'

    row_count = len(rows)
    col_count = max((len(r) for r in rows), default=0)
    if col_count <= 0:
        return '<pre data-kind="table"></pre>'

    starts: dict[tuple[int, int], object] = {}
    for cell in cells:
        r = int(getattr(cell, "row", -1))
        c = int(getattr(cell, "col", -1))
        if r < 0 or c < 0 or r >= row_count or c >= col_count:
            continue
        starts[(r, c)] = cell

    parts: list[str] = ['<table data-kind="table"><tbody>']
    skip_by_row: dict[int, set[int]] = {}
    for r in range(row_count):
        parts.append("<tr>")
        skip_cols: set[int] = set()
        c = 0
        while c < col_count:
            if c in skip_cols or c in skip_by_row.get(r, set()):
                c += 1
                continue

            start = starts.get((r, c))
            if start is None:
                value = rows[r][c] if c < len(rows[r]) else ""
                parts.append(f"<td>{_escape_with_breaks(str(value))}</td>")
                c += 1
                continue

            colspan = max(1, int(getattr(start, "colspan", 1)))
            rowspan = max(1, int(getattr(start, "rowspan", 1)))
            attrs: list[str] = []
            if colspan > 1:
                attrs.append(f' colspan="{colspan}"')
            if rowspan > 1:
                attrs.append(f' rowspan="{rowspan}"')
            attr_text = "".join(attrs)

            value = str(getattr(start, "text", ""))
            if not value and c < len(rows[r]):
                value = str(rows[r][c])
            parts.append(f"<td{attr_text}>{_escape_with_breaks(value)}</td>")

            for covered_col in range(c + 1, min(col_count, c + colspan)):
                skip_cols.add(covered_col)
            if rowspan > 1:
                max_row = min(row_count, r + rowspan)
                max_col = min(col_count, c + colspan)
                for covered_row in range(r + 1, max_row):
                    row_skip = skip_by_row.setdefault(covered_row, set())
                    for covered_col in range(c, max_col):
                        row_skip.add(covered_col)
            c += max(1, colspan)
        parts.append("</tr>")

    parts.append("</tbody></table>")
    return "\n".join(parts)


from __future__ import annotations

import re
from html import unescape as html_unescape
from html.parser import HTMLParser

from ragprep.markdown_table import TableBlock, TextBlock, parse_markdown_blocks, render_pipe_table
from ragprep.structure_ir import Block, Document, Heading, Page, Paragraph, Table
from ragprep.table_grid import TableCell

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_TABLE_OPEN_RE = re.compile(r"<\s*table\b", re.IGNORECASE)
_TABLE_CLOSE_RE = re.compile(r"</\s*table\s*>", re.IGNORECASE)
_ESCAPED_TABLE_OPEN_RE = re.compile(r"&lt;\s*table\b", re.IGNORECASE)
_ESCAPED_TABLE_CLOSE_RE = re.compile(r"&lt;\s*/\s*table\s*&gt;", re.IGNORECASE)
_TABLE_FRAGMENT_LINE_RE = re.compile(
    r"^\s*</?\s*(?:table|thead|tbody|tr|th|td)\b[^>]*>\s*$",
    re.IGNORECASE,
)


def page_from_ocr_markdown(*, page_number: int, markdown: str) -> Page:
    blocks = parse_markdown_blocks(markdown)
    out: list[Block] = []
    for block in blocks:
        if isinstance(block, TableBlock):
            table = block.table
            grid = (table.header,) + table.rows
            out.append(
                Table(
                    text="\n".join(render_pipe_table(table)),
                    grid=grid,
                    cells=None,
                )
            )
            continue

        if isinstance(block, TextBlock):
            out.extend(_blocks_from_text_lines(block.lines))
            continue

    return Page(page_number=page_number, blocks=tuple(out))


def document_from_ocr_markdown_pages(page_markdowns: list[str]) -> Document:
    pages = [
        page_from_ocr_markdown(page_number=idx + 1, markdown=text)
        for idx, text in enumerate(page_markdowns)
    ]
    return Document(pages=tuple(pages))


def _blocks_from_text_lines(lines: tuple[str, ...]) -> list[Block]:
    blocks: list[Block] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        text = "\n".join(line.rstrip() for line in paragraph_lines).strip()
        paragraph_lines = []
        if text:
            if _is_dangling_table_markup_paragraph(text):
                return
            blocks.append(Paragraph(text=text))

    index = 0
    while index < len(lines):
        line = lines[index]
        html_table = _consume_html_table(lines, start_index=index)
        if html_table is not None:
            next_index, table = html_table
            if table is not None:
                flush_paragraph()
                blocks.append(table)
            else:
                paragraph_lines.extend(lines[index:next_index])
            index = next_index
            continue

        heading = _parse_heading(line)
        if heading is not None:
            flush_paragraph()
            blocks.append(heading)
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
            index += 1
            continue

        paragraph_lines.append(line)
        index += 1

    flush_paragraph()
    return blocks


def _parse_heading(line: str) -> Heading | None:
    match = _HEADING_RE.match(line)
    if match is None:
        return None
    level = len(match.group(1))
    text = match.group(2).strip()
    if not text:
        return None
    return Heading(level=level, text=text)


def _consume_html_table(
    lines: tuple[str, ...],
    *,
    start_index: int,
) -> tuple[int, Table | None] | None:
    if start_index < 0 or start_index >= len(lines):
        return None

    line = lines[start_index]
    if not _has_table_open_marker(line):
        return None

    collected: list[str] = [line]
    next_index = start_index + 1
    has_close = _has_table_close_marker(line)
    while not has_close and next_index < len(lines):
        current = lines[next_index]
        collected.append(current)
        has_close = _has_table_close_marker(current)
        next_index += 1

    snippet_raw = "\n".join(collected)
    snippet = _decode_html_table_markup(snippet_raw)
    table = _table_from_html_snippet(snippet)
    return next_index, table


class _SimpleHtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._table_depth = 0
        self._in_row = False
        self._in_cell = False
        self._cell_parts: list[str] = []
        self._cell_rowspan = 1
        self._cell_colspan = 1
        self._current_row: list[tuple[str, int, int]] = []
        self.rows: list[tuple[tuple[str, int, int], ...]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized == "table":
            self._table_depth += 1
            return
        if self._table_depth <= 0:
            return
        if normalized == "tr":
            if self._in_cell:
                self._close_cell()
            if self._in_row:
                self._close_row()
            self._in_row = True
            self._current_row = []
            return
        if normalized in {"td", "th"}:
            if not self._in_row:
                self._in_row = True
                self._current_row = []
            if self._in_cell:
                self._close_cell()
            attrs_dict = {str(k).lower(): ("" if v is None else str(v)) for k, v in attrs}
            self._cell_rowspan = _parse_positive_int(attrs_dict.get("rowspan"))
            self._cell_colspan = _parse_positive_int(attrs_dict.get("colspan"))
            self._in_cell = True
            self._cell_parts = []
            return
        if normalized == "br" and self._in_cell:
            self._cell_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized == "table":
            if self._table_depth <= 0:
                return
            if self._in_cell:
                self._close_cell()
            if self._in_row:
                self._close_row()
            self._table_depth -= 1
            return
        if self._table_depth <= 0:
            return
        if normalized in {"td", "th"}:
            if self._in_cell:
                self._close_cell()
            return
        if normalized == "tr" and self._in_row:
            if self._in_cell:
                self._close_cell()
            self._close_row()

    def handle_data(self, data: str) -> None:
        if self._table_depth > 0 and self._in_cell:
            self._cell_parts.append(data)

    def _close_cell(self) -> None:
        raw = "".join(self._cell_parts)
        normalized = _normalize_table_cell_text(raw)
        self._current_row.append((normalized, self._cell_rowspan, self._cell_colspan))
        self._cell_parts = []
        self._cell_rowspan = 1
        self._cell_colspan = 1
        self._in_cell = False

    def _close_row(self) -> None:
        if self._current_row and any(
            cell_text.strip() or rowspan > 1 or colspan > 1
            for cell_text, rowspan, colspan in self._current_row
        ):
            self.rows.append(tuple(self._current_row))
        self._current_row = []
        self._in_row = False

    def finalize(self) -> None:
        if self._in_cell:
            self._close_cell()
        if self._in_row:
            self._close_row()
        self._table_depth = 0


def _table_from_html_snippet(snippet: str) -> Table | None:
    parser = _SimpleHtmlTableParser()
    try:
        parser.feed(snippet)
        parser.close()
    except Exception:
        return None
    parser.finalize()
    rows = [tuple(cell for cell in row) for row in parser.rows if row]
    if not rows:
        return None
    expanded = _expand_html_table_rows_with_spans(rows)
    if expanded is None:
        return None
    padded, cells = expanded
    text = "\n".join("\t".join(cell for cell in row) for row in padded)
    return Table(text=text, grid=padded, cells=cells if cells else None)


def _parse_positive_int(raw_value: str | None) -> int:
    if raw_value is None:
        return 1
    try:
        parsed = int(str(raw_value).strip())
    except Exception:
        return 1
    return max(1, parsed)


def _expand_html_table_rows_with_spans(
    rows: list[tuple[tuple[str, int, int], ...]],
) -> tuple[tuple[tuple[str, ...], ...], tuple[TableCell, ...]] | None:
    if not rows:
        return None

    anchors: list[tuple[int, int, str, int, int]] = []
    pending_rowspan_by_col: dict[int, int] = {}
    max_col_count = 0

    for row_index, row in enumerate(rows):
        col_index = 0
        for text, raw_rowspan, raw_colspan in row:
            while pending_rowspan_by_col.get(col_index, 0) > 0:
                col_index += 1

            rowspan = max(1, int(raw_rowspan))
            colspan = max(1, int(raw_colspan))
            anchors.append((row_index, col_index, text, rowspan, colspan))

            if rowspan > 1:
                for covered_col in range(col_index, col_index + colspan):
                    pending_rowspan_by_col[covered_col] = max(
                        pending_rowspan_by_col.get(covered_col, 0),
                        rowspan,
                    )

            col_index += colspan

        if pending_rowspan_by_col:
            row_max = max(col_index, max(pending_rowspan_by_col) + 1)
        else:
            row_max = col_index
        max_col_count = max(max_col_count, row_max)

        for covered_col in list(pending_rowspan_by_col):
            remaining = pending_rowspan_by_col[covered_col] - 1
            if remaining <= 0:
                del pending_rowspan_by_col[covered_col]
            else:
                pending_rowspan_by_col[covered_col] = remaining

    row_count = len(rows)
    for row_index, col_index, _text, rowspan, colspan in anchors:
        row_count = max(row_count, row_index + rowspan)
        max_col_count = max(max_col_count, col_index + colspan)

    if row_count <= 0 or max_col_count <= 0:
        return None

    matrix: list[list[str]] = [["" for _ in range(max_col_count)] for _ in range(row_count)]
    cells: list[TableCell] = []
    for row_index, col_index, text, rowspan, colspan in anchors:
        if 0 <= row_index < row_count and 0 <= col_index < max_col_count:
            matrix[row_index][col_index] = text
            cells.append(
                TableCell(
                    row=row_index,
                    col=col_index,
                    text=text,
                    colspan=colspan,
                    rowspan=rowspan,
                )
            )

    padded = tuple(tuple(row) for row in matrix)
    return padded, tuple(cells)


def _normalize_table_cell_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(part.split()) for part in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _has_table_open_marker(text: str) -> bool:
    return bool(_TABLE_OPEN_RE.search(text) or _ESCAPED_TABLE_OPEN_RE.search(text))


def _has_table_close_marker(text: str) -> bool:
    return bool(_TABLE_CLOSE_RE.search(text) or _ESCAPED_TABLE_CLOSE_RE.search(text))


def _decode_html_table_markup(text: str) -> str:
    decoded = text
    for _ in range(2):
        if "<table" in decoded.lower() and "</table>" in decoded.lower():
            break
        next_decoded = html_unescape(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return decoded


def _is_dangling_table_markup_paragraph(text: str) -> bool:
    if not text or not _has_table_open_marker(text):
        return False
    normalized = _decode_html_table_markup(text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return False
    tag_only_count = sum(1 for line in lines if _TABLE_FRAGMENT_LINE_RE.fullmatch(line))
    return tag_only_count == len(lines)

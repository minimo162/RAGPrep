from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

_CODE_FENCE_PREFIX = "```"

_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


class TableAlignment(str, Enum):
    none = "none"
    left = "left"
    right = "right"
    center_ = "center"


@dataclass(frozen=True)
class MarkdownTable:
    header: tuple[str, ...]
    align: tuple[TableAlignment, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class TextBlock:
    lines: tuple[str, ...]


@dataclass(frozen=True)
class TableBlock:
    table: MarkdownTable


Block: TypeAlias = TextBlock | TableBlock


def parse_markdown_blocks(markdown: str) -> list[Block]:
    """
    Split OCR-produced Markdown/text into blocks of:
    - plain text (line-based)
    - pipe tables (header row + separator row + optional body rows)

    Notes:
    - We intentionally only support a conservative subset of pipe tables.
      Unsupported formats are left as plain text (safe fallback).
    - Tables inside fenced code blocks are ignored (treated as text).
    """

    lines = _normalize_newlines(markdown).split("\n")
    blocks: list[Block] = []
    text_buf: list[str] = []

    def flush_text() -> None:
        nonlocal text_buf
        if text_buf:
            blocks.append(TextBlock(lines=tuple(text_buf)))
            text_buf = []

    in_code_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith(_CODE_FENCE_PREFIX):
            in_code_fence = not in_code_fence
            text_buf.append(line)
            i += 1
            continue
        if in_code_fence:
            text_buf.append(line)
            i += 1
            continue

        header = _split_pipe_row(line)
        if header is None or i + 1 >= len(lines):
            text_buf.append(line)
            i += 1
            continue

        align = _parse_separator_row(lines[i + 1], expected_cols=len(header))
        if align is None:
            text_buf.append(line)
            i += 1
            continue

        body_rows: list[tuple[str, ...]] = []
        j = i + 2
        while j < len(lines):
            if lines[j].lstrip().startswith(_CODE_FENCE_PREFIX):
                break
            row = _split_pipe_row(lines[j])
            if row is None:
                break
            if len(row) > len(header):
                break
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            body_rows.append(tuple(row))
            j += 1

        flush_text()
        blocks.append(
            TableBlock(
                table=MarkdownTable(
                    header=tuple(header),
                    align=tuple(align),
                    rows=tuple(body_rows),
                )
            )
        )
        i = j

    flush_text()
    return blocks


def render_markdown_blocks(blocks: list[Block]) -> str:
    lines: list[str] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            lines.extend(block.lines)
            continue
        lines.extend(render_pipe_table(block.table))
    return "\n".join(lines)


def render_pipe_table(table: MarkdownTable) -> list[str]:
    if not table.header:
        return []
    if len(table.align) != len(table.header):
        raise ValueError("table.align length must match table.header length")

    def render_row(cells: tuple[str, ...]) -> str:
        escaped = tuple(_escape_cell(c) for c in cells)
        return "| " + " | ".join(escaped) + " |"

    header = table.header
    separator_cells = tuple(_alignment_to_separator(a) for a in table.align)

    out: list[str] = [render_row(header), render_row(separator_cells)]
    for row in table.rows:
        if len(row) != len(header):
            raise ValueError("row length must match header length")
        out.append(render_row(row))
    return out


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_pipe_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None

    raw_cells = _split_unescaped_pipes(stripped)
    if stripped.startswith("|") and raw_cells and raw_cells[0] == "":
        raw_cells = raw_cells[1:]
    if stripped.endswith("|") and raw_cells and raw_cells[-1] == "":
        raw_cells = raw_cells[:-1]

    cells = [_unescape_pipes(c.strip()) for c in raw_cells]
    if len(cells) < 1:
        return None
    return cells


def _split_unescaped_pipes(text: str) -> list[str]:
    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in text:
        if ch == "|" and not escaped:
            cells.append("".join(buf))
            buf = []
            escaped = False
            continue
        buf.append(ch)
        if ch == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    cells.append("".join(buf))
    return cells


def _unescape_pipes(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == "|":
            out.append("|")
            i += 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _parse_separator_row(line: str, *, expected_cols: int) -> list[TableAlignment] | None:
    cells = _split_pipe_row(line)
    if cells is None or len(cells) != expected_cols:
        return None

    out: list[TableAlignment] = []
    for cell in cells:
        compact = cell.strip().replace(" ", "")
        if not _SEPARATOR_CELL_RE.fullmatch(compact):
            return None
        left = compact.startswith(":")
        right = compact.endswith(":")
        if left and right:
            out.append(TableAlignment.center_)
        elif left:
            out.append(TableAlignment.left)
        elif right:
            out.append(TableAlignment.right)
        else:
            out.append(TableAlignment.none)
    return out


def _alignment_to_separator(align: TableAlignment) -> str:
    if align == TableAlignment.left:
        return ":---"
    if align == TableAlignment.right:
        return "---:"
    if align == TableAlignment.center_:
        return ":---:"
    return "---"


def _escape_cell(text: str) -> str:
    # Markdown pipe table cells cannot contain literal newlines or unescaped pipes.
    normalized = _normalize_newlines(text).replace("\n", "<br>")
    return _escape_unescaped_pipes(normalized)


def _escape_unescaped_pipes(text: str) -> str:
    out: list[str] = []
    backslashes = 0
    for ch in text:
        if ch == "\\":
            backslashes += 1
            out.append(ch)
            continue
        if ch == "|":
            if backslashes % 2 == 0:
                out.append("\\")
            out.append("|")
            backslashes = 0
            continue
        out.append(ch)
        backslashes = 0
    return "".join(out)

from __future__ import annotations

import re

from ragprep.markdown_table import TableBlock, TextBlock, parse_markdown_blocks, render_pipe_table
from ragprep.structure_ir import Block, Document, Heading, Page, Paragraph, Table

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


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
            blocks.append(Paragraph(text=text))

    for line in lines:
        heading = _parse_heading(line)
        if heading is not None:
            flush_paragraph()
            blocks.append(heading)
            continue

        if not line.strip():
            flush_paragraph()
            continue

        paragraph_lines.append(line)

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

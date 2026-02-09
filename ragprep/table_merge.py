from __future__ import annotations

from dataclasses import dataclass

from ragprep.markdown_table import (
    MarkdownTable,
    TableBlock,
    parse_markdown_blocks,
    render_markdown_blocks,
)
from ragprep.pdf_text import Word, normalize_extracted_text
from ragprep.table_grid import build_table_grid
from ragprep.text_merge import merge_ocr_with_pymupdf


@dataclass(frozen=True)
class TableMergeStats:
    applied: bool
    changed_cells: int
    changed_chars: int
    confidence: float | None = None
    reason: str | None = None
    samples: tuple[str, ...] = ()


def merge_markdown_tables_with_pymupdf_words(
    ocr_markdown: str,
    pymupdf_words: list[Word],
    *,
    min_grid_confidence: float = 0.30,
) -> tuple[str, TableMergeStats]:
    """
    Merge OCR-produced Markdown pipe tables with PyMuPDF words.

    - The OCR table structure is treated as SSOT.
    - We attempt conservative, per-cell text corrections using `merge_ocr_with_pymupdf`.
    - On any uncertainty, return the original OCR Markdown unchanged (safe fallback).
    """

    blocks = parse_markdown_blocks(ocr_markdown)
    table_block_indices = [i for i, b in enumerate(blocks) if isinstance(b, TableBlock)]
    if not table_block_indices:
        return ocr_markdown, TableMergeStats(
            applied=False, changed_cells=0, changed_chars=0, reason="no_table"
        )
    if len(table_block_indices) != 1:
        return ocr_markdown, TableMergeStats(
            applied=False,
            changed_cells=0,
            changed_chars=0,
            reason="multiple_tables_unsupported",
        )

    table_index = table_block_indices[0]
    table_block = blocks[table_index]
    assert isinstance(table_block, TableBlock)
    table = table_block.table
    column_count = len(table.header)
    if column_count <= 0:
        return ocr_markdown, TableMergeStats(
            applied=False, changed_cells=0, changed_chars=0, reason="invalid_table_header"
        )

    grid_result = build_table_grid(pymupdf_words, column_count=column_count)
    if not grid_result.ok or grid_result.grid is None:
        return ocr_markdown, TableMergeStats(
            applied=False,
            changed_cells=0,
            changed_chars=0,
            confidence=grid_result.confidence,
            reason=f"grid_failed:{grid_result.reason}",
        )
    if grid_result.confidence < float(min_grid_confidence):
        return ocr_markdown, TableMergeStats(
            applied=False,
            changed_cells=0,
            changed_chars=0,
            confidence=grid_result.confidence,
            reason="grid_low_confidence",
        )

    expected_rows = 1 + len(table.rows)
    if len(grid_result.grid.rows) != expected_rows:
        return ocr_markdown, TableMergeStats(
            applied=False,
            changed_cells=0,
            changed_chars=0,
            confidence=grid_result.confidence,
            reason="row_count_mismatch",
        )

    ocr_rows = [list(table.header)] + [list(r) for r in table.rows]
    grid_rows = list(grid_result.grid.rows)

    changed_cells = 0
    changed_chars = 0
    samples: list[str] = []

    for r_i, (ocr_row, grid_row) in enumerate(zip(ocr_rows, grid_rows, strict=False)):
        for c_i in range(column_count):
            ocr_cell_raw = ocr_row[c_i]
            pym_cell_raw = grid_row[c_i]

            ocr_cell = normalize_extracted_text(ocr_cell_raw).strip()
            pym_cell = normalize_extracted_text(pym_cell_raw).strip()
            if not ocr_cell or not pym_cell:
                continue

            merged_cell, merge_stats = merge_ocr_with_pymupdf(ocr_cell, pym_cell)
            if merged_cell == ocr_cell:
                merged_cell_fallback, merge_stats_fallback = merge_ocr_with_pymupdf(
                    ocr_cell,
                    pym_cell,
                    policy="aggressive",
                    max_changed_ratio=0.45,
                )
                if merged_cell_fallback != ocr_cell:
                    merged_cell = merged_cell_fallback
                    merge_stats = merge_stats_fallback
            if merged_cell == ocr_cell:
                continue
            if not _is_safe_cell_merge(ocr_cell, merged_cell):
                continue

            ocr_row[c_i] = merged_cell
            changed_cells += 1
            changed_chars += int(merge_stats.changed_char_count)
            if len(samples) < 3:
                samples.append(f"r{r_i}c{c_i}: {ocr_cell[:40]} -> {merged_cell[:40]}")

    if changed_cells == 0:
        return ocr_markdown, TableMergeStats(
            applied=False,
            changed_cells=0,
            changed_chars=0,
            confidence=grid_result.confidence,
            reason="no_changes",
        )

    new_table = MarkdownTable(
        header=tuple(ocr_rows[0]),
        align=table.align,
        rows=tuple(tuple(r) for r in ocr_rows[1:]),
    )
    blocks[table_index] = TableBlock(table=new_table)

    merged_markdown = render_markdown_blocks(blocks)
    return merged_markdown, TableMergeStats(
        applied=True,
        changed_cells=changed_cells,
        changed_chars=changed_chars,
        confidence=grid_result.confidence,
        reason=None,
        samples=tuple(samples),
    )


def _is_safe_cell_merge(ocr_cell: str, merged_cell: str) -> bool:
    if "\n" in merged_cell or "\r" in merged_cell:
        return False
    if "|" in merged_cell and "|" not in ocr_cell:
        return False
    return True

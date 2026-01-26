from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from ragprep import pymupdf4llm_markdown
from ragprep.pipeline import (
    PdfToMarkdownProgress,
    ProgressPhase,
    pdf_to_markdown,
)


def _make_pdf_bytes(page_count: int) -> bytes:
    import fitz

    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 72), f"Hello {i + 1}")
    return cast(bytes, doc.tobytes())


def _make_pdf_bytes_single_column_with_sidebar() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    y0 = 72
    line_h = 14
    for i in range(10):
        page.insert_text((72, y0 + i * line_h), f"BODY{i + 1}")

    rect_top = y0 + 1 * line_h - 2  # place NOTE_A between BODY3 and BODY4 (by y)
    rect = fitz.Rect(320, rect_top, 520, rect_top + 60)

    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(fill=(0.9, 0.9, 0.9), color=(0.9, 0.9, 0.9))
    shape.commit()

    page.insert_textbox(rect, "NOTE_A\nNOTE_B", fontsize=12)

    return cast(bytes, doc.tobytes())


def _make_pdf_bytes_two_columns_with_gap() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Left column (two separate blocks to force a non-trivial reading order)
    y = 72
    for line in ["L_TOP_1", "L_TOP_2"]:
        page.insert_text((72, y), line)
        y += 14

    y = 300
    for line in ["L_BOT_1", "L_BOT_2"]:
        page.insert_text((72, y), line)
        y += 14

    # Right column
    y = 72
    for line in ["R_TOP_1", "R_TOP_2"]:
        page.insert_text((320, y), line)
        y += 14

    y = 300
    for line in ["R_BOT_1", "R_BOT_2"]:
        page.insert_text((320, y), line)
        y += 14

    return cast(bytes, doc.tobytes())


def _make_pdf_bytes_single_column_with_multiple_sidebars() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    y0 = 72
    line_h = 14
    for i in range(20):
        page.insert_text((72, y0 + i * line_h), f"BODY{i + 1}")

    def add_sidebar(line_index: int, text: str) -> None:
        rect_top = y0 + (line_index - 1) * line_h - 2
        rect = fitz.Rect(280, rect_top, 520, rect_top + 60)

        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(fill=(0.9, 0.9, 0.9), color=(0.9, 0.9, 0.9))
        shape.commit()

        page.insert_textbox(rect, text, fontsize=12)

    add_sidebar(5, "NOTE_A\nDETAIL_ALPHA")
    add_sidebar(10, "NOTE_B\nDETAIL_BRAVO")
    add_sidebar(15, "NOTE_C\nDETAIL_CHARLIE")

    return cast(bytes, doc.tobytes())


def _make_pdf_bytes_two_columns_with_overlapping_callout() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    for y, line in [(72, "L_TOP_1"), (86, "L_TOP_2"), (300, "L_BOT_1"), (314, "L_BOT_2")]:
        page.insert_text((72, y), line)

    for y, line in [(72, "R_TOP_1"), (86, "R_TOP_2"), (300, "R_BOT_1"), (314, "R_BOT_2")]:
        page.insert_text((320, y), line)

    page.insert_text((190, 160), "NOTE_OVERLAP")

    return cast(bytes, doc.tobytes())


def test_pdf_to_markdown_returns_markdown_and_normalizes_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {"n": 0}

    def fake_to_markdown(_doc: object) -> str:
        calls["n"] += 1
        return "page1\r\n\r\npage2\r"

    monkeypatch.setattr(pymupdf4llm_markdown.pymupdf4llm, "to_markdown", fake_to_markdown)

    assert pdf_to_markdown(_make_pdf_bytes(page_count=2)) == "page1\n\npage2"
    assert calls["n"] == 1


def test_pdf_to_markdown_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pdf_to_markdown(b"")


def test_pdf_to_markdown_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pymupdf4llm_markdown.pymupdf4llm, "to_markdown", lambda _doc: "ok")

    updates: list[PdfToMarkdownProgress] = []

    def on_progress(update: PdfToMarkdownProgress) -> None:
        updates.append(update)

    assert pdf_to_markdown(_make_pdf_bytes(page_count=3), on_progress=on_progress) == "ok"
    assert [(u.phase, u.current, u.total) for u in updates] == [
        (ProgressPhase.rendering, 0, 3),
        (ProgressPhase.done, 3, 3),
    ]


def test_pdf_to_markdown_writes_document_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pymupdf4llm_markdown.pymupdf4llm, "to_markdown", lambda _doc: "hello")

    out_dir = tmp_path / "artifacts"
    result = pdf_to_markdown(_make_pdf_bytes(page_count=1), page_output_dir=out_dir)
    assert result == "hello"

    artifact = out_dir / "document.md"
    assert artifact.exists()
    assert artifact.read_text(encoding="utf-8") == "hello\n"


def test_pdf_to_markdown_invalid_pdf_raises_invalid_pdf_data() -> None:
    with pytest.raises(ValueError, match="Invalid PDF data"):
        pdf_to_markdown(b"not a pdf")


def test_pdf_to_markdown_inserts_sidebar_near_reading_position() -> None:
    markdown = pdf_to_markdown(_make_pdf_bytes_single_column_with_sidebar())

    body3 = markdown.find("BODY3")
    note_a = markdown.find("NOTE_A")
    body4 = markdown.find("BODY4")

    assert body3 != -1
    assert note_a != -1
    assert body4 != -1
    assert body3 < note_a < body4


def test_pdf_to_markdown_orders_two_columns_column_major() -> None:
    markdown = pdf_to_markdown(_make_pdf_bytes_two_columns_with_gap())

    l_bot_2 = markdown.find("L_BOT_2")
    r_top_1 = markdown.find("R_TOP_1")

    assert l_bot_2 != -1
    assert r_top_1 != -1
    assert l_bot_2 < r_top_1


def test_pdf_to_markdown_orders_multiple_sidebars_by_y_and_keeps_near_body() -> None:
    markdown = pdf_to_markdown(_make_pdf_bytes_single_column_with_multiple_sidebars())

    body4 = markdown.find("BODY4")
    body9 = markdown.find("BODY9")
    body15 = markdown.find("BODY15")
    body20 = markdown.find("BODY20")

    note_a = markdown.find("NOTE_A")
    note_b = markdown.find("NOTE_B")
    note_c = markdown.find("NOTE_C")

    assert body4 != -1
    assert body9 != -1
    assert body15 != -1
    assert body20 != -1
    assert note_a != -1
    assert note_b != -1
    assert note_c != -1

    assert markdown.count("NOTE_A") == 1
    assert markdown.count("NOTE_B") == 1
    assert markdown.count("NOTE_C") == 1

    assert note_a < note_b < note_c
    assert body4 < note_a < body9
    assert body9 < note_b < body15
    assert body15 < note_c < body20


def test_pdf_to_markdown_deduplicates_overlapping_callout_in_columns() -> None:
    markdown = pdf_to_markdown(_make_pdf_bytes_two_columns_with_overlapping_callout())

    assert markdown.count("NOTE_OVERLAP") == 1

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

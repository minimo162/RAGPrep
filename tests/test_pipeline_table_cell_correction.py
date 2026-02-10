from __future__ import annotations

from collections.abc import Iterator

import pytest

from ragprep.pdf_text import Word
from ragprep.pipeline import pdf_to_html


def test_pdf_to_html_corrects_html_table_headers_and_item_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_PROFILE", "balanced")

    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 1, iter(["P1"])

    words = [
        Word(x0=10, y0=10, x1=52, y1=20, text="Item", block_no=0, line_no=0, word_no=0),
        Word(x0=120, y0=10, x1=174, y1=20, text="Value", block_no=0, line_no=0, word_no=1),
        Word(x0=10, y0=24, x1=64, y1=34, text="Sales", block_no=0, line_no=1, word_no=0),
        Word(x0=120, y0=24, x1=152, y1=34, text="100", block_no=0, line_no=1, word_no=1),
    ]

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["Item Value\nSales 100"],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [words])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "\n".join(
            [
                "<table>",
                "<tr><th>Iten</th><th>Valne</th></tr>",
                "<tr><td>SaIes</td><td>10O</td></tr>",
                "</table>",
            ]
        ),
    )

    html = pdf_to_html(b"%PDF", full_document=False)

    assert "<th>Item</th>" in html
    assert "<th>Value</th>" in html
    assert "<td>Sales</td>" in html
    assert "<td>100</td>" in html
    assert "Iten" not in html
    assert "Valne" not in html
    assert "SaIes" not in html
    assert "10O" not in html

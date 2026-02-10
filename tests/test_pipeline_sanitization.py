from __future__ import annotations

from collections.abc import Iterator

from ragprep.pipeline import pdf_to_html


def test_pdf_to_html_sanitizes_llm_artifacts(monkeypatch) -> None:
    ocr_output = "\n".join(
        [
            "# TITLE",
            "![image](image_1.png)",
            "Note: The image contains a logo and placeholder URL.",
            "*Note: The image shows a dashboard screenshot.*",
            "*This transcription is an OCR transcription of the visible text and image.*",
            "**Company Name**",
            "---",
            "&lt;div style=\"display:flex\"&gt;",
            "Body text",
            "&lt;/div&gt;",
            "&lt;div style=\"border: 1px solid black; padding:",
            "",
            "---",
            "**kubell**",
            "---",
            "**kubell**",
            "---",
            "**kubell**",
            "---",
            "**kubell**",
            "---",
            "**kubell**",
            "",
            "&lt;table&gt;&lt;tr&gt;&lt;th&gt;A&lt;/th&gt;&lt;th&gt;B&lt;/th&gt;&lt;/tr&gt;&lt;/table&gt;",
        ]
    )

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

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: [ocr_output])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: ocr_output,
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "<h1>TITLE</h1>" in html
    assert "<p>Company Name</p>" in html
    assert "<p>Body text</p>" in html
    assert "![image]" not in html
    assert "Note: The image contains" not in html
    assert "The image shows a dashboard screenshot" not in html
    assert "This transcription is an OCR transcription" not in html
    assert "&lt;div" not in html
    assert "<p>---</p>" not in html
    assert html.count("<p>kubell</p>") <= 2
    assert '<table data-kind="table">' in html

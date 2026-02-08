from __future__ import annotations

from io import BytesIO
from typing import cast

import pytest
from PIL import Image

from ragprep.pdf_text import (
    PageKind,
    analyze_pdf_pages,
    extract_pymupdf_page_sizes,
    extract_pymupdf_page_spans,
    extract_pymupdf_page_texts,
    extract_pymupdf_page_words,
    normalize_extracted_text,
    tokenize_by_char_class,
)


def _make_pdf_bytes_with_text(lines_per_page: list[list[str]]) -> bytes:
    import fitz

    doc = fitz.open()
    for lines in lines_per_page:
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line)
            y += 14
    return cast(bytes, doc.tobytes())


def _make_pdf_bytes_with_fullpage_image() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()

    image = Image.new("RGB", (800, 800), color=(255, 255, 255))
    bio = BytesIO()
    image.save(bio, format="PNG")
    png_bytes = bio.getvalue()

    page.insert_image(page.rect, stream=png_bytes)
    return cast(bytes, doc.tobytes())


def _make_pdf_bytes_with_table_like_text(rows: int = 20) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()

    y = 72
    for i in range(rows):
        page.insert_text((72, y), f"Item{i}")
        page.insert_text((200, y), f"{i}")
        page.insert_text((320, y), f"{i * 10}")
        y += 12

    return cast(bytes, doc.tobytes())


def test_extract_pymupdf_page_texts_returns_text_per_page() -> None:
    pdf_bytes = _make_pdf_bytes_with_text([["Hello 1"], ["Hello 2"]])
    pages = extract_pymupdf_page_texts(pdf_bytes)
    assert len(pages) == 2
    assert "Hello 1" in pages[0]
    assert "Hello 2" in pages[1]


def test_extract_pymupdf_page_spans_returns_spans_per_page() -> None:
    pdf_bytes = _make_pdf_bytes_with_text([["Hello 1", "World 1"], ["Hello 2"]])
    pages = extract_pymupdf_page_spans(pdf_bytes)
    assert len(pages) == 2

    page1 = pages[0]
    page2 = pages[1]
    assert any("Hello 1" in s.text for s in page1)
    assert any("World 1" in s.text for s in page1)
    assert any("Hello 2" in s.text for s in page2)

    for span in page1 + page2:
        assert span.x0 < span.x1
        assert span.y0 < span.y1
        assert span.text.strip() == span.text


def test_extract_pymupdf_page_words_returns_words_per_page() -> None:
    pdf_bytes = _make_pdf_bytes_with_text([["Hello 1", "World 1"], ["Hello 2"]])
    pages = extract_pymupdf_page_words(pdf_bytes)
    assert len(pages) == 2
    assert any("Hello" in w.text for w in pages[0])
    assert any("World" in w.text for w in pages[0])
    assert any("Hello" in w.text for w in pages[1])


def test_extract_pymupdf_page_sizes_returns_page_sizes() -> None:
    pdf_bytes = _make_pdf_bytes_with_text([["Hello 1"], ["Hello 2"]])
    sizes = extract_pymupdf_page_sizes(pdf_bytes)
    assert len(sizes) == 2
    for w, h in sizes:
        assert w > 0
        assert h > 0


def test_normalize_extracted_text_normalizes_newlines_and_controls() -> None:
    raw = "a\r\nb\r\x00c\x07d\u00a0\u3000"
    assert normalize_extracted_text(raw) == "a\nb\ncd"


def test_tokenize_by_char_class_splits_japanese_and_alnum() -> None:
    text = "東京都ABC123ひらがなカタカナ!!"
    assert tokenize_by_char_class(text) == ["東京都", "ABC", "123", "ひらがな", "カタカナ", "!!"]


def test_analyze_pdf_pages_classifies_text_page() -> None:
    pdf_bytes = _make_pdf_bytes_with_text([["Hello world"]])
    pages = analyze_pdf_pages(pdf_bytes)
    assert len(pages) == 1
    page = pages[0]
    assert page.page_kind == PageKind.text
    assert page.has_text_layer is True
    assert page.image_count == 0
    assert page.image_area_ratio == pytest.approx(0.0)
    assert page.table_likelihood == pytest.approx(0.0)


def test_analyze_pdf_pages_classifies_image_page() -> None:
    pdf_bytes = _make_pdf_bytes_with_fullpage_image()
    pages = analyze_pdf_pages(pdf_bytes)
    assert len(pages) == 1
    page = pages[0]
    assert page.page_kind == PageKind.image
    assert page.has_text_layer is False
    assert page.image_count >= 1
    assert page.image_area_ratio >= 0.80


def test_analyze_pdf_pages_estimates_table_likelihood_for_aligned_columns() -> None:
    pdf_bytes = _make_pdf_bytes_with_table_like_text(rows=20)
    pages = analyze_pdf_pages(pdf_bytes)
    assert len(pages) == 1
    page = pages[0]
    assert page.table_likelihood >= 0.70
    assert page.page_kind == PageKind.table

from __future__ import annotations

import json
from typing import cast

import pytest

from ragprep.pdf_text import normalize_extracted_text
from ragprep.pipeline import (
    PdfToMarkdownProgress,
    ProgressPhase,
    merge_ocr_with_pymupdf,
    pdf_to_markdown,
)

_LONG_JP_TEXT = "abcdefghijklmnopqrstuvwxyz" * 8


def _make_pdf_bytes(page_texts: list[str]) -> bytes:
    import fitz

    doc = fitz.open()
    for i, text in enumerate(page_texts):
        page = doc.new_page()
        page.insert_text((72, 72), text or f"Hello {i + 1}")
    return cast(bytes, doc.tobytes())


def test_pdf_to_markdown_concatenates_pages_and_normalizes_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = [object(), object()]
    monkeypatch.setattr(
        "ragprep.pipeline.iter_pdf_images",
        lambda _pdf_bytes: (len(images), iter(images)),
    )

    calls: dict[str, int] = {"n": 0}

    def fake_ocr_image(_image: object) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "page1\r\n"
        if calls["n"] == 2:
            return "page2\r"
        raise AssertionError("Unexpected extra page")

    monkeypatch.setattr("ragprep.pipeline.ocr_image", fake_ocr_image)

    assert pdf_to_markdown(b"pdf") == "page1\n\npage2"


def test_pdf_to_markdown_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pdf_to_markdown(b"")


def test_pdf_to_markdown_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    images = [object(), object(), object()]
    monkeypatch.setattr(
        "ragprep.pipeline.iter_pdf_images",
        lambda _pdf_bytes: (len(images), iter(images)),
    )
    monkeypatch.setattr("ragprep.pipeline.ocr_image", lambda _image: "ok")

    updates: list[PdfToMarkdownProgress] = []

    def on_progress(update: PdfToMarkdownProgress) -> None:
        updates.append(update)

    assert pdf_to_markdown(b"pdf", on_progress=on_progress) == "ok\n\nok\n\nok"

    assert [(u.phase, u.current, u.total) for u in updates] == [
        (ProgressPhase.rendering, 0, 0),
        (ProgressPhase.rendering, 3, 3),
        (ProgressPhase.ocr, 0, 3),
        (ProgressPhase.ocr, 1, 3),
        (ProgressPhase.ocr, 2, 3),
        (ProgressPhase.ocr, 3, 3),
        (ProgressPhase.done, 3, 3),
    ]


def test_merge_ocr_with_pymupdf_corrects_single_kanji() -> None:
    ocr_text = "私は大坂に行く"
    pymupdf_text = "私は大阪に行く"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text)

    assert merged == "私は大阪に行く"
    assert stats.changed_char_count == 1
    assert stats.applied_block_count == 1


def _expected_pymupdf_page_text(pdf_bytes: bytes, page_index: int) -> str:
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    with doc:
        page = doc.load_page(page_index)
        return normalize_extracted_text(str(page.get_text("text") or "")).strip()


def test_pdf_to_markdown_text_first_skips_rendering_and_ocr_for_high_quality_text_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_bytes = _make_pdf_bytes([_LONG_JP_TEXT])
    expected = _expected_pymupdf_page_text(pdf_bytes, 0)

    def fail_render(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Rendering should be skipped")

    monkeypatch.setattr("ragprep.pipeline._render_page_to_image", fail_render)
    monkeypatch.setattr(
        "ragprep.pipeline.ocr_image",
        lambda _image: (_ for _ in ()).throw(AssertionError("OCR should be skipped")),
    )

    assert pdf_to_markdown(pdf_bytes) == expected


def test_pdf_to_markdown_text_first_renders_only_ocr_pages_and_writes_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path_factory.mktemp("artifacts")
    pdf_bytes = _make_pdf_bytes([_LONG_JP_TEXT, "短い"])
    expected_pymupdf_1 = _expected_pymupdf_page_text(pdf_bytes, 0)
    expected_pymupdf_2 = _expected_pymupdf_page_text(pdf_bytes, 1)

    render_calls = {"n": 0}

    def fake_render(*_args: object, **_kwargs: object) -> object:
        render_calls["n"] += 1
        return object()

    monkeypatch.setattr("ragprep.pipeline._render_page_to_image", fake_render)

    ocr_calls = {"n": 0}

    def fake_ocr(_image: object) -> str:
        ocr_calls["n"] += 1
        return "OCR2"

    monkeypatch.setattr("ragprep.pipeline.ocr_image", fake_ocr)

    result = pdf_to_markdown(pdf_bytes, page_output_dir=out_dir)
    assert result == f"{expected_pymupdf_1}\n\nOCR2"
    assert render_calls["n"] == 1
    assert ocr_calls["n"] == 1

    p1_ocr = out_dir / "page-0001.ocr.md"
    p1_pym = out_dir / "page-0001.pymupdf.md"
    p1_merged = out_dir / "page-0001.merged.md"
    p1_meta = out_dir / "page-0001.meta.json"

    p2_ocr = out_dir / "page-0002.ocr.md"
    p2_pym = out_dir / "page-0002.pymupdf.md"
    p2_merged = out_dir / "page-0002.merged.md"
    p2_meta = out_dir / "page-0002.meta.json"

    assert p1_ocr.exists() and p1_pym.exists() and p1_merged.exists() and p1_meta.exists()
    assert p2_ocr.exists() and p2_pym.exists() and p2_merged.exists() and p2_meta.exists()

    assert p1_ocr.read_text(encoding="utf-8").strip() == ""
    assert p1_pym.read_text(encoding="utf-8").strip() == expected_pymupdf_1
    assert p1_merged.read_text(encoding="utf-8").strip() == expected_pymupdf_1

    meta1 = json.loads(p1_meta.read_text(encoding="utf-8"))
    assert meta1["page_number"] == 1
    assert meta1["page_kind"] == "text"
    assert meta1["analysis_available"] is True
    assert meta1["selected_source"] == "pymupdf"
    assert meta1["ocr_required"] is False
    assert isinstance(meta1["ocr_reason"], str) and meta1["ocr_reason"]

    assert p2_ocr.read_text(encoding="utf-8").strip() == "OCR2"
    assert p2_pym.read_text(encoding="utf-8").strip() == expected_pymupdf_2
    assert p2_merged.read_text(encoding="utf-8").strip() == "OCR2"

    meta2 = json.loads(p2_meta.read_text(encoding="utf-8"))
    assert meta2["page_number"] == 2
    assert meta2["page_kind"] == "text"
    assert meta2["analysis_available"] is True
    assert meta2["selected_source"] == "ocr"
    assert meta2["ocr_required"] is True
    assert isinstance(meta2["ocr_reason"], str) and meta2["ocr_reason"]

    merged_pages = [
        p1_merged.read_text(encoding="utf-8").strip(),
        p2_merged.read_text(encoding="utf-8").strip(),
    ]
    combined = "\n\n".join(p for p in merged_pages if p.strip()).strip()
    assert combined == result

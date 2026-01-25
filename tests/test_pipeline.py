from __future__ import annotations

import json
from typing import cast

import pytest

from ragprep.pdf_text import PageAnalysis, PageKind, TextQualityScore
from ragprep.pipeline import (
    PdfToMarkdownProgress,
    ProgressPhase,
    merge_ocr_with_pymupdf,
    pdf_to_markdown,
)


def _make_pdf_bytes(page_count: int) -> bytes:
    import fitz

    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 72), f"Hello {i + 1}")
    return cast(bytes, doc.tobytes())


def test_pdf_to_markdown_concatenates_pages_and_normalizes_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_bytes = _make_pdf_bytes(page_count=2)

    monkeypatch.setattr(
        "ragprep.pipeline.analyze_pdf_pages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Exception),
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

    assert pdf_to_markdown(pdf_bytes) == "page1\n\npage2"


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


def _analysis(
    *,
    page_number: int = 1,
    page_kind: PageKind,
    score: float,
    table_likelihood: float = 0.0,
    normalized_text: str = "pymupdf",
) -> PageAnalysis:
    return PageAnalysis(
        page_number=page_number,
        raw_text=normalized_text,
        normalized_text=normalized_text,
        tokens=("t",),
        has_text_layer=True,
        text_quality=TextQualityScore(
            char_count=10,
            visible_char_count=10,
            visible_ratio=1.0,
            replacement_char_ratio=0.0,
            symbol_ratio=0.0,
            longest_repeat_ratio=0.0,
            score=score,
        ),
        image_count=0,
        image_area_ratio=0.0,
        table_likelihood=table_likelihood,
        page_kind=page_kind,
    )


def test_pdf_to_markdown_skips_ocr_on_high_quality_text_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ragprep.pipeline.analyze_pdf_pages",
        lambda *_args, **_kwargs: [
            _analysis(page_kind=PageKind.text, score=1.0, normalized_text="PYM")
        ],
    )
    monkeypatch.setattr(
        "ragprep.pipeline.iter_pdf_images", lambda _pdf_bytes: (1, iter([object()]))
    )
    monkeypatch.setattr(
        "ragprep.pipeline.ocr_image",
        lambda _image: (_ for _ in ()).throw(AssertionError("OCR should be skipped")),
    )

    assert pdf_to_markdown(b"pdf") == "PYM"


def test_pdf_to_markdown_runs_ocr_when_pymupdf_is_low_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ragprep.pipeline.analyze_pdf_pages",
        lambda *_args, **_kwargs: [
            _analysis(page_kind=PageKind.text, score=0.0, normalized_text="PYM")
        ],
    )
    monkeypatch.setattr(
        "ragprep.pipeline.iter_pdf_images", lambda _pdf_bytes: (1, iter([object()]))
    )

    calls = {"n": 0}

    def fake_ocr(_image: object) -> str:
        calls["n"] += 1
        return "OCR"

    monkeypatch.setattr("ragprep.pipeline.ocr_image", fake_ocr)

    assert pdf_to_markdown(b"pdf") == "OCR"
    assert calls["n"] == 1


def test_pdf_to_markdown_requires_ocr_for_table_and_ambiguous_table_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ragprep.pipeline.analyze_pdf_pages",
        lambda *_args, **_kwargs: [
            _analysis(
                page_kind=PageKind.table, score=1.0, normalized_text="PYM", table_likelihood=1.0
            )
        ],
    )
    monkeypatch.setattr(
        "ragprep.pipeline.iter_pdf_images", lambda _pdf_bytes: (1, iter([object()]))
    )

    calls = {"n": 0}

    def fake_ocr(_image: object) -> str:
        calls["n"] += 1
        return "OCR"

    monkeypatch.setattr("ragprep.pipeline.ocr_image", fake_ocr)
    assert pdf_to_markdown(b"pdf") == "OCR"
    assert calls["n"] == 1

    monkeypatch.setattr(
        "ragprep.pipeline.analyze_pdf_pages",
        lambda *_args, **_kwargs: [
            _analysis(
                page_kind=PageKind.text, score=1.0, normalized_text="PYM", table_likelihood=0.5
            )
        ],
    )
    calls["n"] = 0
    assert pdf_to_markdown(b"pdf") == "OCR"
    assert calls["n"] == 1


def test_pdf_to_markdown_writes_page_artifacts_and_meta(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path_factory.mktemp("artifacts")

    analyses = [
        _analysis(page_number=1, page_kind=PageKind.text, score=1.0, normalized_text="PYM1"),
        _analysis(
            page_number=2,
            page_kind=PageKind.table,
            score=1.0,
            normalized_text="PYM2",
            table_likelihood=1.0,
        ),
    ]

    monkeypatch.setattr("ragprep.pipeline.analyze_pdf_pages", lambda *_a, **_k: analyses)
    monkeypatch.setattr(
        "ragprep.pipeline.iter_pdf_images",
        lambda _pdf_bytes: (2, iter([object(), object()])),
    )

    calls = {"n": 0}

    def fake_ocr(_image: object) -> str:
        calls["n"] += 1
        return "OCR2"

    monkeypatch.setattr("ragprep.pipeline.ocr_image", fake_ocr)

    result = pdf_to_markdown(b"pdf", page_output_dir=out_dir)
    assert result == "PYM1\n\nOCR2"
    assert calls["n"] == 1

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
    assert p1_pym.read_text(encoding="utf-8").strip() == "PYM1"
    assert p1_merged.read_text(encoding="utf-8").strip() == "PYM1"

    meta1 = json.loads(p1_meta.read_text(encoding="utf-8"))
    assert meta1["page_number"] == 1
    assert meta1["page_kind"] == "text"
    assert meta1["selected_source"] == "pymupdf"
    assert meta1["ocr_required"] is False
    assert isinstance(meta1["ocr_reason"], str) and meta1["ocr_reason"]

    assert p2_ocr.read_text(encoding="utf-8").strip() == "OCR2"
    assert p2_pym.read_text(encoding="utf-8").strip() == "PYM2"
    assert p2_merged.read_text(encoding="utf-8").strip() == "OCR2"

    meta2 = json.loads(p2_meta.read_text(encoding="utf-8"))
    assert meta2["page_number"] == 2
    assert meta2["page_kind"] == "table"
    assert meta2["selected_source"] == "ocr"
    assert meta2["ocr_required"] is True
    assert isinstance(meta2["ocr_reason"], str) and meta2["ocr_reason"]

    merged_pages = [
        p1_merged.read_text(encoding="utf-8").strip(),
        p2_merged.read_text(encoding="utf-8").strip(),
    ]
    combined = "\n\n".join(p for p in merged_pages if p.strip()).strip()
    assert combined == result

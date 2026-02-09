from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from ragprep.pdf_text import Word
from ragprep.pipeline import PdfToHtmlProgress, ProgressPhase, pdf_to_html


def test_pdf_to_html_reports_progress_and_renders_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 2, iter(["PAGE1", "PAGE2"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["# Heading\n\nParagraph", "| A | B |\n|---|---|\n| 1 | 2 |"],
    )

    def _fake_ocr(image_b64: str, *, settings: object) -> str:
        _ = settings
        if image_b64 == "PAGE1":
            return "# Heading\n\nParagraph"
        return "| A | B |\n|---|---|\n| 1 | 2 |"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    updates: list[PdfToHtmlProgress] = []
    pages: list[int] = []

    html = pdf_to_html(
        b"%PDF",
        full_document=False,
        on_progress=updates.append,
        on_page=lambda page_index, _html: pages.append(page_index),
    )

    assert "<h1>Heading</h1>" in html
    assert "<p>Paragraph</p>" in html
    assert '<table data-kind="table">' in html
    assert pages == [1, 2]
    assert [(u.phase, u.current, u.total) for u in updates] == [
        (ProgressPhase.rendering, 0, 2),
        (ProgressPhase.rendering, 1, 2),
        (ProgressPhase.rendering, 2, 2),
        (ProgressPhase.done, 2, 2),
    ]


def test_pdf_to_html_keeps_on_page_order_under_parallel_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_PAGE_CONCURRENCY", "3")

    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 3, iter(["P1", "P2", "P3"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["one", "two", "three"],
    )

    def _fake_ocr(image_b64: str, *, settings: object) -> str:
        _ = settings
        if image_b64 == "P1":
            time.sleep(0.15)
            return "one"
        if image_b64 == "P2":
            return "two"
        time.sleep(0.05)
        return "three"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    pages: list[int] = []
    _ = pdf_to_html(
        b"%PDF",
        full_document=False,
        on_page=lambda page_index, _html: pages.append(page_index),
    )
    assert pages == [1, 2, 3]


def test_pdf_to_html_aborts_when_ocr_page_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 2, iter(["A", "B"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["A", "B"],
    )

    def _fake_ocr(image_b64: str, *, settings: object) -> str:
        _ = settings
        if image_b64 == "B":
            raise RuntimeError("upstream failure")
        return "A"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    with pytest.raises(RuntimeError, match="LightOn OCR failed"):
        _ = pdf_to_html(b"%PDF", full_document=False)


def test_pdf_to_html_fails_fast_without_waiting_all_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_PAGE_CONCURRENCY", "2")

    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 2, iter(["FAIL_FAST", "SLOW_PAGE"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["A", "B"])

    def _fake_ocr(image_b64: str, *, settings: object) -> str:
        _ = settings
        if image_b64 == "FAIL_FAST":
            time.sleep(0.05)
            raise RuntimeError("upstream failure")
        time.sleep(0.5)
        return "B"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    started_at = time.perf_counter()
    with pytest.raises(RuntimeError, match="LightOn OCR failed"):
        _ = pdf_to_html(b"%PDF", full_document=False)
    elapsed = time.perf_counter() - started_at
    assert elapsed < 0.3


def test_pdf_to_html_uses_lighton_render_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_RENDER_DPI", "123")
    monkeypatch.setenv("RAGPREP_LIGHTON_RENDER_MAX_EDGE", "456")

    captured: dict[str, int] = {}

    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        captured["dpi"] = int(dpi or 0)
        captured["max_edge"] = int(max_edge or 0)
        _ = max_pages, max_bytes
        return 1, iter(["P1"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["p1"])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "p1",
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "p1" in html
    assert captured["dpi"] == 123
    assert captured["max_edge"] == 456


def test_pdf_to_html_falls_back_to_pymupdf_text_for_low_quality_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: [
            "売上高 100 営業利益 20 経常利益 10 親会社株主に帰属する中間純利益 5 "
            "売上高 100 営業利益 20 経常利益 10 親会社株主に帰属する中間純利益 5"
        ],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "繝?ｼ繝?ｯ???",
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "売上高 100 営業利益 20" in html


def test_pdf_to_html_replaces_truncated_ocr_table_with_pymupdf_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    words: list[Word] = []
    for row in range(8):
        y = 10 + (row * 12)
        words.append(
            Word(
                x0=10.0,
                y0=float(y),
                x1=70.0,
                y1=float(y + 8),
                text=f"R{row}A",
                block_no=0,
                line_no=row,
                word_no=0,
            )
        )
        words.append(
            Word(
                x0=120.0,
                y0=float(y),
                x1=170.0,
                y1=float(y + 8),
                text=f"{row * 10}",
                block_no=0,
                line_no=row,
                word_no=1,
            )
        )
        words.append(
            Word(
                x0=220.0,
                y0=float(y),
                x1=270.0,
                y1=float(y + 8),
                text=f"{row * 100}",
                block_no=0,
                line_no=row,
                word_no=2,
            )
        )

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["中間連結貸借対照表\n（単位：百万円）\nR1A 10 100\nR2A 20 200"],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [words])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "\n".join(
            [
                "<table>",
                "<tr><th>A</th><th>B</th><th>C</th></tr>",
                "<tr><td>R1A</td><td>10</td><td>100</td></tr>",
                "<tr><td>R2A</td><td>20</td><td>200</td></tr>",
            ]
        ),
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert '<table data-kind="table">' in html
    assert "中間連結貸借対照表" in html
    assert "<td>R7A</td>" in html
    assert "<td>70 / 700</td>" in html


def test_pdf_to_html_keeps_ocr_table_markup_even_when_pymupdf_quality_is_higher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: [
            "これは高品質なPyMuPDF本文です。これは高品質なPyMuPDF本文です。"
            "これは高品質なPyMuPDF本文です。これは高品質なPyMuPDF本文です。"
        ],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "\n".join(
            [
                "<table>",
                "<tr><th>項目</th><th>値</th></tr>",
                "<tr><td>売上高</td><td>100</td></tr>",
            ]
        ),
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert '<table data-kind="table">' in html
    assert "<th>項目</th>" in html


def test_pdf_to_html_replaces_table_preface_with_pymupdf_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: [
            "\n".join(
                [
                    "2026年3月期 第2四半期（中間期）決算短信",
                    "代表取締役社長",
                    "毛籠 勝弘",
                    "財務本部経理部長",
                    "渡部 啓治",
                    "売上高 1,000 10.0%",
                ]
            )
        ],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "\n".join(
            [
                "2026年3月期 第2四半期（中間期）決算短信",
                "代表取締役社長",
                "毛麟 骆弘",
                "財務本部立理部長",
                "渡部 童治",
                "<table>",
                "<tr><th>項目</th><th>値</th></tr>",
                "<tr><td>売上高</td><td>100</td></tr>",
                "</table>",
            ]
        ),
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "毛籠 勝弘" in html
    assert "毛麟 骆弘" not in html
    assert '<table data-kind="table">' in html


def test_pdf_to_html_preface_typo_fix_is_localized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: [
            "\n".join(
                [
                    "会社名 マツダ株式会社",
                    "代表取締役社長 毛籠 勝弘",
                    "売上高 1,000 10.0%",
                ]
            )
        ],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "\n".join(
            [
                "会社名 マツ夕株式会社",
                "代表取締役社長 毛籠 勝弘",
                "OCR固有文言XYZ",
                "<table>",
                "<tr><th>項目</th><th>値</th></tr>",
                "<tr><td>売上高</td><td>100</td></tr>",
                "</table>",
            ]
        ),
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "会社名 マツダ株式会社" in html
    assert "会社名 マツ夕株式会社" not in html
    assert "OCR固有文言XYZ" in html


def test_pdf_to_html_repairs_unclosed_table_with_secondary_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    primary_ocr = "\n".join(
        [
            "セグメント情報",
            "<table>",
            "<tr><th>A</th><th>B</th></tr>",
            "<tr><td>1</td><td>2</td></tr>",
            "</table>",
            "II 当中間連結会計期間",
            "<table>",
            '<tr><th>A</th><th rowspan="2',
        ]
    )
    secondary_ocr = "\n".join(
        [
            "<table>",
            "<tr><th>A</th><th>B</th></tr>",
            "<tr><td>10</td><td>11</td></tr>",
            "</table>",
            "(重要な後発事象)",
            "該当事項はありません。",
        ]
    )

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["セグメント情報\n(重要な後発事象)\n該当事項はありません。"],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline._crop_image_base64_to_bottom",
        lambda _image_base64, *, top_ratio=0.45: "CROPPED",
    )

    def _fake_ocr(image_b64: str, *, settings: object) -> str:
        _ = settings
        if image_b64 == "P1":
            return primary_ocr
        if image_b64 == "CROPPED":
            return secondary_ocr
        return primary_ocr

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    html = pdf_to_html(b"%PDF", full_document=False)
    assert html.count('<table data-kind="table">') == 2
    assert "<td>10</td>" in html
    assert "<td>11</td>" in html
    assert "重要な後発事象" in html

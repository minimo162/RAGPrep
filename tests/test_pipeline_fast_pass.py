from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from typing import Any

import pytest
from PIL import Image

from ragprep.pipeline import pdf_to_html


def _iter_one_page(
    _pdf_bytes: bytes,
    *,
    dpi: int | None = None,
    max_edge: int | None = None,
    max_pages: int | None = None,
    max_bytes: int | None = None,
) -> tuple[int, Iterator[str]]:
    _ = dpi, max_edge, max_pages, max_bytes
    return 1, iter(["PAGE1"])


def test_fast_pass_uses_fast_render_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RENDER_DPI", "123")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RENDER_MAX_EDGE", "777")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_MIN_QUALITY", "0")
    captured: dict[str, int] = {}

    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = max_pages, max_bytes
        captured["dpi"] = int(dpi or 0)
        captured["max_edge"] = int(max_edge or 0)
        return 1, iter(["PAGE1"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["good text 123"],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "good text 123",
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "good text 123" in html
    assert captured["dpi"] == 123
    assert captured["max_edge"] == 777


def test_fast_pass_skips_retry_when_quality_conditions_are_not_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RETRY", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_MIN_QUALITY", "0")
    calls = {"ocr": 0}

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["same text"])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])

    def _fake_ocr(_image_b64: str, *, settings: object) -> str:
        _ = settings
        calls["ocr"] += 1
        return "same text"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)
    monkeypatch.setattr(
        "ragprep.pipeline.render_pdf_page_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retry should not run")),
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "same text" in html
    assert calls["ocr"] == 1


def test_fast_pass_retries_and_prefers_retry_when_unclosed_table_is_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RETRY", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_MIN_QUALITY", "1.0")
    calls = {"ocr": 0, "retry_render": 0}

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["ref text"])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])

    def _fake_ocr(_image_b64: str, *, settings: object) -> str:
        _ = settings
        calls["ocr"] += 1
        if calls["ocr"] == 1:
            return "before\n<table>\n<tr><th>A</th>"
        return "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"

    def _fake_render_retry(
        _pdf_bytes: bytes,
        *,
        page_index: int,
        dpi: int,
        max_edge: int,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> Image.Image:
        _ = page_index, dpi, max_edge, max_pages, max_bytes
        calls["retry_render"] += 1
        return Image.new("RGB", (2, 2), color="white")

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)
    monkeypatch.setattr("ragprep.pipeline.render_pdf_page_image", _fake_render_retry)

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "<td>1</td>" in html
    assert "before" not in html
    assert calls["retry_render"] == 1
    assert calls["ocr"] == 2


def test_fast_pass_falls_back_to_primary_when_retry_ocr_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RETRY", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_MIN_QUALITY", "0.95")
    calls = {"ocr": 0}

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["abc"])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline.render_pdf_page_image",
        lambda *args, **kwargs: Image.new("RGB", (2, 2), color="white"),
    )

    def _fake_ocr(_image_b64: str, *, settings: object) -> str:
        _ = settings
        calls["ocr"] += 1
        if calls["ocr"] == 1:
            return "abc"
        raise RuntimeError("retry failed")

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "abc" in html
    assert calls["ocr"] == 2


def test_fast_pass_prefers_retry_when_quality_improves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RETRY", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_MIN_QUALITY", "0.95")
    calls = {"ocr": 0}

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["abc"])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline.render_pdf_page_image",
        lambda *args, **kwargs: Image.new("RGB", (2, 2), color="white"),
    )

    def _fake_ocr(_image_b64: str, *, settings: object) -> str:
        _ = settings
        calls["ocr"] += 1
        if calls["ocr"] == 1:
            return "abc"
        return "This is a much better OCR sentence with digits 12345."

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "much better OCR sentence" in html
    assert calls["ocr"] == 2


def test_fast_pass_can_be_disabled_to_use_legacy_render_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "0")
    monkeypatch.setenv("RAGPREP_LIGHTON_RENDER_DPI", "222")
    monkeypatch.setenv("RAGPREP_LIGHTON_RENDER_MAX_EDGE", "888")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_MIN_QUALITY", "0.99")
    captured: dict[str, int] = {}

    def _fake_iter_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = max_pages, max_bytes
        captured["dpi"] = int(dpi or 0)
        captured["max_edge"] = int(max_edge or 0)
        return 1, iter(["PAGE1"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _fake_iter_pages)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["abc"])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "abc",
    )
    monkeypatch.setattr(
        "ragprep.pipeline.render_pdf_page_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retry should not run")),
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "abc" in html
    assert captured["dpi"] == 222
    assert captured["max_edge"] == 888


def test_fast_pass_closes_unclosed_table_without_extra_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RETRY", "0")
    monkeypatch.setenv("RAGPREP_LIGHTON_SECONDARY_TABLE_REPAIR", "0")
    calls = {"ocr": 0}

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["ref"])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])

    def _fake_ocr(_image_b64: str, *, settings: object) -> str:
        _ = settings
        calls["ocr"] += 1
        return "<table><tr><th>A</th>"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)
    monkeypatch.setattr(
        "ragprep.pipeline.render_pdf_page_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retry should not run")),
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert '<table data-kind="table">' in html
    assert "<th>A</th>" in html
    assert calls["ocr"] == 1


def test_fast_pass_uses_page_type_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_MAX_TOKENS", "8192")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TEXT", "3333")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TABLE", "7777")
    monkeypatch.setenv("RAGPREP_LIGHTON_PAGE_CONCURRENCY", "1")

    def _iter_two_pages(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 2, iter(["P1", "P2"])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_two_pages)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["one", "two"])
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_words",
        lambda _pdf: [[], [object()]],
    )
    monkeypatch.setattr(
        "ragprep.pipeline._estimate_page_table_likelihood",
        lambda words: 0.8 if words else 0.1,
    )

    captured_tokens: list[int] = []

    def _fake_ocr(_image_b64: str, *, settings: Any) -> str:
        captured_tokens.append(int(settings.lighton_max_tokens))
        return "sample text"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "sample text" in html
    assert captured_tokens == [3333, 7777]


def test_fast_pass_downscales_non_table_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_NON_TABLE_MAX_EDGE", "800")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_TABLE_LIKELIHOOD_THRESHOLD", "0.8")

    source_image = Image.new("RGB", (1600, 1000), color="white")
    raw = io.BytesIO()
    source_image.save(raw, format="PNG")
    image_b64 = base64.b64encode(raw.getvalue()).decode("ascii")

    def _iter_one_image_page(
        _pdf_bytes: bytes,
        *,
        dpi: int | None = None,
        max_edge: int | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[int, Iterator[str]]:
        _ = dpi, max_edge, max_pages, max_bytes
        return 1, iter([image_b64])

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_image_page)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["text"])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr("ragprep.pipeline._estimate_page_table_likelihood", lambda _words: 0.0)

    captured_sizes: list[tuple[int, int]] = []

    def _fake_ocr(image_payload: str, *, settings: object) -> str:
        _ = settings
        with Image.open(io.BytesIO(base64.b64decode(image_payload, validate=True))) as image:
            width, height = image.size
            captured_sizes.append((int(width), int(height)))
        return "text"

    monkeypatch.setattr("ragprep.pipeline.lighton_ocr.ocr_image_base64", _fake_ocr)

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "text" in html
    assert captured_sizes
    assert max(captured_sizes[0]) <= 800


def test_fast_postprocess_light_applies_text_correction_but_skips_table_correction_on_non_table_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "light")

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["abc"])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr("ragprep.pipeline._estimate_page_table_likelihood", lambda _words: 0.0)
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "abc",
    )
    called = {"text_correction": 0}

    def _spy_text_correction(*, page: object, pymupdf_text: str, **kwargs: object) -> object:
        _ = pymupdf_text, kwargs
        called["text_correction"] += 1
        return page

    monkeypatch.setattr(
        "ragprep.pipeline._correct_text_blocks_locally_with_pymupdf",
        _spy_text_correction,
    )
    monkeypatch.setattr(
        "ragprep.pipeline._correct_table_blocks_locally_with_pymupdf",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "abc" in html
    assert called["text_correction"] == 1


def test_fast_postprocess_light_corrects_split_name_tokens_with_pymupdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "light")

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: [
            "\n".join(
                [
                    "Representative",
                    "Director",
                    "(Unit: million yen)",
                    "Representative Director",
                    "Net sales",
                ]
            )
        ],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr("ragprep.pipeline._estimate_page_table_likelihood", lambda _words: 0.0)
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "Representat1ve D1rector (Unlt: mllllon yen)",
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "Representative Director" in html
    assert "(Unit: million yen)" in html
    assert "Representat1ve D1rector" not in html


def test_legacy_mode_keeps_full_postprocess_even_if_fast_postprocess_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "0")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "off")

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: ["abc"])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "abc",
    )

    called = {"text_correction": 0}

    def _spy_text_correction(*, page: object, pymupdf_text: str) -> object:
        _ = pymupdf_text
        called["text_correction"] += 1
        return page

    monkeypatch.setattr(
        "ragprep.pipeline._correct_text_blocks_locally_with_pymupdf",
        _spy_text_correction,
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "abc" in html
    assert called["text_correction"] == 1

def test_fast_postprocess_light_splits_compound_toc_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "light")

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: [
            "1. Overview .......... 2\n"
            "(1) Revenue summary .......... 2\n"
            "(2) Financial position .......... 2\n"
            "(3) Forecast .......... 3"
        ],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr("ragprep.pipeline._estimate_page_table_likelihood", lambda _words: 0.0)
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: (
            "1. Overview .......... 2 "
            "(1) Revenue summary .......... 2 "
            "(2) Financial position .......... 2 "
            "(3) Forecast .......... 3"
        ),
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "Overview .......... 2<br />" in html or "Overview .......... 2 <br />" in html
    assert "(1) Revenue summary" in html
    assert "(2) Financial position" in html
    assert "(3) Forecast" in html


def test_fast_postprocess_promotes_section_heading_and_removes_footer_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "light")

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["01 | Company Overview\nkubell\n1"],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr("ragprep.pipeline._estimate_page_table_likelihood", lambda _words: 0.0)
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "01 | Company Overview\n\nkubell\n\n1",
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "<h2>01 | Company Overview</h2>" in html
    assert "<p>kubell</p>" not in html
    assert "<p>1</p>" not in html


def test_fast_postprocess_promotes_business_overview_heading_and_splits_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "light")

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr(
        "ragprep.pipeline.extract_pymupdf_page_texts",
        lambda _pdf: ["1 事業概要｜ビジネスチャット事業 *1 Footnote line"],
    )
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr("ragprep.pipeline._estimate_page_table_likelihood", lambda _words: 0.0)
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: "1 事業概要｜ビジネスチャット事業 *1 Footnote line",
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "<h2>事業概要｜ビジネスチャット事業</h2>" in html
    assert "<p>*1 Footnote line</p>" in html
    assert "1 事業概要｜ビジネスチャット事業" not in html


def test_fast_postprocess_promotes_long_business_overview_heading_with_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "light")

    source = (
        "1 事業概要｜BPaaS（Business Process as a Service）事業 "
        "チャット経由で業務プロセスと人材を組み合わせるサービスです。"
    )

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: [source])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr("ragprep.pipeline._estimate_page_table_likelihood", lambda _words: 0.0)
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: source,
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "<h2>事業概要｜BPaaS（Business Process as a Service）事業</h2>" in html
    assert "チャット経由で業務プロセス" in html


def test_fast_postprocess_splits_compound_issue_paragraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "light")

    source = "\n".join(
        [
            "向き合う社会課題①人口減少",
            "* 出典注記の例",
            "2000年代初頭に人口のピークアウトを迎えた日本は、",
            "世界に先駆けて課題解決のスタートラインに立っている。",
            "持続可能な社会のため、働き方の構造変革が必要不可欠である。",
            "世界に先駆けて挑む、",
            "持続可能な社会への",
            "構造変革",
            "2000年代初頭に",
            "ピークアウト",
            "日本の人口の推移(万人)",
        ]
    )

    monkeypatch.setattr("ragprep.pipeline.iter_pdf_page_png_base64", _iter_one_page)
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_texts", lambda _pdf: [source])
    monkeypatch.setattr("ragprep.pipeline.extract_pymupdf_page_words", lambda _pdf: [[]])
    monkeypatch.setattr("ragprep.pipeline._estimate_page_table_likelihood", lambda _words: 0.0)
    monkeypatch.setattr(
        "ragprep.pipeline.lighton_ocr.ocr_image_base64",
        lambda _image_b64, *, settings: source,
    )

    html = pdf_to_html(b"%PDF", full_document=False)
    assert "<h2>向き合う社会課題①人口減少</h2>" in html
    assert "<p>* 出典注記の例</p>" in html
    assert "2000年代初頭に人口のピークアウトを迎えた日本は、" in html
    assert "<h2>世界に先駆けて挑む、持続可能な社会への構造変革</h2>" in html
    assert "日本の人口の推移(万人)" in html


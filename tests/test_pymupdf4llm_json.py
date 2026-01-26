from __future__ import annotations

import json
from typing import cast

import pytest

from ragprep import pymupdf4llm_json


def _make_pdf_bytes() -> bytes:
    import fitz

    doc = fitz.open()
    doc.new_page()
    return cast(bytes, doc.tobytes())


def test_pdf_bytes_to_json_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="pdf_bytes is empty"):
        pymupdf4llm_json.pdf_bytes_to_json(b"")


def test_pdf_bytes_to_json_filters_header_footer(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = {
        "pages": [
            {
                "boxes": [
                    {"boxclass": "page-header", "text": "HEADER"},
                    {"boxclass": "body", "text": "BODY"},
                    {"boxclass": "page-footer", "text": "FOOTER"},
                ],
                "number": 1,
            },
            {"boxes": [{"boxclass": "body", "text": "BODY2"}], "number": 2},
        ],
        "meta": {"title": "doc"},
    }

    monkeypatch.setattr(
        pymupdf4llm_json.pymupdf4llm,
        "to_json",
        lambda _doc: json.dumps(sample),
    )

    result = pymupdf4llm_json.pdf_bytes_to_json(_make_pdf_bytes())
    data = json.loads(result)

    assert data["meta"]["title"] == "doc"
    assert [box["text"] for box in data["pages"][0]["boxes"]] == ["BODY"]
    assert [box["text"] for box in data["pages"][1]["boxes"]] == ["BODY2"]


def test_pdf_bytes_to_json_falls_back_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pymupdf4llm_json.pymupdf4llm,
        "to_json",
        lambda _doc: "not-json",
    )

    result = pymupdf4llm_json.pdf_bytes_to_json(_make_pdf_bytes())
    assert result == "not-json"

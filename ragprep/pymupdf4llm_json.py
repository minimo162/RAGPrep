from __future__ import annotations

import json
from typing import Any

import pymupdf
import pymupdf.layout as pymupdf_layout
import pymupdf4llm

if pymupdf_layout is None:
    raise RuntimeError("pymupdf.layout failed to import")


def _is_header_or_footer(box: Any) -> bool:
    if not isinstance(box, dict):
        return False
    boxclass = box.get("boxclass")
    return boxclass in {"page-header", "page-footer"}


def _strip_headers_footers(data: Any) -> tuple[Any, bool]:
    if not isinstance(data, dict):
        return data, False

    pages = data.get("pages")
    if not isinstance(pages, list):
        return data, False

    changed = False
    new_pages: list[Any] = []
    for page in pages:
        if not isinstance(page, dict):
            new_pages.append(page)
            continue
        boxes = page.get("boxes")
        if not isinstance(boxes, list):
            new_pages.append(page)
            continue
        filtered = [box for box in boxes if not _is_header_or_footer(box)]
        if len(filtered) != len(boxes):
            changed = True
            page = {**page, "boxes": filtered}
        new_pages.append(page)

    if not changed:
        return data, False

    updated = dict(data)
    updated["pages"] = new_pages
    return updated, True


def _filter_header_footer_json(raw_json: str) -> str:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    filtered, changed = _strip_headers_footers(data)
    if not changed:
        return raw_json

    try:
        return json.dumps(filtered, indent=1)
    except (TypeError, ValueError):
        return raw_json


def pdf_bytes_to_json(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    with doc:
        raw_json = pymupdf4llm.to_json(doc)
    return _filter_header_footer_json(str(raw_json))

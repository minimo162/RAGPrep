from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pymupdf
import pymupdf.layout  # noqa: F401
import pymupdf4llm


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _visible_char_count(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


@dataclass(frozen=True)
class _TextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    visible_chars: int


def _extract_text_blocks(page: Any) -> list[_TextBlock]:
    try:
        raw_blocks = page.get_text("blocks") or []
    except Exception:  # noqa: BLE001
        return []

    blocks: list[_TextBlock] = []
    for item in raw_blocks:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        try:
            x0 = float(item[0])
            y0 = float(item[1])
            x1 = float(item[2])
            y1 = float(item[3])
            text = str(item[4] or "")
        except Exception:  # noqa: BLE001
            continue
        visible_chars = _visible_char_count(text)
        if visible_chars <= 0:
            continue
        blocks.append(_TextBlock(x0=x0, y0=y0, x1=x1, y1=y1, visible_chars=visible_chars))
    return blocks


def _should_use_sorted_text_for_sidebar_page(page: Any) -> bool:
    """
    Heuristic: single-column body + right-side sidebar/callout in the middle of the page.

    We keep this intentionally narrow for task-00. Multi-column handling is deferred.
    """

    blocks = _extract_text_blocks(page)
    if len(blocks) < 2:
        return False

    rect = getattr(page, "rect", None)
    page_width = float(getattr(rect, "width", 0.0)) if rect is not None else 0.0
    if page_width <= 0.0:
        return False

    main = max(blocks, key=lambda b: b.visible_chars)
    if main.visible_chars < 20:
        return False

    # Right-side blocks are potential sidebars/callouts.
    right_blocks = [b for b in blocks if b.x0 >= (page_width * 0.50) and b.visible_chars >= 5]
    if not right_blocks:
        return False

    right_chars = sum(b.visible_chars for b in right_blocks)
    # If the right side has comparable amount of text, it's likely multi-column content.
    if right_chars >= (main.visible_chars * 0.70):
        return False

    right_y0 = min(b.y0 for b in right_blocks)
    right_y1 = max(b.y1 for b in right_blocks)

    # A sidebar we care about lives inside the main flow (so placing it at the end is unnatural).
    if not (main.y0 < right_y0 and main.y1 > right_y1):
        return False

    # Avoid ambiguous cases where the main block is not on the left.
    if main.x0 > (page_width * 0.45):
        return False

    return True


def _extract_sorted_page_text(page: Any) -> str:
    try:
        text = page.get_text("text", sort=True) or ""
    except Exception:  # noqa: BLE001
        text = page.get_text("text") or ""
    return _normalize_newlines(str(text)).strip()


def pdf_bytes_to_markdown(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    with doc:
        page_count = int(doc.page_count)
        sidebar_pages: set[int] = set()
        for i in range(page_count):
            page = doc.load_page(i)
            if _should_use_sorted_text_for_sidebar_page(page):
                sidebar_pages.add(i)

        if not sidebar_pages:
            markdown = pymupdf4llm.to_markdown(doc)
            return _normalize_newlines(str(markdown)).strip()

        parts: list[str] = []
        for i in range(page_count):
            page = doc.load_page(i)
            if i in sidebar_pages:
                parts.append(_extract_sorted_page_text(page))
                continue
            page_md = pymupdf4llm.to_markdown(doc, pages=[i])
            parts.append(_normalize_newlines(str(page_md)).strip())

        markdown = "\n\n".join(part for part in parts if part)

    return _normalize_newlines(str(markdown)).strip()

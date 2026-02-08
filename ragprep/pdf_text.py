from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ragprep.config import get_settings

_REPLACEMENT_CHAR = "\ufffd"

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"[ \t]+")


class PageKind(str, Enum):
    text = "text"
    table = "table"
    image = "image"
    mixed = "mixed"
    empty = "empty"


@dataclass(frozen=True)
class TextQualityScore:
    char_count: int
    visible_char_count: int
    visible_ratio: float
    replacement_char_ratio: float
    symbol_ratio: float
    longest_repeat_ratio: float
    score: float


@dataclass(frozen=True)
class PageAnalysis:
    page_number: int
    raw_text: str
    normalized_text: str
    tokens: tuple[str, ...]
    has_text_layer: bool
    text_quality: TextQualityScore
    image_count: int
    image_area_ratio: float
    table_likelihood: float
    page_kind: PageKind


@dataclass(frozen=True)
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_no: int
    line_no: int
    word_no: int


@dataclass(frozen=True)
class Span:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float | None = None
    flags: int | None = None
    font: str | None = None


def _import_fitz() -> Any:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # noqa: BLE001
        raise ImportError("PyMuPDF is required. Install `pymupdf`.") from exc
    return fitz


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_extracted_text(text: str) -> str:
    """
    Normalize extracted text deterministically.

    Notes:
    - We avoid aggressive whitespace collapsing because it can destroy alignment hints
      (e.g., table-like content). Instead, we normalize line endings and remove control chars.
    """

    normalized = _normalize_newlines(text)
    normalized = normalized.replace("\u00a0", " ").replace("\u3000", " ")
    normalized = _CONTROL_CHARS_RE.sub("", normalized)

    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines)


def _char_class(ch: str) -> str:
    code = ord(ch)
    if 0x3040 <= code <= 0x309F:
        return "hiragana"
    if 0x30A0 <= code <= 0x30FF or 0xFF66 <= code <= 0xFF9D:
        return "katakana"
    if 0x4E00 <= code <= 0x9FFF:
        return "kanji"
    if "0" <= ch <= "9":
        return "digit"
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return "latin"
    return "symbol"


def tokenize_by_char_class(text: str) -> list[str]:
    """
    Tokenize text without language-specific dictionaries.

    The primary goal is stable diff/merge behavior for OCR vs PyMuPDF comparisons.
    """

    tokens: list[str] = []
    buf: list[str] = []
    buf_kind: str | None = None

    def flush() -> None:
        nonlocal buf, buf_kind
        if buf:
            tokens.append("".join(buf))
            buf = []
            buf_kind = None

    for ch in text:
        if ch.isspace():
            flush()
            continue
        kind = _char_class(ch)
        if buf_kind is None:
            buf_kind = kind
            buf.append(ch)
            continue
        if kind == buf_kind:
            buf.append(ch)
            continue
        flush()
        buf_kind = kind
        buf.append(ch)

    flush()
    return tokens


def _longest_repeat_ratio(text: str) -> float:
    if not text:
        return 0.0
    best = 1
    current = 1
    prev = ""
    for ch in text:
        if not prev:
            prev = ch
            continue
        if ch == prev:
            current += 1
        else:
            best = max(best, current)
            current = 1
            prev = ch
    best = max(best, current)
    return best / max(1, len(text))


def score_text_quality(text: str) -> TextQualityScore:
    normalized = normalize_extracted_text(text)
    char_count = len(normalized)
    if char_count == 0:
        return TextQualityScore(
            char_count=0,
            visible_char_count=0,
            visible_ratio=0.0,
            replacement_char_ratio=0.0,
            symbol_ratio=0.0,
            longest_repeat_ratio=0.0,
            score=0.0,
        )

    visible_char_count = sum(1 for ch in normalized if not ch.isspace())
    visible_ratio = visible_char_count / char_count if char_count else 0.0

    replacement_char_ratio = normalized.count(_REPLACEMENT_CHAR) / char_count

    visible_text = _WHITESPACE_RE.sub("", normalized)
    symbol_count = sum(1 for ch in visible_text if _char_class(ch) == "symbol")
    symbol_ratio = symbol_count / max(1, len(visible_text))

    longest_repeat_ratio = _longest_repeat_ratio(visible_text)

    # Deterministic heuristic score [0, 1].
    length_factor = min(1.0, visible_char_count / 80.0)
    score = length_factor
    score *= max(0.0, 1.0 - (replacement_char_ratio * 8.0))
    score *= max(0.0, 1.0 - (symbol_ratio * 2.0))
    score *= max(0.0, 1.0 - (max(0.0, longest_repeat_ratio - 0.1) * 3.0))
    score = max(0.0, min(1.0, score))

    return TextQualityScore(
        char_count=char_count,
        visible_char_count=visible_char_count,
        visible_ratio=visible_ratio,
        replacement_char_ratio=replacement_char_ratio,
        symbol_ratio=symbol_ratio,
        longest_repeat_ratio=longest_repeat_ratio,
        score=score,
    )


def extract_pymupdf_page_texts(pdf_bytes: bytes) -> list[str]:
    """
    Extract per-page text using PyMuPDF.

    This is intended as a supplementary signal for OCR: it does NOT read text inside images.
    """

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    settings = get_settings()
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={settings.max_upload_bytes}"
        )

    fitz = _import_fitz()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid PDF data") from exc

    try:
        page_count = int(doc.page_count)
    except Exception as exc:  # noqa: BLE001
        doc.close()
        raise RuntimeError("Failed to read PDF page count") from exc
    if page_count > settings.max_pages:
        doc.close()
        raise ValueError(f"PDF has {page_count} pages, max_pages={settings.max_pages}")

    pages: list[str] = []
    with doc:
        for i in range(page_count):
            page = doc.load_page(i)
            pages.append(str(page.get_text("text") or ""))
    return pages


def extract_pymupdf_page_sizes(pdf_bytes: bytes) -> list[tuple[float, float]]:
    """
    Extract per-page sizes (width, height) in PyMuPDF page coordinates.
    """

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    settings = get_settings()
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={settings.max_upload_bytes}"
        )

    fitz = _import_fitz()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid PDF data") from exc

    try:
        page_count = int(doc.page_count)
    except Exception as exc:  # noqa: BLE001
        doc.close()
        raise RuntimeError("Failed to read PDF page count") from exc
    if page_count > settings.max_pages:
        doc.close()
        raise ValueError(f"PDF has {page_count} pages, max_pages={settings.max_pages}")

    sizes: list[tuple[float, float]] = []
    with doc:
        for i in range(page_count):
            page = doc.load_page(i)
            rect = getattr(page, "rect", None)
            width = float(getattr(rect, "width", 0.0)) if rect is not None else 0.0
            height = float(getattr(rect, "height", 0.0)) if rect is not None else 0.0
            sizes.append((width, height))
    return sizes


def extract_pymupdf_page_spans(pdf_bytes: bytes) -> list[list[Span]]:
    """
    Extract per-page text spans with bounding boxes using PyMuPDF.

    This reads the PDF text layer only (it does NOT OCR images).
    """

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    settings = get_settings()
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={settings.max_upload_bytes}"
        )

    fitz = _import_fitz()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid PDF data") from exc

    try:
        page_count = int(doc.page_count)
    except Exception as exc:  # noqa: BLE001
        doc.close()
        raise RuntimeError("Failed to read PDF page count") from exc
    if page_count > settings.max_pages:
        doc.close()
        raise ValueError(f"PDF has {page_count} pages, max_pages={settings.max_pages}")

    pages: list[list[Span]] = []
    with doc:
        for i in range(page_count):
            page = doc.load_page(i)
            try:
                data = page.get_text("dict") or {}
            except Exception:  # noqa: BLE001
                pages.append([])
                continue

            blocks = data.get("blocks")
            if not isinstance(blocks, list):
                pages.append([])
                continue

            spans: list[Span] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if int(block.get("type", -1)) != 0:
                    continue
                lines = block.get("lines")
                if not isinstance(lines, list):
                    continue
                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    line_spans = line.get("spans")
                    if not isinstance(line_spans, list):
                        continue
                    for span in line_spans:
                        if not isinstance(span, dict):
                            continue
                        bbox = span.get("bbox")
                        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                            continue
                        try:
                            x0 = float(bbox[0])
                            y0 = float(bbox[1])
                            x1 = float(bbox[2])
                            y1 = float(bbox[3])
                        except Exception:  # noqa: BLE001
                            continue
                        if not (x0 < x1 and y0 < y1):
                            continue

                        text = normalize_extracted_text(str(span.get("text") or "")).strip()
                        if not text:
                            continue

                        size_obj = span.get("size")
                        size = float(size_obj) if isinstance(size_obj, (int, float)) else None
                        flags_obj = span.get("flags")
                        flags = (
                            int(flags_obj)
                            if isinstance(flags_obj, int) and not isinstance(flags_obj, bool)
                            else None
                        )
                        font_obj = span.get("font")
                        font = str(font_obj) if isinstance(font_obj, str) and font_obj else None

                        spans.append(
                            Span(
                                x0=x0,
                                y0=y0,
                                x1=x1,
                                y1=y1,
                                text=text,
                                size=size,
                                flags=flags,
                                font=font,
                            )
                        )

            spans.sort(key=lambda s: (s.y0, s.x0, s.y1, s.x1, s.text))
            pages.append(spans)

    return pages


def extract_pymupdf_page_words(pdf_bytes: bytes) -> list[list[Word]]:
    """
    Extract per-page words (bbox + text) using PyMuPDF text layer.
    """

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    settings = get_settings()
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={settings.max_upload_bytes}"
        )

    fitz = _import_fitz()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid PDF data") from exc

    try:
        page_count = int(doc.page_count)
    except Exception as exc:  # noqa: BLE001
        doc.close()
        raise RuntimeError("Failed to read PDF page count") from exc
    if page_count > settings.max_pages:
        doc.close()
        raise ValueError(f"PDF has {page_count} pages, max_pages={settings.max_pages}")

    pages: list[list[Word]] = []
    with doc:
        for i in range(page_count):
            page = doc.load_page(i)
            words = _extract_words(page)
            words.sort(key=lambda w: (w.y0, w.x0, w.y1, w.x1, w.text))
            pages.append(words)
    return pages


def _extract_words(page: Any) -> list[Word]:
    words_raw = page.get_text("words") or []
    words: list[Word] = []
    if not isinstance(words_raw, list):
        return words

    for item in words_raw:
        if not isinstance(item, (list, tuple)) or len(item) < 8:
            continue
        try:
            x0 = float(item[0])
            y0 = float(item[1])
            x1 = float(item[2])
            y1 = float(item[3])
            text = str(item[4] or "")
            block_no = int(item[5])
            line_no = int(item[6])
            word_no = int(item[7])
        except Exception:  # noqa: BLE001
            continue
        if not text:
            continue
        words.append(
            Word(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                text=text,
                block_no=block_no,
                line_no=line_no,
                word_no=word_no,
            )
        )
    return words


def estimate_table_likelihood(words: Iterable[Word]) -> float:
    """
    Estimate table-likeness from word positions.

    This is intentionally heuristic and should be treated as a weak signal.
    """

    word_list = list(words)
    if len(word_list) < 12:
        return 0.0

    # Group words into "rows" by y-position. PyMuPDF's internal line segmentation can split
    # far-apart columns into different lines, so we use a simple y-bin for robustness.
    row_bin = 5.0
    rows: dict[int, list[Word]] = {}
    for w in word_list:
        key = int(round(w.y0 / row_bin))
        rows.setdefault(key, []).append(w)

    if len(rows) < 6:
        return 0.0

    rows_with_3plus = sum(1 for ws in rows.values() if len(ws) >= 3)
    multiword_ratio = rows_with_3plus / max(1, len(rows))

    # Column alignment: count x-bins that repeat across many rows.
    bin_size = 20.0
    bin_rows: dict[int, set[int]] = {}
    for key, ws in rows.items():
        for w in ws:
            b = int(round(w.x0 / bin_size))
            bin_rows.setdefault(b, set()).add(key)

    min_rows = max(3, round(len(rows) * 0.35))
    repeated_bins = sum(1 for ks in bin_rows.values() if len(ks) >= min_rows)
    alignment_score = min(1.0, repeated_bins / 3.0)

    digit_words = sum(1 for w in word_list if any("0" <= ch <= "9" for ch in w.text))
    digit_ratio = digit_words / max(1, len(word_list))

    score = (alignment_score * 0.55) + (multiword_ratio * 0.30) + (digit_ratio * 0.15)
    return max(0.0, min(1.0, score))


def _safe_find_tables_score(page: Any) -> float:
    """
    Optional table detection using PyMuPDF's `find_tables()`.

    Warning: detection is not perfectly accurate; treat as a weak hint.
    """

    find_tables = getattr(page, "find_tables", None)
    if find_tables is None:
        return 0.0

    try:
        finder = find_tables()
    except Exception:  # noqa: BLE001
        return 0.0

    tables = getattr(finder, "tables", None)
    if isinstance(tables, list) and tables:
        return 1.0
    return 0.0


def _extract_image_signals(page: Any) -> tuple[int, float]:
    try:
        data = page.get_text("dict") or {}
    except Exception:  # noqa: BLE001
        return 0, 0.0

    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        return 0, 0.0

    rect = getattr(page, "rect", None)
    page_width = float(getattr(rect, "width", 0.0)) if rect is not None else 0.0
    page_height = float(getattr(rect, "height", 0.0)) if rect is not None else 0.0
    page_area = page_width * page_height

    image_count = 0
    image_area = 0.0

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if int(block.get("type", -1)) != 1:
            continue
        image_count += 1
        bbox = block.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except Exception:  # noqa: BLE001
            continue
        image_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)

    ratio = (image_area / page_area) if page_area > 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    return image_count, ratio


def classify_page_kind(
    *,
    has_text_layer: bool,
    image_area_ratio: float,
    table_likelihood: float,
) -> PageKind:
    if image_area_ratio >= 0.60:
        return PageKind.image
    if table_likelihood >= 0.65:
        return PageKind.table
    if has_text_layer and image_area_ratio > 0.10:
        return PageKind.mixed
    if has_text_layer:
        return PageKind.text
    if image_area_ratio > 0.10:
        return PageKind.image
    return PageKind.empty


def analyze_pdf_pages(pdf_bytes: bytes, *, use_find_tables: bool = False) -> list[PageAnalysis]:
    """
    Analyze a PDF page-by-page using PyMuPDF.

    - Extracts text per page (not image OCR).
    - Collects weak signals for table/image pages.
    """

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    settings = get_settings()
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={settings.max_upload_bytes}"
        )

    fitz = _import_fitz()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid PDF data") from exc

    try:
        page_count = int(doc.page_count)
    except Exception as exc:  # noqa: BLE001
        doc.close()
        raise RuntimeError("Failed to read PDF page count") from exc
    if page_count > settings.max_pages:
        doc.close()
        raise ValueError(f"PDF has {page_count} pages, max_pages={settings.max_pages}")

    results: list[PageAnalysis] = []
    with doc:
        for i in range(page_count):
            page_number = i + 1
            page = doc.load_page(i)

            raw_text = str(page.get_text("text") or "")
            normalized_text = normalize_extracted_text(raw_text)
            has_text_layer = bool(normalized_text.strip())

            tokens = tuple(tokenize_by_char_class(normalized_text))
            text_quality = score_text_quality(normalized_text)

            words = _extract_words(page)
            table_likelihood = estimate_table_likelihood(words)
            if use_find_tables:
                table_likelihood = max(table_likelihood, _safe_find_tables_score(page))

            image_count, image_area_ratio = _extract_image_signals(page)

            page_kind = classify_page_kind(
                has_text_layer=has_text_layer,
                image_area_ratio=image_area_ratio,
                table_likelihood=table_likelihood,
            )

            results.append(
                PageAnalysis(
                    page_number=page_number,
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    tokens=tokens,
                    has_text_layer=has_text_layer,
                    text_quality=text_quality,
                    image_count=image_count,
                    image_area_ratio=image_area_ratio,
                    table_likelihood=table_likelihood,
                    page_kind=page_kind,
                )
            )

    return results

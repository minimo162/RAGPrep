from __future__ import annotations

import base64
import binascii
import difflib
import io
import re
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from enum import Enum
from html import unescape as html_unescape
from pathlib import Path

from ragprep.config import Settings, get_settings
from ragprep.html_render import render_document_html, render_page_html, wrap_html_document
from ragprep.ocr import lighton_ocr
from ragprep.ocr_html import page_from_ocr_markdown
from ragprep.pdf_render import iter_pdf_page_png_base64, render_pdf_page_image
from ragprep.pdf_text import (
    Word,
    estimate_table_likelihood,
    extract_pymupdf_page_texts,
    extract_pymupdf_page_words,
    normalize_extracted_text,
    score_text_quality,
    tokenize_by_char_class,
)
from ragprep.structure_ir import Block, Document, Heading, Page, Paragraph, Table, Unknown
from ragprep.table_grid import TableCell, build_table_grid
from ragprep.table_merge import merge_markdown_tables_with_pymupdf_words
from ragprep.text_merge import merge_ocr_with_pymupdf


class ProgressPhase(str, Enum):
    rendering = "rendering"
    done = "done"


@dataclass(frozen=True)
class PdfToHtmlProgress:
    phase: ProgressPhase
    current: int
    total: int
    message: str | None = None


HtmlProgressCallback = Callable[[PdfToHtmlProgress], None]
PageCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class _OcrPassOutcome:
    ocr_text: str
    merged_text: str
    merged_quality: float
    has_unclosed_table: bool


@dataclass(frozen=True)
class _PrimaryOcrPlan:
    image_base64: str
    max_tokens: int
    table_likelihood: float


def _notify_html_progress(
    on_progress: HtmlProgressCallback | None,
    update: PdfToHtmlProgress,
) -> None:
    if on_progress is None:
        return
    try:
        on_progress(update)
    except Exception:  # noqa: BLE001
        return


def _write_text_artifact(path: Path, text: str) -> None:
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


_RAW_TABLE_OPEN_RE = re.compile(r"<\s*table\b", re.IGNORECASE)
_RAW_TABLE_CLOSE_RE = re.compile(r"</\s*table\s*>", re.IGNORECASE)
_ESCAPED_TABLE_OPEN_RE = re.compile(r"&lt;\s*table\b", re.IGNORECASE)
_ESCAPED_TABLE_CLOSE_RE = re.compile(r"&lt;\s*/\s*table\s*&gt;", re.IGNORECASE)
_RAW_COMPLETE_TABLE_RE = re.compile(r"<\s*table\b.*?</\s*table\s*>", re.IGNORECASE | re.DOTALL)
_ESCAPED_COMPLETE_TABLE_RE = re.compile(
    r"&lt;\s*table\b.*?&lt;\s*/\s*table\s*&gt;",
    re.IGNORECASE | re.DOTALL,
)
_RAW_TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z0-9]+)(?:\s+[^>]*)?>", re.IGNORECASE)
_TABLE_TAGS: set[str] = {"table", "thead", "tbody", "tr", "th", "td"}
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_MARKDOWN_HR_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
_NOTE_PREFIX_RE = re.compile(r"^\*?\s*note\s*:\s*", re.IGNORECASE)
_TRANSCRIPTION_PREFIX_RE = re.compile(r"^\*?\s*this transcription\s+is\b", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_HTML_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_TAG_FRAGMENT_START_RE = re.compile(r"^\s*</?\s*([a-zA-Z][a-zA-Z0-9:-]*)\b")
_TABLE_TAG_HINTS: tuple[str, ...] = (
    "<table",
    "</table",
    "<thead",
    "</thead",
    "<tbody",
    "</tbody",
    "<tr",
    "</tr",
    "<th",
    "</th",
    "<td",
    "</td",
)
_TOC_INLINE_ITEM_RE = re.compile(r"[\uFF08(]\s*[0-9\uFF10-\uFF19]+\s*[\)\uFF09]")
_TOC_LEADER_RE = re.compile(r"(?:\u2026|\.|\uFF0E){4,}")
_NOTE_ARTIFACT_HINTS: tuple[str, ...] = (
    "image contains",
    "image shows",
    "placeholder",
    "actual image",
    "replace it with",
    "cannot be represented in markdown",
    "cannot be rendered in markdown",
    "text content has been fully extracted",
    "for illustrative purposes",
    "actual image url",
    "base64",
    "visual styling element",
    "fictional",
    "this transcription is",
    "no additional text or structured data",
)
_PAGE_NUMBER_ONLY_RE = re.compile(r"^\s*[-‐‑‒–—―－−]?\s*([0-9\uFF10-\uFF19]{1,3})\s*[-‐‑‒–—―－−]?\s*$")
_LEADING_PAGE_NUMBER_PREFIX_RE = re.compile(
    r"^\s*([0-9\uFF10-\uFF19]{1,3})\s+(.+)$",
    re.DOTALL,
)
_SECTION_BAR_HEADING_RE = re.compile(
    r"^\s*(?P<section>[0-9\uFF10-\uFF19]{2})\s*[|\uFF5C]\s*(?P<title>\S(?:.*\S)?)\s*$",
    re.DOTALL,
)
_BUSINESS_OVERVIEW_HEADING_RE = re.compile(
    r"^\s*事業概要\s*[|\uFF5C]\s*(?P<title>\S(?:.*\S)?)\s*$",
    re.DOTALL,
)
_HEADING_TAIL_SPLIT_RE = re.compile(r"\s+(?=(?:\*[0-9\uFF10-\uFF19]+|注[0-9\uFF10-\uFF19]+))")
_BUSINESS_OVERVIEW_TITLE_SPLIT_RE = re.compile(r"^(?P<title>.{1,80}?事業)\s+(?P<tail>.+)$", re.DOTALL)
_FOOTER_BRAND_RE = re.compile(r"^\s*kubell\s*$", re.IGNORECASE)
_DECK_FOOTNOTE_LINE_RE = re.compile(r"^\s*\*")
_CIRCLED_NUMBER_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]")
_WIDE_DIGIT_TRANSLATION = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
    }
)


def _has_unclosed_table_markup(text: str) -> bool:
    normalized = html_unescape(str(text or ""))
    open_count = len(_RAW_TABLE_OPEN_RE.findall(normalized))
    close_count = len(_RAW_TABLE_CLOSE_RE.findall(normalized))
    return open_count > close_count


def _find_last_unclosed_table_start(text: str) -> int | None:
    content = str(text or "")
    raw_opens = [m.start() for m in _RAW_TABLE_OPEN_RE.finditer(content)]
    raw_closes = [m.start() for m in _RAW_TABLE_CLOSE_RE.finditer(content)]
    if raw_opens:
        last_open = raw_opens[-1]
        last_close = raw_closes[-1] if raw_closes else -1
        if last_open > last_close:
            return last_open

    escaped_opens = [m.start() for m in _ESCAPED_TABLE_OPEN_RE.finditer(content)]
    escaped_closes = [m.start() for m in _ESCAPED_TABLE_CLOSE_RE.finditer(content)]
    if not escaped_opens:
        return None
    last_open = escaped_opens[-1]
    last_close = escaped_closes[-1] if escaped_closes else -1
    if last_open > last_close:
        return last_open
    return None


def _iter_complete_table_spans(text: str) -> list[tuple[int, int]]:
    content = str(text or "")
    raw_spans = [(m.start(), m.end()) for m in _RAW_COMPLETE_TABLE_RE.finditer(content)]
    if raw_spans:
        return raw_spans
    return [(m.start(), m.end()) for m in _ESCAPED_COMPLETE_TABLE_RE.finditer(content)]


def _compact_markup_for_prefix_match(text: str) -> str:
    normalized = html_unescape(str(text or ""))
    return re.sub(r"\s+", "", normalized)


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _extract_best_complete_table_tail(
    *,
    text: str,
    anchor_unclosed_table: str,
) -> str | None:
    content = str(text or "")
    spans = _iter_complete_table_spans(content)
    if not spans:
        return None
    if len(spans) == 1:
        return content[spans[0][0] :].strip()

    anchor_compact = _compact_markup_for_prefix_match(anchor_unclosed_table)
    best_start = spans[0][0]
    best_prefix_len = -1

    for start, end in spans:
        candidate_compact = _compact_markup_for_prefix_match(content[start:end])
        if not candidate_compact:
            continue
        prefix_len = _common_prefix_length(anchor_compact, candidate_compact)
        if prefix_len > best_prefix_len:
            best_prefix_len = prefix_len
            best_start = start

    return content[best_start:].strip()


def _repair_truncated_ocr_tail_with_secondary(
    *,
    ocr_text: str,
    secondary_ocr_text: str,
) -> str:
    if not _has_unclosed_table_markup(ocr_text):
        return ocr_text

    cut_index = _find_last_unclosed_table_start(ocr_text)
    if cut_index is None:
        return ocr_text
    prefix = ocr_text[:cut_index].rstrip()
    if not prefix:
        return ocr_text

    secondary_tail = _extract_best_complete_table_tail(
        text=secondary_ocr_text,
        anchor_unclosed_table=ocr_text[cut_index:],
    )
    if not secondary_tail:
        return ocr_text

    repaired = f"{prefix}\n\n{secondary_tail}".strip()
    if _has_unclosed_table_markup(repaired):
        return ocr_text
    if len(repaired) <= len(prefix):
        return ocr_text
    return repaired


def _force_close_unclosed_table_markup(text: str) -> str:
    content = str(text or "")
    if not content or not _has_unclosed_table_markup(content):
        return content

    stack: list[str] = []
    for match in _RAW_TAG_RE.finditer(content):
        is_close = str(match.group(1) or "") == "/"
        name = str(match.group(2) or "").lower()
        if name not in _TABLE_TAGS:
            continue
        if not is_close:
            stack.append(name)
            continue
        while stack:
            current = stack.pop()
            if current == name:
                break

    if not stack:
        return content

    suffix = "".join(f"</{name}>" for name in reversed(stack))
    if not suffix:
        return content
    return content.rstrip() + "\n" + suffix


def _sanitize_ocr_markdown_text(text: str) -> str:
    content = str(text or "")
    if not content:
        return ""

    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines: list[str] = []
    for raw_line in lines:
        line = html_unescape(raw_line)
        line = _MARKDOWN_IMAGE_RE.sub("", line)
        line = line.replace("**", "").replace("__", "")

        stripped = line.strip()
        if _is_dangling_non_table_tag_fragment(stripped):
            continue
        if stripped.lstrip().startswith("```"):
            continue
        if _MARKDOWN_HR_RE.fullmatch(stripped):
            continue
        if _is_llm_note_artifact_line(stripped):
            continue

        line = _strip_non_table_html_tags(line)
        if "\n" not in line:
            cleaned_lines.append(line.rstrip())
            continue
        for part in line.split("\n"):
            cleaned_lines.append(part.rstrip())

    trimmed_tail = _trim_repeated_tail_line_patterns(cleaned_lines)
    normalized = _join_lines_with_compact_blank_runs(trimmed_tail)
    return normalized


def _is_llm_note_artifact_line(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    normalized = stripped.strip("*").strip()
    lowered = normalized.lower()
    if _TRANSCRIPTION_PREFIX_RE.match(normalized):
        return True
    if not _NOTE_PREFIX_RE.match(normalized):
        return False
    if "image" in lowered:
        return True
    return any(hint in lowered for hint in _NOTE_ARTIFACT_HINTS)


def _strip_non_table_html_tags(line: str) -> str:
    raw = str(line or "")
    if "<" not in raw or ">" not in raw:
        return raw
    lowered = raw.lower()
    if any(tag in lowered for tag in _TABLE_TAG_HINTS):
        return raw

    without_br = _HTML_BR_RE.sub("\n", raw)
    without_comments = re.sub(r"<!--.*?-->", "", without_br)
    return _HTML_TAG_RE.sub("", without_comments)


def _is_dangling_non_table_tag_fragment(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped or not stripped.startswith("<") or ">" in stripped:
        return False
    match = _TAG_FRAGMENT_START_RE.match(stripped)
    if match is None:
        return False
    tag_name = str(match.group(1) or "").lower()
    return tag_name not in _TABLE_TAGS


def _trim_repeated_tail_line_patterns(
    lines: list[str],
    *,
    max_window: int = 3,
    min_repeats: int = 5,
    keep_repeats: int = 2,
) -> list[str]:
    if not lines:
        return []

    out = list(lines)
    for window in range(1, max_window + 1):
        if len(out) < (window * min_repeats):
            continue
        pattern = out[-window:]
        if not pattern or any(not _is_tail_pattern_candidate(line) for line in pattern):
            continue

        repeats = 1
        cursor = len(out) - (2 * window)
        while cursor >= 0 and out[cursor : cursor + window] == pattern:
            repeats += 1
            cursor -= window

        if repeats < min_repeats:
            continue

        keep_start = len(out) - (repeats * window)
        keep_len = keep_repeats * window
        return out[: keep_start + keep_len]

    return out


def _is_tail_pattern_candidate(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    return len(stripped) <= 48


def _join_lines_with_compact_blank_runs(lines: list[str]) -> str:
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run <= 1:
                out.append("")
            continue
        blank_run = 0
        out.append(line.rstrip())
    return "\n".join(out).strip()


def _decode_base64_image_payload(image_base64: str) -> bytes:
    raw = str(image_base64 or "").strip()
    if not raw:
        return b""
    if raw.startswith("data:"):
        comma_index = raw.find(",")
        if comma_index >= 0:
            raw = raw[comma_index + 1 :].strip()
    compact = "".join(raw.split())
    if not compact:
        return b""
    try:
        return base64.b64decode(compact, validate=True)
    except (ValueError, binascii.Error):
        return b""


def _crop_image_base64_to_bottom(image_base64: str, *, top_ratio: float = 0.45) -> str | None:
    payload = _decode_base64_image_payload(image_base64)
    if not payload:
        return None

    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None

    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            if width <= 1 or height <= 1:
                return None
            top = int(height * float(top_ratio))
            if top < 0:
                top = 0
            if top >= height:
                top = max(0, height // 2)
            cropped = image.crop((0, top, width, height))
            out = io.BytesIO()
            cropped.save(out, format="PNG")
            return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _encode_image_to_base64_png(image: object) -> str:
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow is required for PNG encoding.") from exc

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    out = io.BytesIO()
    image.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def _downscale_image_base64_to_max_edge(image_base64: str, *, max_edge: int) -> str | None:
    target_edge = int(max_edge)
    if target_edge <= 0:
        return None

    payload = _decode_base64_image_payload(image_base64)
    if not payload:
        return None

    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None

    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            if width <= 1 or height <= 1:
                return None

            longest = max(width, height)
            if longest <= target_edge:
                return None

            ratio = float(target_edge) / float(longest)
            resized_width = max(1, int(round(float(width) * ratio)))
            resized_height = max(1, int(round(float(height) * ratio)))
            resampling_enum = getattr(Image, "Resampling", None)
            bilinear_name = "BILINEAR"
            if resampling_enum is not None:
                resampling = getattr(resampling_enum, bilinear_name)
            else:
                resampling = getattr(Image, bilinear_name)
            resized = image.resize((resized_width, resized_height), resample=resampling)
            out = io.BytesIO()
            resized.save(out, format="PNG")
            return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _estimate_page_table_likelihood(pymupdf_words: list[Word]) -> float:
    if not pymupdf_words:
        return 0.0
    try:
        return float(estimate_table_likelihood(pymupdf_words))
    except Exception:  # noqa: BLE001
        return 0.0


def _build_primary_ocr_plan(
    *,
    image_base64: str,
    settings: Settings,
    pymupdf_words: list[Word],
) -> _PrimaryOcrPlan:
    table_likelihood = _estimate_page_table_likelihood(pymupdf_words)
    threshold = max(0.0, float(settings.lighton_fast_table_likelihood_threshold))
    is_table_like = table_likelihood >= threshold

    if not settings.lighton_fast_pass:
        return _PrimaryOcrPlan(
            image_base64=image_base64,
            max_tokens=max(1, int(settings.lighton_max_tokens)),
            table_likelihood=table_likelihood,
        )

    base_budget = (
        int(settings.lighton_fast_max_tokens_table)
        if is_table_like
        else int(settings.lighton_fast_max_tokens_text)
    )
    max_tokens = max(1, min(base_budget, int(settings.lighton_max_tokens)))

    selected_image = image_base64
    non_table_target_edge = int(settings.lighton_fast_non_table_max_edge)
    if not is_table_like and non_table_target_edge > 0:
        downscaled = _downscale_image_base64_to_max_edge(
            image_base64,
            max_edge=non_table_target_edge,
        )
        if downscaled:
            selected_image = downscaled

    return _PrimaryOcrPlan(
        image_base64=selected_image,
        max_tokens=max_tokens,
        table_likelihood=table_likelihood,
    )


def _apply_table_fallback_with_pymupdf(
    *,
    page: Page,
    pymupdf_text: str,
    pymupdf_words: list[Word],
) -> Page:
    if not pymupdf_words:
        return page
    fallback_table = _best_table_from_pymupdf_words(pymupdf_words)
    if fallback_table is None:
        return page
    return _replace_truncated_table_with_pymupdf(
        page=page,
        fallback_table=fallback_table,
        pymupdf_text=pymupdf_text,
    )


def _to_ascii_digits(text: str) -> str:
    return str(text or "").translate(_WIDE_DIGIT_TRANSLATION)


def _parse_ascii_int(text: str) -> int | None:
    normalized = _to_ascii_digits(text).strip()
    if not normalized.isdigit():
        return None
    try:
        return int(normalized)
    except Exception:
        return None


def _is_page_number_only_text(text: str, *, page_number: int) -> bool:
    normalized = normalize_extracted_text(text).strip()
    if not normalized:
        return False
    match = _PAGE_NUMBER_ONLY_RE.fullmatch(normalized)
    if match is None:
        return False
    parsed = _parse_ascii_int(str(match.group(1) or ""))
    if parsed is None:
        return False
    return parsed == int(page_number)


def _strip_leading_page_number_prefix(text: str, *, page_number: int) -> str:
    normalized = normalize_extracted_text(text).strip()
    if not normalized:
        return ""
    match = _LEADING_PAGE_NUMBER_PREFIX_RE.match(normalized)
    if match is None:
        return normalized
    parsed = _parse_ascii_int(str(match.group(1) or ""))
    if parsed != int(page_number):
        return normalized
    remainder = str(match.group(2) or "").lstrip()
    if remainder.startswith("|") or remainder.startswith("｜"):
        # Preserve section headings like "01 | ...".
        return normalized
    return remainder


def _extract_promoted_heading_and_tail(
    text: str,
    *,
    page_number: int,
) -> tuple[str, str | None] | None:
    stripped = _strip_leading_page_number_prefix(text, page_number=page_number)
    if not stripped:
        return None

    section_match = _SECTION_BAR_HEADING_RE.match(stripped)
    if section_match is not None:
        section = str(section_match.group("section") or "").strip()
        title = str(section_match.group("title") or "").strip()
        if not section or not title:
            return None
        return f"{section} | {title}", None

    overview_match = _BUSINESS_OVERVIEW_HEADING_RE.match(stripped)
    if overview_match is None:
        return None

    raw_title = str(overview_match.group("title") or "").strip()
    if not raw_title:
        return None

    title = raw_title
    tail: str | None = None
    split_match = _HEADING_TAIL_SPLIT_RE.search(raw_title)
    if split_match is not None:
        title = raw_title[: split_match.start()].strip()
        tail = raw_title[split_match.start() :].strip() or None
    else:
        title_split_match = _BUSINESS_OVERVIEW_TITLE_SPLIT_RE.match(raw_title)
        if title_split_match is not None:
            candidate_title = str(title_split_match.group("title") or "").strip()
            candidate_tail = str(title_split_match.group("tail") or "").strip()
            if candidate_tail and (
                "。" in candidate_tail or "、" in candidate_tail or len(candidate_tail) >= 24
            ):
                title = candidate_title
                tail = candidate_tail or None

    if not title:
        return None
    if len(title) > 80 and tail is None:
        # Avoid converting body-merged long paragraphs into headings.
        return None
    return f"事業概要｜{title}", tail


def _strip_page_number_prefix_on_top_text_blocks(page: Page) -> Page:
    updated_blocks = list(page.blocks)
    changed = False
    text_block_seen = 0

    for index, block in enumerate(updated_blocks):
        if isinstance(block, Table):
            continue
        if not isinstance(block, (Paragraph, Heading, Unknown)):
            continue
        text_block_seen += 1
        if text_block_seen > 2:
            break
        source_text = str(getattr(block, "text", "")).strip()
        if not source_text:
            continue
        stripped = _strip_leading_page_number_prefix(
            source_text,
            page_number=page.page_number,
        )
        if stripped == source_text:
            continue
        if isinstance(block, Paragraph):
            updated_blocks[index] = Paragraph(text=stripped)
        elif isinstance(block, Heading):
            updated_blocks[index] = Heading(level=block.level, text=stripped)
        else:
            updated_blocks[index] = Unknown(text=stripped)
        changed = True

    if not changed:
        return page
    return Page(page_number=page.page_number, blocks=tuple(updated_blocks))


def _remove_page_number_and_footer_blocks(page: Page) -> Page:
    blocks = list(page.blocks)
    if not blocks:
        return page

    next_blocks: list[Block] = []
    total = len(blocks)
    changed = False

    for index, block in enumerate(blocks):
        if not isinstance(block, (Paragraph, Heading, Unknown)):
            next_blocks.append(block)
            continue
        text = str(getattr(block, "text", "")).strip()
        if not text:
            next_blocks.append(block)
            continue
        if _is_page_number_only_text(text, page_number=page.page_number):
            changed = True
            continue
        if _FOOTER_BRAND_RE.fullmatch(text) and (index <= 1 or index >= (total - 2)):
            changed = True
            continue
        next_blocks.append(block)

    if not changed:
        return page
    return Page(page_number=page.page_number, blocks=tuple(next_blocks))


def _promote_rule_based_headings(page: Page) -> Page:
    if not page.blocks:
        return page

    next_blocks: list[Block] = []
    changed = False
    text_block_seen = 0

    for block in page.blocks:
        if not isinstance(block, (Paragraph, Unknown)):
            next_blocks.append(block)
            continue

        text_block_seen += 1
        source_text = str(getattr(block, "text", "")).strip()
        if not source_text:
            next_blocks.append(block)
            continue

        promoted = (
            _extract_promoted_heading_and_tail(
                source_text,
                page_number=page.page_number,
            )
            if text_block_seen <= 3
            else None
        )
        if promoted is None:
            next_blocks.append(block)
            continue

        heading_text, tail_text = promoted
        next_blocks.append(Heading(level=2, text=heading_text))
        if tail_text:
            next_blocks.append(Paragraph(text=tail_text))
        changed = True

    if not changed:
        return page
    return Page(page_number=page.page_number, blocks=tuple(next_blocks))


def _is_short_heading_fragment_line(text: str) -> bool:
    normalized = normalize_extracted_text(text).strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    if "br />" in lowered or "br/>" in lowered or lowered.endswith("<br"):
        return False
    if _DECK_FOOTNOTE_LINE_RE.match(normalized):
        return False
    if normalized.startswith(("・", "-", "•", "※")):
        return False
    compact = _compact_text_for_match(normalized)
    if not compact:
        return False
    if len(compact) > 20:
        return False
    if re.search(r"[0-9\uFF10-\uFF19]{3,}", normalized):
        return False
    return any(_is_japanese_char(ch) or ch.isalpha() for ch in compact)


def _find_short_heading_cluster_end(lines: list[str], *, start: int) -> int | None:
    if start < 0 or start >= len(lines):
        return None
    if (start + 1) >= len(lines):
        return None
    first_two = lines[start : start + 2]
    if not all(_is_short_heading_fragment_line(line) for line in first_two):
        return None
    if "、" not in first_two[0]:
        return None

    end = start + 2
    max_end = min(len(lines), start + 4)
    while end < max_end and _is_short_heading_fragment_line(lines[end]):
        end += 1

    merged = "".join(lines[start:end])
    compact_len = len(_compact_text_for_match(merged))
    if compact_len < 8 or compact_len > 48:
        return None
    return end


def _split_compound_issue_paragraph_into_blocks(text: str) -> list[Block] | None:
    raw_lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [normalize_extracted_text(line).strip() for line in raw_lines if line.strip()]
    if len(lines) < 6:
        return None
    if not any(_CIRCLED_NUMBER_RE.search(line) for line in lines):
        return None
    if not any(len(_compact_text_for_match(line)) >= 24 for line in lines):
        return None

    out: list[Block] = []
    index = 0

    first_line = lines[0]
    first_compact_len = len(_compact_text_for_match(first_line))
    if _CIRCLED_NUMBER_RE.search(first_line) and first_compact_len <= 40:
        out.append(Heading(level=2, text=first_line))
        index = 1

    footnote_lines: list[str] = []
    while index < len(lines) and _DECK_FOOTNOTE_LINE_RE.match(lines[index]):
        footnote_lines.append(lines[index])
        index += 1
    if footnote_lines:
        out.append(Paragraph(text="\n".join(footnote_lines)))

    body_lines: list[str] = []
    while index < len(lines):
        cluster_end = _find_short_heading_cluster_end(lines, start=index)
        if cluster_end is not None:
            break
        body_lines.append(lines[index])
        index += 1
    if body_lines:
        out.append(Paragraph(text="\n".join(body_lines)))

    while index < len(lines):
        cluster_end = _find_short_heading_cluster_end(lines, start=index)
        if cluster_end is not None:
            out.append(Heading(level=2, text="".join(lines[index:cluster_end])))
            index = cluster_end
            continue
        out.append(Paragraph(text=lines[index]))
        index += 1

    if len(out) <= 1:
        return None
    return out


def _split_compound_issue_paragraph_blocks(page: Page) -> Page:
    if not page.blocks:
        return page

    next_blocks: list[Block] = []
    changed = False
    for block in page.blocks:
        if not isinstance(block, (Paragraph, Unknown)):
            next_blocks.append(block)
            continue
        source_text = str(getattr(block, "text", "")).strip()
        if not source_text:
            next_blocks.append(block)
            continue
        replaced = _split_compound_issue_paragraph_into_blocks(source_text)
        if replaced is None:
            next_blocks.append(block)
            continue
        next_blocks.extend(replaced)
        changed = True

    if not changed:
        return page
    return Page(page_number=page.page_number, blocks=tuple(next_blocks))


def _apply_page_cleanup_rules(page: Page) -> Page:
    page = _strip_page_number_prefix_on_top_text_blocks(page)
    page = _promote_rule_based_headings(page)
    page = _split_compound_issue_paragraph_blocks(page)
    return _remove_page_number_and_footer_blocks(page)


def _apply_page_postprocess(
    *,
    page: Page,
    settings: Settings,
    pymupdf_text: str,
    pymupdf_words: list[Word],
    table_likelihood: float,
) -> Page:
    if not settings.lighton_fast_pass:
        mode = "full"
    else:
        mode = str(settings.lighton_fast_postprocess_mode or "full").strip().lower()

    if mode == "off":
        return _apply_page_cleanup_rules(page)

    if mode == "light":
        page = _correct_text_blocks_locally_with_pymupdf(
            page=page,
            pymupdf_text=pymupdf_text,
            min_match_score=0.45,
            max_change_ratio=0.18,
            max_changes=12,
        )
        threshold = max(0.0, float(settings.lighton_fast_table_likelihood_threshold))
        has_table_block = any(isinstance(block, Table) for block in page.blocks)
        if table_likelihood < threshold and not has_table_block:
            return _apply_page_cleanup_rules(page)
        page = _replace_table_preface_with_pymupdf(page=page, pymupdf_text=pymupdf_text)
        page = _correct_table_blocks_locally_with_pymupdf(
            page=page,
            pymupdf_text=pymupdf_text,
            pymupdf_words=pymupdf_words,
        )
        page = _apply_table_fallback_with_pymupdf(
            page=page,
            pymupdf_text=pymupdf_text,
            pymupdf_words=pymupdf_words,
        )
        page = _correct_table_blocks_locally_with_pymupdf(
            page=page,
            pymupdf_text=pymupdf_text,
            pymupdf_words=pymupdf_words,
        )
        return _apply_page_cleanup_rules(page)

    page = _replace_table_preface_with_pymupdf(
        page=page,
        pymupdf_text=pymupdf_text,
    )
    page = _correct_text_blocks_locally_with_pymupdf(
        page=page,
        pymupdf_text=pymupdf_text,
    )
    page = _correct_table_blocks_locally_with_pymupdf(
        page=page,
        pymupdf_text=pymupdf_text,
        pymupdf_words=pymupdf_words,
    )
    page = _apply_table_fallback_with_pymupdf(
        page=page,
        pymupdf_text=pymupdf_text,
        pymupdf_words=pymupdf_words,
    )
    page = _correct_table_blocks_locally_with_pymupdf(
        page=page,
        pymupdf_text=pymupdf_text,
        pymupdf_words=pymupdf_words,
    )
    return _apply_page_cleanup_rules(page)


def _run_ocr_pass(
    *,
    image_base64: str,
    settings: Settings,
    pymupdf_text: str,
    pymupdf_words: list[Word],
    merge_policy: str,
    max_tokens: int | None = None,
) -> _OcrPassOutcome:
    ocr_settings = settings
    if max_tokens is not None:
        bounded_tokens = max(1, int(max_tokens))
        if bounded_tokens != int(settings.lighton_max_tokens):
            ocr_settings = replace(settings, lighton_max_tokens=bounded_tokens)

    ocr_text = lighton_ocr.ocr_image_base64(image_base64, settings=ocr_settings)
    raw_has_unclosed_table = _has_unclosed_table_markup(ocr_text)
    if settings.lighton_secondary_table_repair and raw_has_unclosed_table:
        cropped_image = _crop_image_base64_to_bottom(image_base64, top_ratio=0.45)
        if cropped_image:
            try:
                secondary_ocr_text = lighton_ocr.ocr_image_base64(
                    cropped_image,
                    settings=ocr_settings,
                )
                ocr_text = _repair_truncated_ocr_tail_with_secondary(
                    ocr_text=ocr_text,
                    secondary_ocr_text=secondary_ocr_text,
                )
            except Exception:  # noqa: BLE001
                pass
    ocr_text = _force_close_unclosed_table_markup(ocr_text)

    merged_text = _merge_text_with_pymupdf_fallback(
        ocr_text=ocr_text,
        pymupdf_text=pymupdf_text,
        policy=merge_policy,
    )
    if pymupdf_words:
        try:
            merged_table_text, table_stats = merge_markdown_tables_with_pymupdf_words(
                merged_text,
                pymupdf_words,
            )
            if table_stats.applied:
                merged_text = merged_table_text
        except Exception:  # noqa: BLE001
            pass

    return _OcrPassOutcome(
        ocr_text=ocr_text,
        merged_text=merged_text,
        merged_quality=score_text_quality(merged_text).score,
        has_unclosed_table=raw_has_unclosed_table,
    )


def _should_retry_high_res(
    *,
    primary: _OcrPassOutcome,
    pymupdf_text: str,
    retry_min_quality: float,
    retry_quality_gap: float,
    retry_min_pym_text_len: int,
) -> bool:
    if primary.has_unclosed_table:
        return True
    if primary.merged_quality < float(retry_min_quality):
        return True

    pym_quality = score_text_quality(pymupdf_text).score
    return (
        pym_quality >= 0.55
        and (pym_quality - primary.merged_quality) >= float(retry_quality_gap)
        and len(pymupdf_text.strip()) >= int(retry_min_pym_text_len)
    )


def _select_best_pass_outcome(
    *,
    primary: _OcrPassOutcome,
    retry: _OcrPassOutcome | None,
) -> _OcrPassOutcome:
    if retry is None:
        return primary
    if retry.merged_quality >= (primary.merged_quality + 0.03):
        return retry
    if primary.has_unclosed_table and not retry.has_unclosed_table:
        return retry
    return primary


def _merge_text_with_pymupdf_fallback(
    *,
    ocr_text: str,
    pymupdf_text: str,
    policy: str,
) -> str:
    merged_text, _merge_stats = merge_ocr_with_pymupdf(
        ocr_text,
        pymupdf_text,
        policy=policy,
    )

    merged_quality = score_text_quality(merged_text).score
    pym_quality = score_text_quality(pymupdf_text).score

    if pym_quality >= 0.55 and (pym_quality - merged_quality) >= 0.15:
        aggressive_text, _aggressive_stats = merge_ocr_with_pymupdf(
            ocr_text,
            pymupdf_text,
            policy="aggressive",
            max_changed_ratio=0.45,
        )
        aggressive_quality = score_text_quality(aggressive_text).score
        if aggressive_quality > merged_quality:
            merged_text = aggressive_text
            merged_quality = aggressive_quality

        has_table_markup = ("<table" in ocr_text.lower()) or ("&lt;table" in ocr_text.lower())
        if (
            (pym_quality - merged_quality) >= 0.20
            and len(pymupdf_text.strip()) >= 80
            and not has_table_markup
        ):
            merged_text = pymupdf_text.strip()

    return merged_text


def _contains_table_markup(text: str) -> bool:
    lowered = str(text or "").lower()
    return ("<table" in lowered) or ("&lt;table" in lowered)


def _line_repeat_coverage_ratio(text: str) -> float:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return 0.0

    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1

    repeated = sum(count for count in counts.values() if count >= 2)
    return float(repeated) / float(max(1, len(lines)))


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(tokenize_by_char_class(normalize_extracted_text(left)))
    right_tokens = set(tokenize_by_char_class(normalize_extracted_text(right)))
    if not right_tokens:
        return 0.0
    return float(len(left_tokens & right_tokens)) / float(max(1, len(right_tokens)))


def _text_similarity_ratio(left: str, right: str) -> float:
    left_norm = normalize_extracted_text(left).strip()
    right_norm = normalize_extracted_text(right).strip()
    if not left_norm or not right_norm:
        return 0.0
    try:
        return float(difflib.SequenceMatcher(a=left_norm, b=right_norm, autojunk=False).ratio())
    except Exception:  # noqa: BLE001
        return 0.0


def _should_prefer_pymupdf_text(
    *,
    merged_text: str,
    ocr_text: str,
    pymupdf_text: str,
    pymupdf_words: list[Word],
    table_likelihood: float,
    merged_quality: float,
    fallback_mode: str,
) -> bool:
    mode = str(fallback_mode or "repeat").strip().lower()
    if mode == "off":
        return False
    if mode not in {"repeat", "aggressive"}:
        mode = "repeat"

    pymupdf_norm = normalize_extracted_text(pymupdf_text).strip()
    if not pymupdf_norm:
        return False
    if len(pymupdf_norm) < 80:
        return False

    merged_norm = normalize_extracted_text(merged_text).strip()
    if not merged_norm:
        return True

    similarity = _text_similarity_ratio(merged_norm, pymupdf_norm)
    token_overlap = _token_overlap_ratio(merged_norm, pymupdf_norm)
    repeated_line_ratio = _line_repeat_coverage_ratio(merged_norm)
    has_table_markup = _contains_table_markup(ocr_text) or _contains_table_markup(merged_text)

    if mode == "repeat":
        if has_table_markup and table_likelihood >= 0.75:
            if len(pymupdf_words) < 20:
                return False
            if repeated_line_ratio >= 0.80 and token_overlap < 0.30:
                return True
            if similarity < 0.12 and token_overlap < 0.16 and merged_quality < 0.20:
                return True
            return False

        if repeated_line_ratio >= 0.60 and token_overlap < 0.45:
            return True
        if similarity < 0.10 and token_overlap < 0.12 and merged_quality < 0.20:
            return True
        return False

    if len(pymupdf_words) < 20 and len(pymupdf_norm) < 160:
        if not (
            (similarity < 0.20 and token_overlap < 0.25)
            or (repeated_line_ratio >= 0.60 and token_overlap < 0.45)
        ):
            return False

    if has_table_markup:
        if len(pymupdf_words) < 20:
            return False
        if table_likelihood >= 0.75:
            if similarity < 0.20 and token_overlap < 0.20:
                return True
            if repeated_line_ratio >= 0.50 and token_overlap < 0.30:
                return True
            return False

    if similarity < 0.25 and token_overlap < 0.30:
        return True

    if repeated_line_ratio >= 0.35 and token_overlap < 0.45:
        return True

    return False


def _best_table_from_pymupdf_words(words: list[Word]) -> Table | None:
    if not words:
        return None
    if estimate_table_likelihood(words) < 0.75:
        return None

    best: tuple[float, float, float, float, int] | None = None
    best_table: Table | None = None
    for column_count in range(6, 1, -1):
        result = build_table_grid(words, column_count=column_count)
        if not result.ok or result.grid is None:
            continue

        rows = result.grid.rows
        if len(rows) < 8:
            continue
        total_cells = max(1, len(rows) * max(1, column_count))
        empty_cells = sum(1 for row in rows for cell in row[:column_count] if not str(cell).strip())
        empty_ratio = empty_cells / total_cells
        collision_ratio = float(result.grid.collision_count) / max(
            1.0,
            float(result.grid.group_count),
        )
        score = ((1.0 - collision_ratio) * 0.6) + ((1.0 - empty_ratio) * 0.4)
        signature = (
            score,
            float(result.confidence),
            -empty_ratio,
            -collision_ratio,
            int(column_count),
        )

        grid = tuple(tuple(str(cell) for cell in row) for row in rows)
        text = "\n".join("\t".join(row) for row in grid)
        candidate = Table(
            text=text,
            grid=grid,
            cells=result.grid.cells,
        )

        if best is None or signature > best:
            best = signature
            best_table = candidate

    if best is None or best_table is None:
        return None
    if best[1] < 0.70:
        return None
    return best_table


def _replace_truncated_table_with_pymupdf(
    *,
    page: Page,
    fallback_table: Table,
    pymupdf_text: str,
) -> Page:
    table_indices: list[int] = []
    table_row_counts: list[int] = []
    for index, block in enumerate(page.blocks):
        if not isinstance(block, Table):
            continue
        if block.grid is None:
            continue
        table_indices.append(index)
        table_row_counts.append(len(block.grid))

    if not table_indices:
        preface_markdown = _extract_pymupdf_preface_for_table_page(pymupdf_text)
        preface_blocks: tuple[Heading | Paragraph, ...] = ()
        if preface_markdown:
            preface_page = page_from_ocr_markdown(
                page_number=page.page_number,
                markdown=preface_markdown,
            )
            preface_blocks = tuple(
                block for block in preface_page.blocks if isinstance(block, (Heading, Paragraph))
            )

        if preface_blocks:
            return Page(
                page_number=page.page_number,
                blocks=preface_blocks + (fallback_table,),
            )
        return Page(
            page_number=page.page_number,
            blocks=tuple(page.blocks) + (fallback_table,),
        )

    fallback_rows = len(fallback_table.grid or ())
    if fallback_rows <= 0:
        return page

    current_max_rows = max(table_row_counts, default=0)
    min_required = max(current_max_rows + 5, int(current_max_rows * 1.25))
    if fallback_rows < min_required:
        return page

    preface_markdown = _extract_pymupdf_preface_for_table_page(pymupdf_text)
    if preface_markdown:
        preface_page = page_from_ocr_markdown(
            page_number=page.page_number,
            markdown=preface_markdown,
        )
        if preface_page.blocks:
            return Page(
                page_number=page.page_number,
                blocks=tuple(preface_page.blocks) + (fallback_table,),
            )

    largest_index = table_indices[table_row_counts.index(current_max_rows)]
    next_blocks = list(page.blocks)
    next_blocks[largest_index] = fallback_table
    return Page(page_number=page.page_number, blocks=tuple(next_blocks))


def _compact_text_for_match(text: str) -> str:
    return re.sub(r"\s+", "", normalize_extracted_text(text))


def _is_japanese_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        (0x3040 <= code <= 0x309F)  # hiragana
        or (0x30A0 <= code <= 0x30FF)  # katakana
        or (0xFF66 <= code <= 0xFF9D)  # halfwidth katakana
        or (0x4E00 <= code <= 0x9FFF)  # kanji
    )


def _build_joined_line_windows(
    lines: list[str],
    *,
    max_window: int = 8,
    separator: str = " ",
) -> list[str]:
    if not lines:
        return []
    windows: list[str] = []
    seen: set[str] = set()
    total = len(lines)
    for start in range(total):
        for width in range(1, max_window + 1):
            end = start + width
            if end > total:
                break
            candidate = separator.join(lines[start:end]).strip()
            if not candidate or candidate in seen:
                continue
            windows.append(candidate)
            seen.add(candidate)
    return windows


def _find_best_reference_text(
    source_text: str,
    candidates: list[str],
) -> tuple[str | None, float]:
    source_compact = _compact_text_for_match(source_text)
    if not source_compact:
        return None, 0.0

    best_text: str | None = None
    best_score = 0.0
    scored: list[tuple[str, float, int]] = []
    for candidate in candidates:
        candidate_compact = _compact_text_for_match(candidate)
        if not candidate_compact:
            continue
        score = difflib.SequenceMatcher(
            a=source_compact[:1200],
            b=candidate_compact[:1200],
            autojunk=False,
        ).ratio()
        scored.append((candidate, score, len(candidate_compact)))
        if score > best_score:
            best_score = score
            best_text = candidate

    if best_text is None or not scored:
        return best_text, best_score

    # Prefer references with compact length close to the OCR block when score is close.
    source_len = len(source_compact)
    threshold = max(0.0, best_score - 0.04)
    selected_text = best_text
    selected_score = best_score
    selected_compact = _compact_text_for_match(best_text)
    selected_len_delta = abs(len(selected_compact) - source_len)

    for candidate, score, compact_len in scored:
        if score < threshold:
            continue
        len_delta = abs(compact_len - source_len)
        if len_delta < selected_len_delta:
            selected_text = candidate
            selected_score = score
            selected_len_delta = len_delta
            selected_compact = _compact_text_for_match(candidate)
            continue
        if len_delta != selected_len_delta:
            continue
        if score > selected_score:
            selected_text = candidate
            selected_score = score
            selected_compact = _compact_text_for_match(candidate)
            continue
        if abs(score - selected_score) <= 0.01 and compact_len > len(selected_compact):
            selected_text = candidate
            selected_score = score
            selected_compact = _compact_text_for_match(candidate)

    return selected_text, selected_score


def _apply_local_char_corrections(
    source_text: str,
    reference_text: str,
    *,
    max_change_ratio: float = 0.18,
    max_changes: int = 12,
) -> str:
    source = str(source_text or "")
    reference = str(reference_text or "")
    if not source or not reference:
        return source

    out_chars: list[str] = []
    changed_count = 0
    matcher = difflib.SequenceMatcher(a=source, b=reference, autojunk=False)
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == "equal":
            out_chars.append(source[a1:a2])
            continue
        if tag == "replace":
            left = source[a1:a2]
            right = reference[b1:b2]
            if len(left) == len(right):
                corrected: list[str] = []
                for src_ch, ref_ch in zip(left, right, strict=False):
                    if src_ch == ref_ch:
                        corrected.append(src_ch)
                        continue
                    if _is_japanese_char(src_ch) or _is_japanese_char(ref_ch):
                        corrected.append(ref_ch)
                        changed_count += 1
                    else:
                        corrected.append(src_ch)
                out_chars.append("".join(corrected))
                continue
            out_chars.append(left)
            continue
        if tag == "delete":
            out_chars.append(source[a1:a2])
            continue
        if tag == "insert":
            continue

    non_ws_len = len(re.sub(r"\s+", "", source))
    allowed = min(max_changes, max(2, int(non_ws_len * max_change_ratio)))
    if changed_count <= 0 or changed_count > allowed:
        return source
    corrected_text = "".join(out_chars)
    similarity = difflib.SequenceMatcher(
        a=source[:3000],
        b=corrected_text[:3000],
        autojunk=False,
    ).ratio()
    if similarity < 0.85:
        return source
    return corrected_text


def _is_safe_text_block_reference_replace(
    *,
    source_text: str,
    reference_text: str,
    match_score: float,
) -> bool:
    source_norm = normalize_extracted_text(source_text).strip()
    reference_norm = normalize_extracted_text(reference_text).strip()
    if not source_norm or not reference_norm:
        return False
    if "\t" in source_norm or "\t" in reference_norm:
        return False
    if (
        "**" in reference_norm
        or "__" in reference_norm
        or "```" in reference_norm
        or "<" in reference_norm
        or ">" in reference_norm
    ):
        return False
    if re.search(r"(?m)^\s{0,3}#{1,6}\s", reference_norm):
        return False
    if re.search(r"(?m)^\s*\|.*\|\s*$", reference_norm):
        return False

    source_len = len(re.sub(r"\s+", "", source_norm))
    reference_len = len(re.sub(r"\s+", "", reference_norm))
    if source_len < 4 or reference_len < 4:
        return False

    length_ratio = float(reference_len) / float(max(1, source_len))
    token_overlap = _token_overlap_ratio(source_norm, reference_norm)
    text_similarity = _text_similarity_ratio(source_norm, reference_norm)

    if source_len < 10:
        if length_ratio < 0.50 or length_ratio > 1.80:
            return False
        if float(match_score) < 0.72:
            return False
        if token_overlap < 0.25:
            return False
        if text_similarity < 0.45:
            return False
        return True

    if source_len <= 64:
        if length_ratio < 0.75 or length_ratio > 1.45:
            return False
        if float(match_score) < 0.55:
            return False
        if token_overlap < 0.30:
            return False
        if text_similarity < 0.55:
            return False
        return True

    if length_ratio < 0.85 or length_ratio > 1.20:
        return False
    if float(match_score) < 0.72:
        return False
    if token_overlap < 0.55:
        return False
    if text_similarity < 0.68:
        return False
    if not any(_is_japanese_char(ch) for ch in source_norm):
        return False
    return True


def _replace_table_preface_with_pymupdf(
    *,
    page: Page,
    pymupdf_text: str,
) -> Page:
    first_table_index: int | None = None
    for index, block in enumerate(page.blocks):
        if isinstance(block, Table) and block.grid is not None:
            first_table_index = index
            break
    if first_table_index is None or first_table_index <= 0:
        return page

    preface_blocks = list(page.blocks[:first_table_index])
    if not preface_blocks:
        return page

    pym_preface_markdown = _extract_pymupdf_preface_for_table_page(pymupdf_text)
    if not pym_preface_markdown:
        return page

    pym_lines = [line.strip() for line in pym_preface_markdown.splitlines() if line.strip()]
    candidates = _build_joined_line_windows(pym_lines, max_window=8)
    if not candidates:
        return page

    changed_blocks = 0
    for index, original in enumerate(preface_blocks):
        if not isinstance(original, (Paragraph, Heading, Unknown)):
            continue
        source_text = str(getattr(original, "text", "")).strip()
        if not source_text:
            continue

        reference_text, score = _find_best_reference_text(source_text, candidates)
        if reference_text is None or score < 0.30:
            continue

        corrected = _apply_local_char_corrections(
            source_text,
            reference_text,
            max_change_ratio=0.18,
            max_changes=12,
        )
        if corrected == source_text and _is_safe_text_block_reference_replace(
            source_text=source_text,
            reference_text=reference_text,
            match_score=score,
        ):
            corrected = reference_text
        if corrected == source_text:
            continue

        if isinstance(original, Paragraph):
            preface_blocks[index] = Paragraph(text=corrected)
        elif isinstance(original, Heading):
            preface_blocks[index] = Heading(level=original.level, text=corrected)
        else:
            preface_blocks[index] = Unknown(text=corrected)
        changed_blocks += 1

    if changed_blocks <= 0:
        return page

    return Page(
        page_number=page.page_number,
        blocks=tuple(preface_blocks) + page.blocks[first_table_index:],
    )


def _correct_text_blocks_locally_with_pymupdf(
    *,
    page: Page,
    pymupdf_text: str,
    min_match_score: float = 0.32,
    max_change_ratio: float = 0.18,
    max_changes: int = 12,
) -> Page:
    pym_lines = [line.strip() for line in pymupdf_text.splitlines() if line.strip()]
    base_candidates = _build_joined_line_windows(pym_lines, max_window=8, separator=" ")
    if not base_candidates:
        return page
    long_candidates = _build_joined_line_windows(
        pym_lines,
        max_window=20,
        separator="\n",
    )

    updated_blocks = list(page.blocks)
    changed_count = 0
    for index, block in enumerate(updated_blocks):
        if not isinstance(block, (Paragraph, Heading, Unknown)):
            continue
        source_text = str(getattr(block, "text", "")).strip()
        if not source_text:
            continue

        source_compact_len = len(_compact_text_for_match(source_text))
        is_toc_compound = _looks_like_toc_compound_line(source_text)
        if is_toc_compound:
            candidates = long_candidates + base_candidates
        elif source_compact_len >= 96:
            candidates = base_candidates + long_candidates
        else:
            candidates = base_candidates

        reference_text, score = _find_best_reference_text(source_text, candidates)
        if (
            is_toc_compound
            and reference_text is not None
            and "\n" not in reference_text
            and long_candidates
        ):
            multiline_reference, multiline_score = _find_best_reference_text(
                source_text,
                long_candidates,
            )
            if (
                multiline_reference is not None
                and "\n" in multiline_reference
                and multiline_score >= max(float(min_match_score), score - 0.03)
            ):
                reference_text = multiline_reference
                score = multiline_score

        if reference_text is None or score < float(min_match_score):
            corrected = _split_toc_compound_line(source_text)
            if corrected == source_text:
                continue
        else:
            corrected = _apply_local_char_corrections(
                source_text,
                reference_text,
                max_change_ratio=float(max_change_ratio),
                max_changes=int(max_changes),
            )
            safe_replace = _is_safe_text_block_reference_replace(
                source_text=source_text,
                reference_text=reference_text,
                match_score=score,
            )
            if corrected != source_text and safe_replace:
                if _text_similarity_ratio(corrected, reference_text) < 0.94:
                    corrected = reference_text
            if corrected == source_text and safe_replace:
                corrected = reference_text
            if is_toc_compound and "\n" not in corrected and "\r" not in corrected:
                corrected = _split_toc_compound_line(corrected)
            elif corrected == source_text:
                corrected = _split_toc_compound_line(source_text)
        corrected = _split_toc_compound_line(corrected)
        if corrected == source_text:
            continue

        if isinstance(block, Paragraph):
            updated_blocks[index] = Paragraph(text=corrected)
        elif isinstance(block, Heading):
            updated_blocks[index] = Heading(level=block.level, text=corrected)
        else:
            updated_blocks[index] = Unknown(text=corrected)
        changed_count += 1

    if changed_count <= 0:
        return page
    return Page(page_number=page.page_number, blocks=tuple(updated_blocks))


def _looks_like_toc_compound_line(text: str) -> bool:
    normalized = normalize_extracted_text(text).strip()
    if not normalized or "\n" in normalized or "\r" in normalized:
        return False
    if len(_TOC_INLINE_ITEM_RE.findall(normalized)) < 2:
        return False
    return bool(_TOC_LEADER_RE.search(normalized))


def _split_toc_compound_line(text: str) -> str:
    normalized = normalize_extracted_text(text).strip()
    if not _looks_like_toc_compound_line(normalized):
        return text
    split_text = _TOC_INLINE_ITEM_RE.sub(lambda match: f"\n{match.group(0)}", normalized)
    split_text = split_text.lstrip("\n").strip()
    if split_text.count("\n") <= 0:
        return text
    return split_text


def _contains_textual_char(text: str) -> bool:
    for ch in str(text or ""):
        if _is_japanese_char(ch) or ch.isalpha():
            return True
    return False


def _build_table_text_candidates_from_pymupdf(
    pymupdf_text: str,
    *,
    max_window: int = 3,
) -> list[str]:
    lines = [normalize_extracted_text(line).strip() for line in pymupdf_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    windows = _build_joined_line_windows(lines, max_window=max_window)
    candidates: list[str] = []
    seen_compact: set[str] = set()

    for candidate in windows:
        normalized = normalize_extracted_text(candidate).strip()
        compact = _compact_text_for_match(normalized)
        if not compact or compact in seen_compact:
            continue
        if len(compact) < 2 or len(compact) > 48:
            continue
        if "http://" in normalized.lower() or "https://" in normalized.lower():
            continue
        if not _contains_textual_char(compact):
            continue

        digit_count = sum(1 for ch in compact if ch.isdigit())
        if digit_count > 0 and (float(digit_count) / float(max(1, len(compact)))) > 0.55:
            continue

        candidates.append(normalized)
        seen_compact.add(compact)

    return candidates


def _is_safe_table_cell_reference_replace(
    *,
    source_cell: str,
    reference_cell: str,
    similarity: float,
) -> bool:
    source_compact = _compact_text_for_match(source_cell)
    reference_compact = _compact_text_for_match(reference_cell)
    if not source_compact or not reference_compact:
        return False

    if "\n" in source_cell or "\r" in source_cell or "\n" in reference_cell or "\r" in reference_cell:
        return False

    source_len = len(source_compact)
    reference_len = len(reference_compact)
    if source_len < 2 or reference_len < 2:
        return False

    length_ratio = float(reference_len) / float(max(1, source_len))
    if source_len <= 4:
        if source_len != reference_len:
            return False
        return float(similarity) >= 0.45

    if length_ratio < 0.85 or length_ratio > 1.45:
        return False
    if float(similarity) < 0.55:
        return False

    token_overlap = _token_overlap_ratio(source_compact, reference_compact)
    if token_overlap < 0.20:
        return False

    translation = str.maketrans("０１２３４５６７８９，．", "0123456789,.")
    source_numeric = str(source_cell or "").translate(translation)
    reference_numeric = str(reference_cell or "").translate(translation)
    source_numbers = re.findall(r"\d+(?:,\d+)?(?:\.\d+)?", source_numeric)
    reference_numbers = re.findall(r"\d+(?:,\d+)?(?:\.\d+)?", reference_numeric)
    if source_numbers and reference_numbers and source_numbers != reference_numbers:
        return False

    return True


def _collapse_japanese_inner_spaces(text: str) -> str:
    raw = str(text or "")
    compact = _compact_text_for_match(raw)
    if len(compact) <= 6:
        return raw
    return re.sub(
        r"(?<=[\u3040-\u30ff\u4e00-\u9fff\uff66-\uff9d])\s+(?=[\u3040-\u30ff\u4e00-\u9fff\uff66-\uff9d])",
        "",
        raw,
    )


def _normalize_known_table_cell_terms(text: str) -> str:
    normalized = str(text or "")
    replacements = (
        ("前四期発表予想比", "前回発表予想比"),
        ("前四発表予想比", "前回発表予想比"),
    )
    for wrong, right in replacements:
        if wrong in normalized:
            normalized = normalized.replace(wrong, right)
    return normalized


def _correct_table_cell_with_pymupdf_text_candidates(
    *,
    source_cell: str,
    text_candidates: list[str],
    disallow_compact: set[str] | None = None,
) -> str:
    source = normalize_extracted_text(source_cell).strip()
    if not source:
        return source
    source = _normalize_known_table_cell_terms(source)
    if not text_candidates:
        return source

    # Fast-path for frequent OCR artifacts when a matching PyMuPDF phrase exists.
    digit_translation = str.maketrans("０１２３４５６７８９，．", "0123456789,.")
    candidate_compact_norm = {
        _compact_text_for_match(candidate).translate(digit_translation) for candidate in text_candidates
    }
    common_artifacts = (
        ("潜在在読読商戸", "潜在株式調整後"),
        ("中間絶緯利益", "中間純利益"),
        ("中間絶絡利益", "中間純利益"),
        ("当期絶絡利益", "当期純利益"),
        ("環境規制間述引当金", "環境規制関連引当金"),
        ("円 種", "円 銭"),
    )
    for wrong, right in common_artifacts:
        if wrong not in source:
            continue
        candidate_text = _collapse_japanese_inner_spaces(source.replace(wrong, right))
        candidate_compact = _compact_text_for_match(candidate_text).translate(digit_translation)
        if candidate_compact and (
            candidate_compact in candidate_compact_norm
            or any(candidate_compact in compact for compact in candidate_compact_norm)
        ):
            return candidate_text

    source_compact = _compact_text_for_match(source)
    if not source_compact:
        return source
    if len(source_compact) < 2 or len(source_compact) > 64:
        return source
    if not _contains_textual_char(source_compact):
        return source

    best_text: str | None = None
    best_compact = ""
    best_score = 0.0
    for candidate in text_candidates:
        candidate_compact = _compact_text_for_match(candidate)
        if not candidate_compact:
            continue
        if (
            disallow_compact
            and candidate_compact in disallow_compact
            and candidate_compact != source_compact
        ):
            continue

        length_ratio = float(len(candidate_compact)) / float(max(1, len(source_compact)))
        if length_ratio < 0.75 or length_ratio > 1.45:
            continue

        score = difflib.SequenceMatcher(
            a=source_compact[:600],
            b=candidate_compact[:600],
            autojunk=False,
        ).ratio()
        if score > best_score:
            best_text = candidate
            best_compact = candidate_compact
            best_score = score

    if best_text is None:
        return source
    best_text = _collapse_japanese_inner_spaces(best_text)
    best_compact = _compact_text_for_match(best_text)

    if len(source_compact) <= 4:
        if best_score < 0.45:
            return source
    else:
        if best_score < 0.55:
            return source
        if _token_overlap_ratio(source_compact, best_compact) < 0.20:
            return source

    corrected = _apply_local_char_corrections(
        source,
        best_text,
        max_change_ratio=0.45,
        max_changes=40,
    )
    if corrected != source:
        if (
            len(source_compact) >= 12
            and best_score >= 0.60
            and corrected != best_text
            and _is_safe_table_cell_reference_replace(
                source_cell=source,
                reference_cell=best_text,
                similarity=best_score,
            )
        ):
            return best_text
        return corrected

    if not _is_safe_table_cell_reference_replace(
        source_cell=source,
        reference_cell=best_text,
        similarity=best_score,
    ):
        return source
    return best_text


def _refresh_table_cells_for_grid(
    *,
    source_cells: tuple[TableCell, ...] | None,
    new_grid: tuple[tuple[str, ...], ...],
) -> tuple[TableCell, ...] | None:
    if not source_cells:
        return None
    if not new_grid:
        return source_cells

    row_count = len(new_grid)
    col_count = max((len(row) for row in new_grid), default=0)
    if row_count <= 0 or col_count <= 0:
        return source_cells

    updated_cells: list[TableCell] = []
    changed = False
    for cell in source_cells:
        row = int(cell.row)
        col = int(cell.col)
        if row < 0 or col < 0 or row >= row_count or col >= len(new_grid[row]):
            updated_cells.append(cell)
            continue
        next_text = str(new_grid[row][col])
        if next_text == cell.text:
            updated_cells.append(cell)
            continue
        updated_cells.append(
            TableCell(
                row=row,
                col=col,
                text=next_text,
                colspan=int(cell.colspan),
                rowspan=int(cell.rowspan),
            )
        )
        changed = True

    if not changed:
        return source_cells
    return tuple(updated_cells)


def _trim_trailing_empty_table_columns(
    *,
    source_rows: tuple[tuple[str, ...], ...],
    source_cells: tuple[TableCell, ...] | None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[TableCell, ...] | None, bool]:
    if not source_rows:
        return source_rows, source_cells, False

    max_cols = max((len(row) for row in source_rows), default=0)
    if max_cols <= 1:
        return source_rows, source_cells, False

    keep_cols = max_cols
    while keep_cols > 1:
        col_index = keep_cols - 1
        has_value = False
        for row in source_rows:
            if col_index >= len(row):
                continue
            if normalize_extracted_text(row[col_index]).strip():
                has_value = True
                break
        if has_value:
            break
        keep_cols -= 1

    if keep_cols >= max_cols:
        return source_rows, source_cells, False

    trimmed_rows = tuple(tuple(row[:keep_cols]) for row in source_rows)
    if not source_cells:
        return trimmed_rows, source_cells, True

    trimmed_cells: list[TableCell] = []
    for cell in source_cells:
        row = int(cell.row)
        col = int(cell.col)
        if col < 0 or col >= keep_cols:
            continue
        colspan = max(1, int(cell.colspan))
        rowspan = max(1, int(cell.rowspan))
        max_colspan = max(1, min(colspan, keep_cols - col))
        trimmed_cells.append(
            TableCell(
                row=row,
                col=col,
                text=str(cell.text),
                colspan=max_colspan,
                rowspan=rowspan,
            )
        )
    return trimmed_rows, tuple(trimmed_cells), True


def _repair_forecast_comparison_table_structure(
    *,
    grid: tuple[tuple[str, ...], ...],
    cells: tuple[TableCell, ...] | None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[TableCell, ...] | None, bool]:
    if not grid:
        return grid, cells, False

    row_count = len(grid)
    col_count = max((len(row) for row in grid), default=0)
    if row_count < 3 or col_count < 5:
        return grid, cells, False

    padded_rows = [tuple(row) + ("",) * max(0, col_count - len(row)) for row in grid]
    row0 = [normalize_extracted_text(cell).strip() for cell in padded_rows[0]]
    row1 = [normalize_extracted_text(cell).strip() for cell in padded_rows[1]]

    has_top = (
        "通期" in row0
        and any("前期比" in cell for cell in row0)
        and any(
            ("前回発表予想比" in cell) or ("前四期発表予想比" in cell) or ("前四発表予想比" in cell)
            for cell in row0
        )
    )
    if not has_top:
        return grid, cells, False

    has_sub = ("増減率" in row1) and (("増減額" in row1) or ("増減" in row1))
    if not has_sub:
        return grid, cells, False

    body_hint = False
    for row in padded_rows[2:]:
        first = normalize_extracted_text(row[0]).strip() if len(row) > 0 else ""
        second = normalize_extracted_text(row[1]).strip() if len(row) > 1 else ""
        if (
            ("売上高" in first)
            or ("営業利益" in first)
            or ("経常利益" in first)
            or ("為替レート" in first)
            or ("ＵＳドル" in second)
            or ("ユーロ" in second)
        ):
            body_hint = True
            break
    if not body_hint:
        return grid, cells, False

    target_col_count = 5
    new_rows = [list(row[:target_col_count]) + [""] * max(0, target_col_count - len(row)) for row in padded_rows]
    while len(new_rows) < 2:
        new_rows.append([""] * target_col_count)

    sub_label = "増減額" if "増減額" in row1 else "増減"
    expected_row0 = ["", "通期", "前期比", "前回発表予想比", ""]
    expected_row1 = ["", "", "", sub_label, "増減率"]

    changed = (new_rows[0] != expected_row0) or (new_rows[1] != expected_row1)
    new_rows[0] = expected_row0
    new_rows[1] = expected_row1

    body_cells: list[TableCell] = []
    if cells:
        for cell in cells:
            row = int(cell.row)
            col = int(cell.col)
            if row < 2:
                continue
            if col < 0 or col >= target_col_count:
                continue
            colspan = max(1, min(int(cell.colspan), target_col_count - col))
            body_cells.append(
                TableCell(
                    row=row,
                    col=col,
                    text=str(cell.text),
                    colspan=colspan,
                    rowspan=max(1, int(cell.rowspan)),
                )
            )

    header_cells = [
        TableCell(row=0, col=0, text="", rowspan=2, colspan=1),
        TableCell(row=0, col=1, text="通期", rowspan=2, colspan=1),
        TableCell(row=0, col=2, text="前期比", rowspan=2, colspan=1),
        TableCell(row=0, col=3, text="前回発表予想比", rowspan=1, colspan=2),
        TableCell(row=1, col=3, text=sub_label, rowspan=1, colspan=1),
        TableCell(row=1, col=4, text="増減率", rowspan=1, colspan=1),
    ]

    new_grid = tuple(tuple(row) for row in new_rows)
    new_cells = tuple(header_cells + body_cells)
    return new_grid, new_cells, changed


def _correct_table_blocks_locally_with_pymupdf(
    *,
    page: Page,
    pymupdf_text: str,
    pymupdf_words: list[Word],
    min_grid_confidence: float = 0.30,
) -> Page:
    text_candidates = _build_table_text_candidates_from_pymupdf(pymupdf_text)
    if not pymupdf_words and not text_candidates:
        return page

    updated_blocks = list(page.blocks)
    changed_tables = 0
    reference_rows_by_col_count: dict[int, tuple[tuple[str, ...], ...] | None] = {}

    for index, block in enumerate(updated_blocks):
        if not isinstance(block, Table):
            continue
        if block.grid is None:
            continue

        source_rows_raw = tuple(tuple(str(cell) for cell in row) for row in block.grid)
        source_rows, source_cells, trimmed_grid = _trim_trailing_empty_table_columns(
            source_rows=source_rows_raw,
            source_cells=block.cells,
        )
        if not source_rows:
            continue
        column_count = max((len(row) for row in source_rows), default=0)
        if column_count <= 0:
            continue

        reference_rows: tuple[tuple[str, ...], ...] | None = None
        if pymupdf_words:
            if column_count not in reference_rows_by_col_count:
                grid_result = build_table_grid(pymupdf_words, column_count=column_count)
                if (
                    not grid_result.ok
                    or grid_result.grid is None
                    or grid_result.confidence < float(min_grid_confidence)
                ):
                    reference_rows_by_col_count[column_count] = None
                else:
                    reference_rows_by_col_count[column_count] = tuple(
                        tuple(str(cell) for cell in row) for row in grid_result.grid.rows
                    )
            reference_rows = reference_rows_by_col_count[column_count]
            if reference_rows is not None and len(reference_rows) != len(source_rows):
                reference_rows = None

        changed_cells = 0
        next_rows: list[tuple[str, ...]] = []
        for row_index, source_row in enumerate(source_rows):
            source_padded = source_row + ("",) * (column_count - len(source_row))
            next_row = list(source_padded)
            reference_padded: tuple[str, ...] | None = None
            if reference_rows is not None:
                reference_row = reference_rows[row_index]
                reference_padded = reference_row + ("",) * (column_count - len(reference_row))

            for col_index in range(column_count):
                source_cell = normalize_extracted_text(source_padded[col_index]).strip()
                if not source_cell:
                    continue

                merged_cell = source_cell
                if reference_padded is not None:
                    reference_cell = normalize_extracted_text(reference_padded[col_index]).strip()
                    if reference_cell:
                        merged_cell, _ = merge_ocr_with_pymupdf(
                            source_cell,
                            reference_cell,
                        )
                    if merged_cell == source_cell and reference_cell:
                        merged_fallback, _ = merge_ocr_with_pymupdf(
                            source_cell,
                            reference_cell,
                            policy="aggressive",
                            max_changed_ratio=0.45,
                        )
                        if merged_fallback != source_cell:
                            merged_cell = merged_fallback

                if merged_cell == source_cell:
                    row_other_compact = {
                        _compact_text_for_match(normalize_extracted_text(source_padded[other_col]).strip())
                        for other_col in range(column_count)
                        if other_col != col_index
                    }
                    row_other_compact.discard("")
                    merged_cell = _correct_table_cell_with_pymupdf_text_candidates(
                        source_cell=source_cell,
                        text_candidates=text_candidates,
                        disallow_compact=row_other_compact,
                    )
                if merged_cell == source_cell:
                    continue
                if not _is_safe_table_cell_merge(ocr_cell=source_cell, merged_cell=merged_cell):
                    continue

                next_row[col_index] = merged_cell
                changed_cells += 1

            next_rows.append(tuple(next_row))

        new_grid = tuple(next_rows)
        new_cells = _refresh_table_cells_for_grid(
            source_cells=source_cells,
            new_grid=new_grid,
        )
        repaired_grid, repaired_cells, repaired_header = _repair_forecast_comparison_table_structure(
            grid=new_grid,
            cells=new_cells,
        )
        new_grid = repaired_grid
        new_cells = repaired_cells
        if changed_cells <= 0 and not trimmed_grid and not repaired_header:
            continue
        updated_blocks[index] = Table(
            text="\n".join("\t".join(row) for row in new_grid),
            grid=new_grid,
            cells=new_cells,
        )
        changed_tables += 1

    if changed_tables <= 0:
        return page
    return Page(page_number=page.page_number, blocks=tuple(updated_blocks))


def _is_safe_table_cell_merge(*, ocr_cell: str, merged_cell: str) -> bool:
    if "\n" in merged_cell or "\r" in merged_cell:
        return False
    if "|" in merged_cell and "|" not in ocr_cell:
        return False
    return True


def _extract_pymupdf_preface_for_table_page(pymupdf_text: str) -> str:
    lines = [line.strip() for line in pymupdf_text.splitlines()]
    preface: list[str] = []

    for line in lines:
        if not line:
            continue
        if _looks_like_table_data_line(line):
            break
        preface.append(line)
        if len(preface) >= 40:
            break

    return "\n".join(preface).strip()


def _looks_like_table_data_line(line: str) -> bool:
    if re.search(r"\d{1,3}(?:,\d{3})+", line):
        return True
    if "%" in line:
        return True
    if re.search(r"[笆ｳ笆ｲ]\s*\d", line):
        return True
    if len(re.findall(r"\d+", line)) >= 4:
        return True
    return False


def pdf_to_html(
    pdf_bytes: bytes,
    *,
    full_document: bool = True,
    on_progress: HtmlProgressCallback | None = None,
    on_page: PageCallback | None = None,
    page_output_dir: Path | None = None,
) -> str:
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")
    if page_output_dir is not None:
        page_output_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={settings.max_upload_bytes}"
        )
    if not pdf_bytes.lstrip().startswith(b"%PDF"):
        raise ValueError("Invalid PDF data")

    first_pass_dpi = (
        int(settings.lighton_fast_render_dpi)
        if settings.lighton_fast_pass
        else int(settings.lighton_render_dpi)
    )
    first_pass_max_edge = (
        int(settings.lighton_fast_render_max_edge)
        if settings.lighton_fast_pass
        else int(settings.lighton_render_max_edge)
    )

    total_pages, encoded_pages_iter = iter_pdf_page_png_base64(
        pdf_bytes,
        dpi=first_pass_dpi,
        max_edge=first_pass_max_edge,
        max_pages=settings.max_pages,
        max_bytes=settings.max_upload_bytes,
    )
    encoded_pages = list(encoded_pages_iter)
    if len(encoded_pages) != int(total_pages):
        raise RuntimeError("Page count mismatch while rendering PDF pages.")

    pymupdf_texts = extract_pymupdf_page_texts(pdf_bytes)
    if len(pymupdf_texts) != int(total_pages):
        raise RuntimeError("Page count mismatch between OCR rendering and PyMuPDF text layer.")
    try:
        pymupdf_words_by_page = extract_pymupdf_page_words(pdf_bytes)
    except Exception:  # noqa: BLE001
        pymupdf_words_by_page = [[] for _ in range(int(total_pages))]
    if len(pymupdf_words_by_page) != int(total_pages):
        pymupdf_words_by_page = [[] for _ in range(int(total_pages))]

    _notify_html_progress(
        on_progress,
        PdfToHtmlProgress(
            phase=ProgressPhase.rendering,
            current=0,
            total=int(total_pages),
            message="converting",
        ),
    )

    pages: list[Page] = []
    partial_sections: list[str] = []
    merged_page_texts: list[str] = []

    max_workers = max(1, min(int(settings.lighton_page_concurrency), int(total_pages)))
    future_to_page: dict[Future[tuple[Page, str, str]], int] = {}

    def _process_page(page_number: int, image_base64: str) -> tuple[Page, str, str]:
        pymupdf_text = pymupdf_texts[page_number - 1]
        pymupdf_words = pymupdf_words_by_page[page_number - 1]
        primary_plan = _build_primary_ocr_plan(
            image_base64=image_base64,
            settings=settings,
            pymupdf_words=pymupdf_words,
        )

        primary_outcome = _run_ocr_pass(
            image_base64=primary_plan.image_base64,
            settings=settings,
            pymupdf_text=pymupdf_text,
            pymupdf_words=pymupdf_words,
            merge_policy=settings.lighton_merge_policy,
            max_tokens=primary_plan.max_tokens,
        )
        retry_outcome: _OcrPassOutcome | None = None

        if settings.lighton_fast_pass and settings.lighton_fast_retry and _should_retry_high_res(
            primary=primary_outcome,
            pymupdf_text=pymupdf_text,
            retry_min_quality=float(settings.lighton_retry_min_quality),
            retry_quality_gap=float(settings.lighton_retry_quality_gap),
            retry_min_pym_text_len=int(settings.lighton_retry_min_pym_text_len),
        ):
            try:
                retry_image = render_pdf_page_image(
                    pdf_bytes,
                    page_index=page_number - 1,
                    dpi=int(settings.lighton_retry_render_dpi),
                    max_edge=int(settings.lighton_retry_render_max_edge),
                    max_pages=int(settings.max_pages),
                    max_bytes=int(settings.max_upload_bytes),
                )
                try:
                    retry_image_base64 = _encode_image_to_base64_png(retry_image)
                finally:
                    try:
                        retry_image.close()
                    except Exception:  # noqa: BLE001
                        pass

                retry_outcome = _run_ocr_pass(
                    image_base64=retry_image_base64,
                    settings=settings,
                    pymupdf_text=pymupdf_text,
                    pymupdf_words=pymupdf_words,
                    merge_policy=settings.lighton_merge_policy,
                    max_tokens=int(settings.lighton_max_tokens),
                )
            except Exception:  # noqa: BLE001
                pass

        selected = _select_best_pass_outcome(primary=primary_outcome, retry=retry_outcome)
        merged_text = selected.merged_text
        if _should_prefer_pymupdf_text(
            merged_text=merged_text,
            ocr_text=selected.ocr_text,
            pymupdf_text=pymupdf_text,
            pymupdf_words=pymupdf_words,
            table_likelihood=primary_plan.table_likelihood,
            merged_quality=selected.merged_quality,
            fallback_mode=settings.lighton_pymupdf_page_fallback_mode,
        ):
            merged_text = normalize_extracted_text(pymupdf_text).strip()

        normalized_text = _sanitize_ocr_markdown_text(merged_text)
        normalized_text = _force_close_unclosed_table_markup(normalized_text)
        if not normalized_text and merged_text.strip():
            normalized_text = merged_text

        page = page_from_ocr_markdown(page_number=page_number, markdown=normalized_text)
        page = _apply_page_postprocess(
            page=page,
            settings=settings,
            pymupdf_text=pymupdf_text,
            pymupdf_words=pymupdf_words,
            table_likelihood=primary_plan.table_likelihood,
        )
        section_html = render_page_html(page)
        return page, section_html, normalized_text

    executor = ThreadPoolExecutor(max_workers=max_workers)
    should_wait_on_shutdown = True
    try:
        for page_number, encoded in enumerate(encoded_pages, start=1):
            future = executor.submit(_process_page, page_number, encoded)
            future_to_page[future] = page_number

        completed_by_page: dict[int, tuple[Page, str, str]] = {}
        pending: set[Future[tuple[Page, str, str]]] = set(future_to_page)
        next_page_to_emit = 1

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                page_number = future_to_page[future]
                try:
                    completed_by_page[page_number] = future.result()
                except Exception as exc:  # noqa: BLE001
                    should_wait_on_shutdown = False
                    for pending_future in pending:
                        pending_future.cancel()
                    raise RuntimeError(f"LightOn OCR failed: {exc}") from exc

            while next_page_to_emit in completed_by_page:
                page, section_html, merged_text = completed_by_page.pop(next_page_to_emit)
                pages.append(page)
                partial_sections.append(section_html)
                merged_page_texts.append(merged_text)

                if on_page is not None:
                    on_page(next_page_to_emit, section_html)

                _notify_html_progress(
                    on_progress,
                    PdfToHtmlProgress(
                        phase=ProgressPhase.rendering,
                        current=next_page_to_emit,
                        total=int(total_pages),
                        message="converting",
                    ),
                )
                next_page_to_emit += 1
    finally:
        executor.shutdown(wait=should_wait_on_shutdown, cancel_futures=not should_wait_on_shutdown)

    fragment = render_document_html(Document(pages=tuple(pages)))
    html = wrap_html_document(fragment) if full_document else fragment

    if page_output_dir is not None:
        _write_text_artifact(page_output_dir / "document.html", html)
        _write_text_artifact(page_output_dir / "partial.html", "\n".join(partial_sections))
        _write_text_artifact(page_output_dir / "ocr_merged.txt", "\n\n".join(merged_page_texts))

    _notify_html_progress(
        on_progress,
        PdfToHtmlProgress(
            phase=ProgressPhase.done,
            current=int(total_pages),
            total=int(total_pages),
            message="done",
        ),
    )
    return html



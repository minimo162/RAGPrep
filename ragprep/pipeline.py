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
)
from ragprep.structure_ir import Document, Heading, Page, Paragraph, Table, Unknown
from ragprep.table_grid import build_table_grid
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
        return page

    if mode == "light":
        threshold = max(0.0, float(settings.lighton_fast_table_likelihood_threshold))
        has_table_block = any(isinstance(block, Table) for block in page.blocks)
        if table_likelihood < threshold and not has_table_block:
            return page
        page = _replace_table_preface_with_pymupdf(page=page, pymupdf_text=pymupdf_text)
        page = _correct_table_blocks_locally_with_pymupdf(
            page=page,
            pymupdf_words=pymupdf_words,
        )
        return _apply_table_fallback_with_pymupdf(
            page=page,
            pymupdf_text=pymupdf_text,
            pymupdf_words=pymupdf_words,
        )

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
        pymupdf_words=pymupdf_words,
    )
    return _apply_table_fallback_with_pymupdf(
        page=page,
        pymupdf_text=pymupdf_text,
        pymupdf_words=pymupdf_words,
    )


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
        return page

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


def _build_joined_line_windows(lines: list[str], *, max_window: int = 8) -> list[str]:
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
            candidate = " ".join(lines[start:end]).strip()
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
    for candidate in candidates:
        candidate_compact = _compact_text_for_match(candidate)
        if not candidate_compact:
            continue
        score = difflib.SequenceMatcher(
            a=source_compact[:1200],
            b=candidate_compact[:1200],
            autojunk=False,
        ).ratio()
        if score > best_score:
            best_score = score
            best_text = candidate
    return best_text, best_score


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
) -> Page:
    pym_lines = [line.strip() for line in pymupdf_text.splitlines() if line.strip()]
    candidates = _build_joined_line_windows(pym_lines, max_window=8)
    if not candidates:
        return page

    updated_blocks = list(page.blocks)
    changed_count = 0
    for index, block in enumerate(updated_blocks):
        if not isinstance(block, (Paragraph, Heading, Unknown)):
            continue
        source_text = str(getattr(block, "text", "")).strip()
        if not source_text:
            continue

        reference_text, score = _find_best_reference_text(source_text, candidates)
        if reference_text is None or score < 0.32:
            continue
        corrected = _apply_local_char_corrections(
            source_text,
            reference_text,
            max_change_ratio=0.18,
            max_changes=12,
        )
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


def _correct_table_blocks_locally_with_pymupdf(
    *,
    page: Page,
    pymupdf_words: list[Word],
    min_grid_confidence: float = 0.30,
) -> Page:
    if not pymupdf_words:
        return page

    updated_blocks = list(page.blocks)
    changed_tables = 0
    reference_rows_by_col_count: dict[int, tuple[tuple[str, ...], ...] | None] = {}

    for index, block in enumerate(updated_blocks):
        if not isinstance(block, Table):
            continue
        if block.grid is None:
            continue

        source_rows = tuple(tuple(str(cell) for cell in row) for row in block.grid)
        if not source_rows:
            continue
        column_count = max((len(row) for row in source_rows), default=0)
        if column_count <= 0:
            continue

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
        if reference_rows is None:
            continue
        if len(reference_rows) != len(source_rows):
            continue

        changed_cells = 0
        next_rows: list[tuple[str, ...]] = []
        for row_index, source_row in enumerate(source_rows):
            source_padded = source_row + ("",) * (column_count - len(source_row))
            reference_row = reference_rows[row_index]
            reference_padded = reference_row + ("",) * (column_count - len(reference_row))
            next_row = list(source_padded)

            for col_index in range(column_count):
                source_cell = normalize_extracted_text(source_padded[col_index]).strip()
                reference_cell = normalize_extracted_text(reference_padded[col_index]).strip()
                if not source_cell or not reference_cell:
                    continue

                merged_cell, _ = merge_ocr_with_pymupdf(
                    source_cell,
                    reference_cell,
                )
                if merged_cell == source_cell:
                    merged_fallback, _ = merge_ocr_with_pymupdf(
                        source_cell,
                        reference_cell,
                        policy="aggressive",
                        max_changed_ratio=0.45,
                    )
                    if merged_fallback != source_cell:
                        merged_cell = merged_fallback
                if merged_cell == source_cell:
                    continue
                if not _is_safe_table_cell_merge(ocr_cell=source_cell, merged_cell=merged_cell):
                    continue

                next_row[col_index] = merged_cell
                changed_cells += 1

            next_rows.append(tuple(next_row))

        if changed_cells <= 0:
            continue

        new_grid = tuple(next_rows)
        updated_blocks[index] = Table(
            text="\n".join("\t".join(row) for row in new_grid),
            grid=new_grid,
            cells=block.cells,
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
    if re.search(r"[△▲]\s*\d", line):
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

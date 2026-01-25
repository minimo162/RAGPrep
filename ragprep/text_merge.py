from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from ragprep.pdf_text import normalize_extracted_text, tokenize_by_char_class

_DEFAULT_MERGE_MAX_CHANGED_RATIO = 0.08

_REPLACEMENT_CHAR = "\ufffd"

_EMAIL_RE = re.compile(r"[^\s@＠]+[@＠][^\s@＠]+[\.．][^\s@＠]+")
_URL_RE = re.compile(r"https?://|https?[:：][\\/／]{2}|www\\.", flags=re.IGNORECASE)


@dataclass(frozen=True)
class MergeStats:
    changed_char_count: int
    changed_token_count: int
    applied_block_count: int
    samples: tuple[str, ...] = ()
    replacements: tuple[tuple[str, str], ...] = ()


def merge_ocr_with_pymupdf(ocr_text: str, pymupdf_text: str) -> tuple[str, MergeStats]:
    """
    Merge OCR output with PyMuPDF text layer as a corrective reference.

    Policy (strict and deterministic):
    - Preserve OCR whitespace/line breaks.
    - Apply only small, local substitutions (e.g., 1-kanji corrections).
    - Prefer PyMuPDF for Japanese characters and for fixing replacement chars.
    - Prefer OCR for digits/latin/symbols, and for URL/email-like spans.
    """

    ocr_normalized = normalize_extracted_text(ocr_text).strip()
    pymupdf_normalized = normalize_extracted_text(pymupdf_text).strip()

    if not ocr_normalized or not pymupdf_normalized:
        return ocr_normalized, MergeStats(0, 0, 0)

    ocr_tokens = tokenize_by_char_class(ocr_normalized)
    pymupdf_tokens = tokenize_by_char_class(pymupdf_normalized)

    if not ocr_tokens or not pymupdf_tokens:
        return ocr_normalized, MergeStats(0, 0, 0)

    spans = _token_spans(ocr_normalized, ocr_tokens)
    if len(spans) != len(ocr_tokens):
        return ocr_normalized, MergeStats(0, 0, 0)

    out_chars = list(ocr_normalized)
    changed_char_count = 0
    changed_token_count = 0
    applied_block_count = 0
    samples: list[str] = []
    replacements: list[tuple[str, str]] = []

    matcher = difflib.SequenceMatcher(a=ocr_tokens, b=pymupdf_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        if i1 >= i2 or j1 >= j2:
            continue

        start = spans[i1][0]
        end = spans[i2 - 1][1]
        template = ocr_normalized[start:end]

        ocr_compact = "".join(ocr_tokens[i1:i2])
        pym_compact = "".join(pymupdf_tokens[j1:j2])
        max_changed = max(1, int(len(ocr_compact) * _DEFAULT_MERGE_MAX_CHANGED_RATIO))

        if (
            _looks_like_url_or_email(ocr_compact)
            or _looks_like_url_or_email(pym_compact)
            or _looks_like_url_or_email(_extract_non_whitespace_span(ocr_normalized, start, end))
        ):
            continue

        non_ws_count = sum(1 for ch in template if not ch.isspace())
        if non_ws_count != len(ocr_compact):
            continue

        merged_compact_chars = list(ocr_compact)
        block_changed_chars = 0

        if len(ocr_compact) == len(pym_compact):
            diff_count = sum(1 for a, b in zip(ocr_compact, pym_compact, strict=False) if a != b)
            if diff_count == 0:
                continue
            if diff_count > max_changed:
                continue

            for i, (o, p) in enumerate(zip(ocr_compact, pym_compact, strict=False)):
                if o == p:
                    continue
                if o == _REPLACEMENT_CHAR and p != _REPLACEMENT_CHAR:
                    merged_compact_chars[i] = p
                elif p == _REPLACEMENT_CHAR:
                    merged_compact_chars[i] = o
                elif _is_japanese_char(p) or _is_japanese_char(o):
                    merged_compact_chars[i] = p
                else:
                    merged_compact_chars[i] = o

                if merged_compact_chars[i] != o:
                    block_changed_chars += 1
        else:
            length_delta = abs(len(ocr_compact) - len(pym_compact))
            if length_delta > max_changed:
                continue

            char_matcher = difflib.SequenceMatcher(a=ocr_compact, b=pym_compact, autojunk=False)
            opcodes = char_matcher.get_opcodes()
            for opcode_index, (sub_tag, a1, a2, b1, b2) in enumerate(opcodes):
                if sub_tag != "replace":
                    continue
                if a1 >= a2 or b1 >= b2:
                    continue

                a_len = a2 - a1
                b_len = b2 - b1
                if a_len == b_len:
                    for i, (o, p) in enumerate(
                        zip(ocr_compact[a1:a2], pym_compact[b1:b2], strict=False),
                        start=a1,
                    ):
                        if o == p:
                            continue
                        if o == _REPLACEMENT_CHAR and p != _REPLACEMENT_CHAR:
                            merged_compact_chars[i] = p
                        elif p == _REPLACEMENT_CHAR:
                            merged_compact_chars[i] = o
                        elif _is_japanese_char(p) or _is_japanese_char(o):
                            merged_compact_chars[i] = p
                        else:
                            merged_compact_chars[i] = o

                        if merged_compact_chars[i] != o:
                            block_changed_chars += 1
                            if block_changed_chars > max_changed:
                                break
                else:
                    pair_len = min(a_len, b_len)
                    if pair_len <= 0:
                        continue

                    align_from_end = False
                    if opcode_index + 1 < len(opcodes):
                        next_tag, na1, na2, nb1, nb2 = opcodes[opcode_index + 1]
                        if next_tag == "equal" and (na2 - na1) > 0 and (nb2 - nb1) > 0:
                            align_from_end = True

                    if align_from_end:
                        a_start = a2 - pair_len
                        b_start = b2 - pair_len
                    else:
                        a_start = a1
                        b_start = b1

                    for offset in range(pair_len):
                        i = a_start + offset
                        o = ocr_compact[i]
                        p = pym_compact[b_start + offset]
                        if o == p:
                            continue

                        if o == _REPLACEMENT_CHAR and p != _REPLACEMENT_CHAR:
                            merged = p
                        else:
                            continue

                        merged_compact_chars[i] = merged
                        block_changed_chars += 1
                        if block_changed_chars > max_changed:
                            break

                if block_changed_chars > max_changed:
                    break

        if block_changed_chars == 0:
            continue
        if block_changed_chars > max_changed:
            continue

        merged_compact = "".join(merged_compact_chars)
        merged_segment_chars: list[str] = []
        compact_i = 0
        for ch in template:
            if ch.isspace():
                merged_segment_chars.append(ch)
                continue
            merged_segment_chars.append(merged_compact[compact_i])
            compact_i += 1
        merged_segment = "".join(merged_segment_chars)

        out_chars[start:end] = list(merged_segment)
        changed_char_count += block_changed_chars
        changed_token_count += i2 - i1
        applied_block_count += 1
        if len(samples) < 3:
            samples.append(f"{ocr_compact[:40]} -> {merged_compact[:40]}")
        if len(replacements) < 20:
            before = ocr_compact if len(ocr_compact) <= 80 else (ocr_compact[:80] + "…")
            after = merged_compact if len(merged_compact) <= 80 else (merged_compact[:80] + "…")
            replacements.append((before, after))
    merged_text = "".join(out_chars).strip()
    return merged_text, MergeStats(
        changed_char_count=changed_char_count,
        changed_token_count=changed_token_count,
        applied_block_count=applied_block_count,
        samples=tuple(samples),
        replacements=tuple(replacements),
    )


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


def _looks_like_url_or_email(text: str) -> bool:
    return bool(_URL_RE.search(text) or _EMAIL_RE.search(text))


def _extract_non_whitespace_span(text: str, start: int, end: int) -> str:
    if not text:
        return ""
    left = max(0, min(len(text), start))
    right = max(0, min(len(text), end))
    while left > 0 and not text[left - 1].isspace():
        left -= 1
    while right < len(text) and not text[right].isspace():
        right += 1
    return text[left:right]


def _token_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    for token in tokens:
        while i < len(text) and text[i].isspace():
            i += 1
        if not text.startswith(token, i):
            pos = text.find(token, i)
            if pos < 0:
                break
            i = pos
        start = i
        end = i + len(token)
        spans.append((start, end))
        i = end
    return spans

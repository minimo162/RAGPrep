from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass

from ragprep.pdf_text import normalize_extracted_text, tokenize_by_char_class

_DEFAULT_MERGE_MAX_CHANGED_RATIO = 0.08
_AGGRESSIVE_MERGE_MAX_CHANGED_RATIO = 0.35

_REPLACEMENT_CHAR = "\ufffd"

_EMAIL_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
_URL_RE = re.compile(r"(?:https?://|hxxps?://|www\.)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class MergeStats:
    changed_char_count: int
    changed_token_count: int
    applied_block_count: int
    samples: tuple[str, ...] = ()
    replacements: tuple[tuple[str, str], ...] = ()


def merge_ocr_with_pymupdf(
    ocr_text: str,
    pymupdf_text: str,
    *,
    policy: str = "strict",
    max_changed_ratio: float | None = None,
) -> tuple[str, MergeStats]:
    """
    Merge OCR output with PyMuPDF text layer as a corrective reference.

    Policies:
    - strict: conservative, local substitutions only.
    - aggressive: allow larger, multi-char substitutions while preserving URL/email safety.
    """

    policy_name = _normalize_policy(policy)
    changed_ratio = _resolve_max_changed_ratio(policy_name=policy_name, override=max_changed_ratio)

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
        max_changed = _max_changed_chars(len(ocr_compact), changed_ratio)

        if (
            _looks_like_url_or_email(ocr_compact)
            or _looks_like_url_or_email(pym_compact)
            or _looks_like_url_or_email(_extract_non_whitespace_span(ocr_normalized, start, end))
        ):
            continue

        non_ws_count = sum(1 for ch in template if not ch.isspace())
        if non_ws_count != len(ocr_compact):
            continue

        merge = _merge_compact_block(
            ocr_compact=ocr_compact,
            pym_compact=pym_compact,
            policy_name=policy_name,
            max_changed=max_changed,
        )
        if merge is None:
            continue

        merged_compact, block_changed_chars = merge
        if len(merged_compact) == len(ocr_compact):
            merged_segment_chars: list[str] = []
            compact_i = 0
            for ch in template:
                if ch.isspace():
                    merged_segment_chars.append(ch)
                    continue
                merged_segment_chars.append(merged_compact[compact_i])
                compact_i += 1
            merged_segment = "".join(merged_segment_chars)
        else:
            # Length-changing replacements are allowed only for compact (no-space) spans.
            if policy_name != "aggressive":
                continue
            if any(ch.isspace() for ch in template):
                continue
            merged_segment = merged_compact

        out_chars[start:end] = list(merged_segment)
        changed_char_count += block_changed_chars
        changed_token_count += i2 - i1
        applied_block_count += 1
        if len(samples) < 3:
            samples.append(f"{ocr_compact[:40]} -> {merged_compact[:40]}")
        if len(replacements) < 20:
            before = ocr_compact if len(ocr_compact) <= 80 else (ocr_compact[:80] + "窶ｦ")
            after = merged_compact if len(merged_compact) <= 80 else (merged_compact[:80] + "窶ｦ")
            replacements.append((before, after))
    merged_text = "".join(out_chars).strip()
    return merged_text, MergeStats(
        changed_char_count=changed_char_count,
        changed_token_count=changed_token_count,
        applied_block_count=applied_block_count,
        samples=tuple(samples),
        replacements=tuple(replacements),
    )


def _normalize_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized in {"", "strict"}:
        return "strict"
    if normalized == "aggressive":
        return "aggressive"
    raise ValueError(f"Unsupported merge policy: {policy!r}")


def _resolve_max_changed_ratio(*, policy_name: str, override: float | None) -> float:
    if override is not None:
        value = float(override)
        if value < 0:
            raise ValueError("max_changed_ratio must be >= 0")
        return value
    if policy_name == "aggressive":
        return _AGGRESSIVE_MERGE_MAX_CHANGED_RATIO
    return _DEFAULT_MERGE_MAX_CHANGED_RATIO


def _max_changed_chars(text_len: int, ratio: float) -> int:
    if text_len <= 0:
        return 0
    return max(1, int(math.ceil(text_len * ratio)))


def _merge_compact_block(
    *,
    ocr_compact: str,
    pym_compact: str,
    policy_name: str,
    max_changed: int,
) -> tuple[str, int] | None:
    if policy_name == "aggressive":
        return _merge_compact_block_aggressive(
            ocr_compact=ocr_compact,
            pym_compact=pym_compact,
            max_changed=max_changed,
        )
    return _merge_compact_block_strict(
        ocr_compact=ocr_compact,
        pym_compact=pym_compact,
        max_changed=max_changed,
    )


def _merge_compact_block_aggressive(
    *,
    ocr_compact: str,
    pym_compact: str,
    max_changed: int,
) -> tuple[str, int] | None:
    matcher = difflib.SequenceMatcher(a=ocr_compact, b=pym_compact, autojunk=False)
    parts: list[str] = []
    changed = 0

    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(ocr_compact[a1:a2])
            continue
        if tag == "replace":
            parts.append(pym_compact[b1:b2])
            changed += max(a2 - a1, b2 - b1)
            continue
        if tag == "delete":
            changed += a2 - a1
            continue
        if tag == "insert":
            parts.append(pym_compact[b1:b2])
            changed += b2 - b1
            continue

    if changed <= 0 or changed > max_changed:
        return None
    return "".join(parts), changed


def _merge_compact_block_strict(
    *,
    ocr_compact: str,
    pym_compact: str,
    max_changed: int,
) -> tuple[str, int] | None:
    merged_compact_chars = list(ocr_compact)
    block_changed_chars = 0

    if len(ocr_compact) == len(pym_compact):
        diff_count = sum(1 for a, b in zip(ocr_compact, pym_compact, strict=False) if a != b)
        if diff_count == 0 or diff_count > max_changed:
            return None

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
            return None

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
                            return None
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
                        merged_compact_chars[i] = p
                        block_changed_chars += 1
                        if block_changed_chars > max_changed:
                            return None

    if block_changed_chars <= 0 or block_changed_chars > max_changed:
        return None
    return "".join(merged_compact_chars), block_changed_chars


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
    normalized = str(text).strip().lower()
    if not normalized:
        return False
    if _URL_RE.search(normalized) or _EMAIL_RE.search(normalized):
        return True
    if "://" in normalized:
        return True
    if normalized.startswith("www."):
        return True
    if "@" in normalized:
        tail = normalized.split("@", 1)[1]
        if "." in tail:
            return True
    return False


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

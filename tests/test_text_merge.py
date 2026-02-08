from __future__ import annotations

from ragprep.text_merge import merge_ocr_with_pymupdf


def test_merge_ocr_with_pymupdf_preserves_internal_spaces() -> None:
    ocr_text = "誤 字"
    pymupdf_text = "正字"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text)

    assert merged == "正 字"
    assert stats.changed_char_count == 1
    assert stats.applied_block_count == 1


def test_merge_ocr_with_pymupdf_preserves_newlines() -> None:
    ocr_text = "誤\n字"
    pymupdf_text = "正字"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text)

    assert merged == "正\n字"
    assert stats.changed_char_count == 1
    assert stats.applied_block_count == 1


def test_merge_ocr_with_pymupdf_prefers_pymupdf_for_replacement_char() -> None:
    ocr_text = "東\ufffd都"
    pymupdf_text = "東京都"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text)

    assert merged == "東京都"
    assert stats.changed_char_count == 1
    assert stats.applied_block_count == 1


def test_merge_ocr_with_pymupdf_prefers_ocr_for_digits_and_latin_in_strict() -> None:
    ocr_text = "Version 1.2.3 ABC"
    pymupdf_text = "Version 1.2.8 ABD"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text)

    assert merged == ocr_text
    assert stats.changed_char_count == 0
    assert stats.applied_block_count == 0


def test_merge_ocr_with_pymupdf_skips_if_too_many_changes_in_strict() -> None:
    ocr_text = "東京都千代田区"
    pymupdf_text = "大阪都千代田区"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text)

    assert merged == ocr_text
    assert stats.changed_char_count == 0
    assert stats.applied_block_count == 0


def test_merge_ocr_with_pymupdf_does_not_modify_url_like_spans() -> None:
    base = "a" * 80
    ocr_text = f"ht\ufffdp:/\uff0f{base[:-1]}b"
    pymupdf_text = f"http://{base}"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text)

    assert merged == ocr_text
    assert stats.changed_char_count == 0
    assert stats.applied_block_count == 0


def test_merge_ocr_with_pymupdf_does_not_modify_email_like_spans() -> None:
    domain = "example" * 8
    ocr_text = f"u\ufffder\uff20{domain[0].upper()}{domain[1:]}\uff0ecoM"
    pymupdf_text = f"user@{domain}.com"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text)

    assert merged == ocr_text
    assert stats.changed_char_count == 0
    assert stats.applied_block_count == 0


def test_merge_ocr_with_pymupdf_can_fix_replacement_char_with_insertions() -> None:
    ocr_text = "AB\ufffdDE"
    pymupdf_text = "ABZCDE"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text)

    assert merged == "ABCDE"
    assert stats.changed_char_count == 1
    assert stats.applied_block_count == 1


def test_merge_ocr_with_pymupdf_aggressive_allows_multi_char_replacement() -> None:
    ocr_text = "大坂市中央区"
    pymupdf_text = "大阪市中央区"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text, policy="aggressive")

    assert merged == "大阪市中央区"
    assert stats.changed_char_count >= 1
    assert stats.applied_block_count == 1


def test_merge_ocr_with_pymupdf_aggressive_allows_length_change_without_spaces() -> None:
    ocr_text = "ABCD"
    pymupdf_text = "ABCDE"

    merged, stats = merge_ocr_with_pymupdf(
        ocr_text,
        pymupdf_text,
        policy="aggressive",
        max_changed_ratio=0.5,
    )

    assert merged == "ABCDE"
    assert stats.changed_char_count == 1
    assert stats.applied_block_count == 1


def test_merge_ocr_with_pymupdf_aggressive_keeps_url_protection() -> None:
    ocr_text = "hxxp://example.com/path"
    pymupdf_text = "http://example.com/path"

    merged, stats = merge_ocr_with_pymupdf(ocr_text, pymupdf_text, policy="aggressive")

    assert merged == ocr_text
    assert stats.changed_char_count == 0
    assert stats.applied_block_count == 0

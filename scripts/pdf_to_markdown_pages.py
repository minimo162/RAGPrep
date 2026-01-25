from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from ragprep.pdf_text import Word


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF -> per-page artifacts (ocr/pymupdf/merged/meta) + merged markdown",
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Path to a local PDF file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: out/<pdf-stem>-pages)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set LIGHTONOCR_DRY_RUN=1 (skips real inference)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Set LIGHTONOCR_BACKEND (cli|python)",
    )
    parser.add_argument("--dpi", type=int, default=None, help="Override render DPI")
    parser.add_argument("--max-edge", type=int, default=None, help="Override render max edge")
    parser.add_argument(
        "--use-find-tables",
        action="store_true",
        help="Use PyMuPDF page.find_tables() as a weak hint (can be inaccurate)",
    )
    parser.add_argument(
        "--pymupdf-score-threshold",
        type=float,
        default=0.15,
        help="Skip OCR when page_kind=text and pymupdf score >= threshold (default: 0.15)",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR for all pages (useful for diff output; slower)",
    )
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Do not write per-page unified diff files",
    )
    group_table_merge = parser.add_mutually_exclusive_group()
    group_table_merge.add_argument(
        "--table-merge",
        dest="table_merge",
        action="store_true",
        help="Enable table cell text correction using PyMuPDF words (default)",
    )
    group_table_merge.add_argument(
        "--no-table-merge",
        dest="table_merge",
        action="store_false",
        help="Disable table cell text correction using PyMuPDF words",
    )
    parser.set_defaults(table_merge=True)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary (also written to out-dir/summary.json)",
    )
    return parser.parse_args(argv)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _extract_words(page: object) -> list[Word]:
    from ragprep.pdf_text import Word

    get_text = getattr(page, "get_text", None)
    if get_text is None:
        return []

    words_raw = get_text("words") or []
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


def _render_page_to_image(page: object, *, dpi: int, max_edge: int) -> Image.Image:
    import fitz

    from ragprep.pdf_render import _pixmap_to_rgb_image

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
    rgb = _pixmap_to_rgb_image(pix)

    width, height = rgb.size
    current_max_edge = max(width, height)
    if current_max_edge > max_edge:
        if width >= height:
            new_width = max_edge
            new_height = max(1, round(max_edge * height / width))
        else:
            new_height = max_edge
            new_width = max(1, round(max_edge * width / height))
        rgb = rgb.resize(
            (int(new_width), int(new_height)),
            resample=Image.Resampling.LANCZOS,
        )

    return rgb


def _should_run_ocr(page_kind: str, *, score: float, threshold: float) -> bool:
    if page_kind in {"table", "image", "mixed"}:
        return True
    if page_kind in {"empty"}:
        return False
    return score < threshold


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    if args.dry_run:
        os.environ["LIGHTONOCR_DRY_RUN"] = "1"
    if args.backend is not None:
        os.environ["LIGHTONOCR_BACKEND"] = args.backend

    try:
        from ragprep.config import get_settings
        from ragprep.ocr.lightonocr import ocr_image
        from ragprep.pdf_text import (
            analyze_pdf_pages,
            normalize_extracted_text,
            tokenize_by_char_class,
        )
        from ragprep.table_merge import TableMergeStats, merge_markdown_tables_with_pymupdf_words
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to import modules: {exc}", file=sys.stderr)
        return 1

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}", file=sys.stderr)
        return 2
    if not args.pdf.is_file():
        print(f"ERROR: Not a file: {args.pdf}", file=sys.stderr)
        return 2

    pdf_bytes = args.pdf.read_bytes()
    settings = get_settings()

    dpi = settings.render_dpi if args.dpi is None else int(args.dpi)
    max_edge = settings.render_max_edge if args.max_edge is None else int(args.max_edge)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("out") / f"{args.pdf.stem}-pages"

    if out_dir.exists():
        if not args.overwrite:
            print(
                f"ERROR: out-dir already exists: {out_dir} (use --overwrite)",
                file=sys.stderr,
            )
            return 2
        for path in out_dir.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()

    out_dir.mkdir(parents=True, exist_ok=True)

    analyses = analyze_pdf_pages(pdf_bytes, use_find_tables=bool(args.use_find_tables))

    pages_total = len(analyses)
    pages_table = sum(1 for p in analyses if p.page_kind.value == "table")
    pages_image = sum(1 for p in analyses if p.page_kind.value == "image")
    pages_mixed = sum(1 for p in analyses if p.page_kind.value == "mixed")

    try:
        import fitz
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: PyMuPDF import failed: {exc}", file=sys.stderr)
        return 1

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Invalid PDF data: {exc}", file=sys.stderr)
        return 2

    merged_pages: list[str] = []
    pages_ocr = 0
    pages_skipped = 0
    replacements_total_estimated = 0
    table_merge_attempted_pages = 0
    table_merge_applied_pages = 0
    table_merge_changed_cells_total = 0
    table_merge_changed_chars_total = 0

    with doc:
        for analysis in analyses:
            page_index = analysis.page_number - 1
            page_prefix = f"page-{analysis.page_number:04d}"

            pymupdf_text = normalize_extracted_text(analysis.raw_text).strip()
            pymupdf_path = out_dir / f"{page_prefix}.pymupdf.md"
            pymupdf_path.write_text(pymupdf_text + ("\n" if pymupdf_text else ""), encoding="utf-8")

            score = float(analysis.text_quality.score)
            page_kind = analysis.page_kind.value

            ocr_required = args.force_ocr or _should_run_ocr(
                page_kind,
                score=score,
                threshold=float(args.pymupdf_score_threshold),
            )
            selected_source = "pymupdf"
            if page_kind == "empty":
                selected_source = "empty"
            elif ocr_required:
                selected_source = "ocr"

            ocr_path = out_dir / f"{page_prefix}.ocr.md"
            merged_path = out_dir / f"{page_prefix}.merged.md"
            meta_path = out_dir / f"{page_prefix}.meta.json"

            ocr_text = ""
            render_s: float | None = None
            ocr_s: float | None = None
            table_merge_stats: TableMergeStats | None = None
            merged_text = ""

            if selected_source == "ocr":
                page = doc.load_page(page_index)

                render_start = time.perf_counter()
                image = _render_page_to_image(page, dpi=dpi, max_edge=max_edge)
                render_s = time.perf_counter() - render_start

                ocr_start = time.perf_counter()
                ocr_text = _normalize_newlines(ocr_image(image)).strip()
                ocr_s = time.perf_counter() - ocr_start

                pages_ocr += 1
                merged_text = ocr_text

                if (
                    bool(args.table_merge)
                    and page_kind == "table"
                    and bool(analysis.has_text_layer)
                ):
                    threshold = float(args.pymupdf_score_threshold)
                    if score >= threshold:
                        try:
                            words = _extract_words(page)
                            merged_table_text, table_merge_stats = (
                                merge_markdown_tables_with_pymupdf_words(ocr_text, words)
                            )
                            if table_merge_stats.applied:
                                merged_text = merged_table_text
                        except Exception:  # noqa: BLE001
                            table_merge_stats = TableMergeStats(
                                applied=False,
                                changed_cells=0,
                                changed_chars=0,
                                confidence=None,
                                reason="exception",
                            )
                    else:
                        table_merge_stats = TableMergeStats(
                            applied=False,
                            changed_cells=0,
                            changed_chars=0,
                            confidence=None,
                            reason=f"text_quality<{threshold}",
                        )
            else:
                pages_skipped += 1

            if ocr_text:
                ocr_path.write_text(ocr_text + "\n", encoding="utf-8")
            else:
                ocr_path.write_text("", encoding="utf-8")

            if selected_source == "ocr":
                if not merged_text:
                    merged_text = ocr_text
            elif selected_source == "pymupdf":
                merged_text = pymupdf_text
            else:
                merged_text = ""

            merged_path.write_text(merged_text + ("\n" if merged_text else ""), encoding="utf-8")
            merged_pages.append(merged_text)

            if table_merge_stats is not None:
                table_merge_attempted_pages += 1
                if table_merge_stats.applied:
                    table_merge_applied_pages += 1
                table_merge_changed_cells_total += int(table_merge_stats.changed_cells)
                table_merge_changed_chars_total += int(table_merge_stats.changed_chars)

            replaced_tokens_estimated = 0
            replaced_chars_estimated = 0
            if selected_source == "ocr" and pymupdf_text and ocr_text:
                ocr_tokens = tokenize_by_char_class(normalize_extracted_text(ocr_text))
                pym_tokens = tokenize_by_char_class(pymupdf_text)
                try:
                    from difflib import SequenceMatcher

                    matcher = SequenceMatcher(a=ocr_tokens, b=pym_tokens, autojunk=False)
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag != "replace":
                            continue
                        replaced_tokens_estimated += max(i2 - i1, j2 - j1)
                        replaced_chars_estimated += max(
                            len("".join(ocr_tokens[i1:i2])),
                            len("".join(pym_tokens[j1:j2])),
                        )
                except Exception:  # noqa: BLE001
                    replaced_tokens_estimated = 0
                    replaced_chars_estimated = 0

            replacements_total_estimated += replaced_tokens_estimated

            if not args.no_diff and selected_source == "ocr" and pymupdf_text and ocr_text:
                try:
                    import difflib

                    diff_lines = difflib.unified_diff(
                        ocr_text.splitlines(),
                        pymupdf_text.splitlines(),
                        fromfile=f"{page_prefix}.ocr.md",
                        tofile=f"{page_prefix}.pymupdf.md",
                        lineterm="",
                    )
                    diff_path = out_dir / f"{page_prefix}.diff.txt"
                    diff_path.write_text("\n".join(diff_lines) + "\n", encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass

            meta = {
                "page_number": analysis.page_number,
                "page_kind": page_kind,
                "selected_source": selected_source,
                "ocr_required": bool(ocr_required),
                "ocr_skipped": selected_source != "ocr",
                "pymupdf": {
                    "score": score,
                    "visible_char_count": analysis.text_quality.visible_char_count,
                    "replacement_char_ratio": analysis.text_quality.replacement_char_ratio,
                    "symbol_ratio": analysis.text_quality.symbol_ratio,
                },
                "table_likelihood": analysis.table_likelihood,
                "image_count": analysis.image_count,
                "image_area_ratio": analysis.image_area_ratio,
                "table_merge": {
                    "attempted": table_merge_stats is not None,
                    "applied": bool(table_merge_stats.applied) if table_merge_stats else False,
                    "changed_cells": (
                        int(table_merge_stats.changed_cells) if table_merge_stats is not None else 0
                    ),
                    "changed_chars": (
                        int(table_merge_stats.changed_chars) if table_merge_stats is not None else 0
                    ),
                    "confidence": (
                        float(table_merge_stats.confidence)
                        if (
                            table_merge_stats is not None
                            and table_merge_stats.confidence is not None
                        )
                        else None
                    ),
                    "reason": table_merge_stats.reason if table_merge_stats is not None else None,
                    "samples": list(table_merge_stats.samples)
                    if table_merge_stats is not None
                    else [],
                },
                "timing_s": {"render": render_s, "ocr": ocr_s},
                "diff_estimate": {
                    "replaced_tokens": replaced_tokens_estimated,
                    "replaced_chars": replaced_chars_estimated,
                },
            }
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    merged_markdown = "\n\n".join(p.strip() for p in merged_pages if p.strip()).strip()
    merged_out = out_dir / "merged.md"
    merged_out.write_text(merged_markdown + ("\n" if merged_markdown else ""), encoding="utf-8")

    summary = {
        "pdf": str(args.pdf),
        "out_dir": str(out_dir),
        "settings": {"dpi": dpi, "max_edge": max_edge},
        "flags": {
            "use_find_tables": bool(args.use_find_tables),
            "pymupdf_score_threshold": float(args.pymupdf_score_threshold),
            "force_ocr": bool(args.force_ocr),
            "table_merge": bool(args.table_merge),
        },
        "pages_total": pages_total,
        "pages_ocr": pages_ocr,
        "pages_skipped": pages_skipped,
        "pages_table": pages_table,
        "pages_image": pages_image,
        "pages_mixed": pages_mixed,
        "replacements_total_estimated": replacements_total_estimated,
        "table_merge": {
            "attempted_pages": table_merge_attempted_pages,
            "applied_pages": table_merge_applied_pages,
            "changed_cells_total": table_merge_changed_cells_total,
            "changed_chars_total": table_merge_changed_chars_total,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"out_dir: {out_dir}")
    print(
        "pages:"
        f" total={pages_total}"
        f" ocr={pages_ocr}"
        f" skipped={pages_skipped}"
        f" table={pages_table}"
        f" image={pages_image}"
        f" mixed={pages_mixed}"
    )
    print(f"replacements_total_estimated={replacements_total_estimated}")
    print(f"merged_md: {merged_out}")

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

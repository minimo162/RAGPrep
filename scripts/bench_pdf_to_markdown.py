from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from PIL import Image


@dataclass(frozen=True)
class BenchRun:
    page_count: int
    pages_ocr: int
    pages_skipped: int
    pages_table: int
    pages_image: int
    pages_mixed: int
    replacements_total_estimated: int | None
    table_merge_attempted_pages: int | None
    table_merge_applied_pages: int | None
    table_merge_changed_cells_total: int | None
    table_merge_changed_chars_total: int | None
    read_s: float
    open_s: float
    analyze_s: float
    render_s: float
    render_pages_s: list[float]
    ocr_s: float
    ocr_pages_s: list[float]
    total_s: float


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark: PDF render + OCR timing breakdown")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", type=Path, help="Path to a local PDF file")
    group.add_argument(
        "--synthetic-pages",
        type=int,
        help="Generate a synthetic PDF with N pages and benchmark it",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of benchmark runs (default: 1)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Hybrid mode: skip OCR on high-quality text pages (table/image pages still OCR)",
    )
    parser.add_argument(
        "--pymupdf-score-threshold",
        type=float,
        default=0.15,
        help=(
            "Hybrid threshold: skip OCR when page_kind=text and score >= threshold (default: 0.15)"
        ),
    )
    parser.add_argument(
        "--use-find-tables",
        action="store_true",
        help="Use PyMuPDF page.find_tables() as a weak hint (can be inaccurate)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "Optional: read per-page meta JSONs from this directory to report "
            "replacements_total_estimated"
        ),
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
    parser.add_argument("--max-pages", type=int, default=None, help="Override max pages limit")
    parser.add_argument("--max-bytes", type=int, default=None, help="Override max bytes limit")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON (last run + averages)",
    )
    return parser.parse_args(argv)


def _make_synthetic_pdf_bytes(page_count: int) -> bytes:
    if page_count <= 0:
        raise ValueError("--synthetic-pages must be > 0")

    import fitz

    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 72), f"Hello {i + 1}")
    return doc.tobytes()


def _read_pdf_bytes(pdf_path: Path) -> tuple[bytes, float]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Not a file: {pdf_path}")

    start = time.perf_counter()
    data = pdf_path.read_bytes()
    return data, time.perf_counter() - start


def _open_pdf(pdf_bytes: bytes) -> tuple[object, int, float]:
    import fitz

    start = time.perf_counter()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid PDF data") from exc
    elapsed = time.perf_counter() - start
    return doc, int(doc.page_count), elapsed


def _render_doc_to_images(
    doc: object,
    *,
    dpi: int,
    max_edge: int,
) -> tuple[list[Image.Image], list[float]]:
    import fitz

    from ragprep.pdf_render import _pixmap_to_rgb_image

    if not hasattr(doc, "load_page"):
        raise TypeError("doc must be a PyMuPDF document")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    images: list[Image.Image] = []
    per_page: list[float] = []

    page_count = int(getattr(doc, "page_count", 0))
    for page_index in range(page_count):
        start = time.perf_counter()
        page = doc.load_page(page_index)
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

        images.append(rgb)
        per_page.append(time.perf_counter() - start)

    return images, per_page


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


def _run_once(
    pdf_bytes: bytes,
    *,
    read_s: float,
    dpi: int,
    max_edge: int,
    max_pages: int,
    max_bytes: int,
    hybrid: bool,
    pymupdf_score_threshold: float,
    use_find_tables: bool,
    artifacts_dir: Path | None,
) -> BenchRun:
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")
    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    if max_edge <= 0:
        raise ValueError("max_edge must be > 0")
    if max_pages <= 0:
        raise ValueError("max_pages must be > 0")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if len(pdf_bytes) > max_bytes:
        raise ValueError(f"PDF too large ({len(pdf_bytes)} bytes), max_bytes={max_bytes}")

    total_start = time.perf_counter()

    analyze_start = time.perf_counter()
    try:
        from ragprep.pdf_text import analyze_pdf_pages
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import PyMuPDF analysis module: {exc}") from exc

    analyses = analyze_pdf_pages(pdf_bytes, use_find_tables=use_find_tables)
    analyze_s = time.perf_counter() - analyze_start

    pages_table = sum(1 for p in analyses if p.page_kind.value == "table")
    pages_image = sum(1 for p in analyses if p.page_kind.value == "image")
    pages_mixed = sum(1 for p in analyses if p.page_kind.value == "mixed")

    replacements_total_estimated: int | None = None
    table_merge_attempted_pages: int | None = None
    table_merge_applied_pages: int | None = None
    table_merge_changed_cells_total: int | None = None
    table_merge_changed_chars_total: int | None = None
    if artifacts_dir is not None:
        try:
            import json

            total = 0
            tm_attempted = 0
            tm_applied = 0
            tm_cells = 0
            tm_chars = 0
            for meta_path in sorted(artifacts_dir.glob("page-*.meta.json")):
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                diff_est = data.get("diff_estimate")
                if isinstance(diff_est, dict):
                    value = diff_est.get("replaced_tokens")
                    if isinstance(value, int):
                        total += value
                value2 = data.get("replacements_total")
                if isinstance(value2, int):
                    total += value2

                table_merge = data.get("table_merge")
                if isinstance(table_merge, dict):
                    if table_merge.get("attempted") is True:
                        tm_attempted += 1
                    if table_merge.get("applied") is True:
                        tm_applied += 1
                    changed_cells = table_merge.get("changed_cells")
                    if isinstance(changed_cells, int):
                        tm_cells += changed_cells
                    changed_chars = table_merge.get("changed_chars")
                    if isinstance(changed_chars, int):
                        tm_chars += changed_chars
            replacements_total_estimated = total
            table_merge_attempted_pages = tm_attempted
            table_merge_applied_pages = tm_applied
            table_merge_changed_cells_total = tm_cells
            table_merge_changed_chars_total = tm_chars
        except Exception:  # noqa: BLE001
            replacements_total_estimated = None
            table_merge_attempted_pages = None
            table_merge_applied_pages = None
            table_merge_changed_cells_total = None
            table_merge_changed_chars_total = None

    doc, page_count, open_s = _open_pdf(pdf_bytes)
    if page_count > max_pages:
        raise ValueError(f"PDF has {page_count} pages, max_pages={max_pages}")

    try:
        from ragprep.ocr.lightonocr import ocr_image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import OCR module: {exc}") from exc

    ocr_pages_s: list[float] = []
    render_pages_s: list[float] = []
    pages_ocr = 0
    pages_skipped = 0

    with doc:
        render_start = time.perf_counter()
        ocr_start = time.perf_counter()

        if not hybrid:
            images, render_pages_s = _render_doc_to_images(doc, dpi=dpi, max_edge=max_edge)
            render_s = time.perf_counter() - render_start

            for image in images:
                page_start = time.perf_counter()
                _ = ocr_image(image)
                ocr_pages_s.append(time.perf_counter() - page_start)
            ocr_s = time.perf_counter() - ocr_start

            pages_ocr = page_count
            pages_skipped = 0
        else:
            for analysis in analyses:
                page_kind = analysis.page_kind.value
                score = float(analysis.text_quality.score)
                ocr_required = _should_run_ocr(
                    page_kind,
                    score=score,
                    threshold=pymupdf_score_threshold,
                )
                if not ocr_required:
                    pages_skipped += 1
                    continue

                page = doc.load_page(int(analysis.page_number) - 1)

                page_render_start = time.perf_counter()
                image = _render_page_to_image(page, dpi=dpi, max_edge=max_edge)
                render_pages_s.append(time.perf_counter() - page_render_start)

                page_ocr_start = time.perf_counter()
                _ = ocr_image(image)
                ocr_pages_s.append(time.perf_counter() - page_ocr_start)

                pages_ocr += 1

            render_s = sum(render_pages_s)
            ocr_s = sum(ocr_pages_s)

    total_s = time.perf_counter() - total_start
    return BenchRun(
        page_count=page_count,
        pages_ocr=pages_ocr,
        pages_skipped=pages_skipped,
        pages_table=pages_table,
        pages_image=pages_image,
        pages_mixed=pages_mixed,
        replacements_total_estimated=replacements_total_estimated,
        table_merge_attempted_pages=table_merge_attempted_pages,
        table_merge_applied_pages=table_merge_applied_pages,
        table_merge_changed_cells_total=table_merge_changed_cells_total,
        table_merge_changed_chars_total=table_merge_changed_chars_total,
        read_s=read_s,
        open_s=open_s,
        analyze_s=analyze_s,
        render_s=render_s,
        render_pages_s=render_pages_s,
        ocr_s=ocr_s,
        ocr_pages_s=ocr_pages_s,
        total_s=total_s,
    )


def _print_run(run: BenchRun, *, label: str) -> None:
    print(f"[{label}] pages={run.page_count}")
    print(
        "page_stats:"
        f" ocr={run.pages_ocr}"
        f" skipped={run.pages_skipped}"
        f" table={run.pages_table}"
        f" image={run.pages_image}"
        f" mixed={run.pages_mixed}"
    )
    if run.replacements_total_estimated is not None:
        print(f"replacements_total_estimated={run.replacements_total_estimated}")
    else:
        print("replacements_total_estimated=n/a")
    if run.table_merge_attempted_pages is not None:
        print(
            "table_merge:"
            f" attempted_pages={run.table_merge_attempted_pages}"
            f" applied_pages={run.table_merge_applied_pages}"
            f" changed_cells_total={run.table_merge_changed_cells_total}"
            f" changed_chars_total={run.table_merge_changed_chars_total}"
        )
    else:
        print("table_merge=n/a")
    print(
        "timing_s:"
        f" read={run.read_s:.3f}"
        f" open={run.open_s:.3f}"
        f" analyze={run.analyze_s:.3f}"
        f" render={run.render_s:.3f}"
        f" ocr={run.ocr_s:.3f}"
        f" total={run.total_s:.3f}"
    )
    if run.render_pages_s:
        render_avg = mean(run.render_pages_s)
        render_max = max(run.render_pages_s)
        print(
            f"render_pages_s: min={min(run.render_pages_s):.3f}"
            f" avg={render_avg:.3f}"
            f" max={render_max:.3f}"
        )
    if run.ocr_pages_s:
        ocr_avg = mean(run.ocr_pages_s)
        ocr_max = max(run.ocr_pages_s)
        print(f"ocr_pages_s:    min={min(run.ocr_pages_s):.3f} avg={ocr_avg:.3f} max={ocr_max:.3f}")


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
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to import settings: {exc}", file=sys.stderr)
        return 1

    settings = get_settings()
    dpi = settings.render_dpi if args.dpi is None else args.dpi
    max_edge = settings.render_max_edge if args.max_edge is None else args.max_edge
    max_pages = settings.max_pages if args.max_pages is None else args.max_pages
    max_bytes = settings.max_upload_bytes if args.max_bytes is None else args.max_bytes

    source: str
    try:
        if args.pdf is not None:
            pdf_bytes, read_s = _read_pdf_bytes(args.pdf)
            source = str(args.pdf)
        else:
            pdf_bytes = _make_synthetic_pdf_bytes(int(args.synthetic_pages))
            read_s = 0.0
            source = f"synthetic:{int(args.synthetic_pages)}"
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    repeat = int(args.repeat)
    if repeat <= 0:
        print("ERROR: --repeat must be > 0", file=sys.stderr)
        return 2

    print(f"source: {source}")
    print(f"env: LIGHTONOCR_DRY_RUN={os.getenv('LIGHTONOCR_DRY_RUN')!r}")
    print(f"env: LIGHTONOCR_BACKEND={os.getenv('LIGHTONOCR_BACKEND')!r}")
    artifacts_dir = str(args.artifacts_dir) if args.artifacts_dir else None
    print(
        f"settings: dpi={dpi} max_edge={max_edge} max_pages={max_pages} max_bytes={max_bytes}"
        f" hybrid={bool(args.hybrid)} pymupdf_score_threshold={float(args.pymupdf_score_threshold)}"
        f" use_find_tables={bool(args.use_find_tables)} artifacts_dir={artifacts_dir}"
    )
    print("")

    runs: list[BenchRun] = []
    for i in range(repeat):
        try:
            run = _run_once(
                pdf_bytes,
                read_s=read_s,
                dpi=int(dpi),
                max_edge=int(max_edge),
                max_pages=int(max_pages),
                max_bytes=int(max_bytes),
                hybrid=bool(args.hybrid),
                pymupdf_score_threshold=float(args.pymupdf_score_threshold),
                use_find_tables=bool(args.use_find_tables),
                artifacts_dir=args.artifacts_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: Benchmark failed: {exc}", file=sys.stderr)
            return 1
        runs.append(run)
        _print_run(run, label=f"run {i + 1}/{repeat}")
        print("")

    avg = BenchRun(
        page_count=runs[-1].page_count,
        pages_ocr=round(mean(r.pages_ocr for r in runs)),
        pages_skipped=round(mean(r.pages_skipped for r in runs)),
        pages_table=runs[-1].pages_table,
        pages_image=runs[-1].pages_image,
        pages_mixed=runs[-1].pages_mixed,
        replacements_total_estimated=runs[-1].replacements_total_estimated,
        table_merge_attempted_pages=runs[-1].table_merge_attempted_pages,
        table_merge_applied_pages=runs[-1].table_merge_applied_pages,
        table_merge_changed_cells_total=runs[-1].table_merge_changed_cells_total,
        table_merge_changed_chars_total=runs[-1].table_merge_changed_chars_total,
        read_s=mean(r.read_s for r in runs),
        open_s=mean(r.open_s for r in runs),
        analyze_s=mean(r.analyze_s for r in runs),
        render_s=mean(r.render_s for r in runs),
        render_pages_s=[],
        ocr_s=mean(r.ocr_s for r in runs),
        ocr_pages_s=[],
        total_s=mean(r.total_s for r in runs),
    )
    _print_run(avg, label=f"avg ({repeat} runs)")

    if args.json:
        import json

        def as_dict(r: BenchRun) -> dict[str, object]:
            return {
                "page_count": r.page_count,
                "pages_ocr": r.pages_ocr,
                "pages_skipped": r.pages_skipped,
                "pages_table": r.pages_table,
                "pages_image": r.pages_image,
                "pages_mixed": r.pages_mixed,
                "replacements_total_estimated": r.replacements_total_estimated,
                "table_merge_attempted_pages": r.table_merge_attempted_pages,
                "table_merge_applied_pages": r.table_merge_applied_pages,
                "table_merge_changed_cells_total": r.table_merge_changed_cells_total,
                "table_merge_changed_chars_total": r.table_merge_changed_chars_total,
                "read_s": r.read_s,
                "open_s": r.open_s,
                "analyze_s": r.analyze_s,
                "render_s": r.render_s,
                "render_pages_s": r.render_pages_s,
                "ocr_s": r.ocr_s,
                "ocr_pages_s": r.ocr_pages_s,
                "total_s": r.total_s,
            }

        payload = {
            "source": source,
            "env": {
                "LIGHTONOCR_DRY_RUN": os.getenv("LIGHTONOCR_DRY_RUN"),
                "LIGHTONOCR_BACKEND": os.getenv("LIGHTONOCR_BACKEND"),
            },
            "settings": {
                "dpi": dpi,
                "max_edge": max_edge,
                "max_pages": max_pages,
                "max_bytes": max_bytes,
            },
            "last": as_dict(runs[-1]),
            "avg": as_dict(avg),
        }
        print("")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

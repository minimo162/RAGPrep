from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class BenchRun:
    page_count: int
    convert_s: float


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark: PDF -> HTML total conversion time")

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


def _read_pdf_bytes(pdf_path: Path) -> bytes:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Not a file: {pdf_path}")
    return pdf_path.read_bytes()


def _count_pages(pdf_bytes: bytes) -> int:
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    with doc:
        return int(doc.page_count)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    try:
        from ragprep.config import get_settings
        from ragprep.pipeline import pdf_to_html
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to import app modules: {exc}", file=sys.stderr)
        return 1

    try:
        if args.pdf is not None:
            pdf_bytes = _read_pdf_bytes(args.pdf)
        else:
            pdf_bytes = _make_synthetic_pdf_bytes(int(args.synthetic_pages))
        page_count = _count_pages(pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to prepare PDF bytes: {exc}", file=sys.stderr)
        return 2

    if page_count <= 0:
        print("ERROR: PDF has no pages", file=sys.stderr)
        return 2

    settings = get_settings()
    runs: list[BenchRun] = []
    for _ in range(int(args.repeat)):
        start = time.perf_counter()
        _ = pdf_to_html(pdf_bytes, full_document=False)
        elapsed = time.perf_counter() - start
        runs.append(BenchRun(page_count=page_count, convert_s=elapsed))

    convert_mean = mean(r.convert_s for r in runs)
    convert_min = min(r.convert_s for r in runs)
    convert_max = max(r.convert_s for r in runs)
    s_per_page = convert_mean / max(1, page_count)

    print(
        f"pages={page_count} backend={settings.pdf_backend} repeat={int(args.repeat)} "
        f"convert_s_mean={convert_mean:.3f} min={convert_min:.3f} max={convert_max:.3f} "
        f"s_per_page={s_per_page:.3f}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

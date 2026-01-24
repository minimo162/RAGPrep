from __future__ import annotations

import argparse
import io
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
    read_s: float
    open_s: float
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
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pix.tobytes("png")
        with io.BytesIO(png_bytes) as buf:
            pil_image = Image.open(buf)
            pil_image.load()
            rgb = pil_image.convert("RGB")

        width, height = rgb.size
        current_max_edge = max(width, height)
        if current_max_edge != max_edge:
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


def _run_once(
    pdf_bytes: bytes,
    *,
    read_s: float,
    dpi: int,
    max_edge: int,
    max_pages: int,
    max_bytes: int,
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

    doc, page_count, open_s = _open_pdf(pdf_bytes)
    if page_count > max_pages:
        raise ValueError(f"PDF has {page_count} pages, max_pages={max_pages}")

    with doc:
        render_start = time.perf_counter()
        images, render_pages_s = _render_doc_to_images(doc, dpi=dpi, max_edge=max_edge)
        render_s = time.perf_counter() - render_start

    try:
        from ragprep.ocr.lightonocr import ocr_image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import OCR module: {exc}") from exc

    ocr_pages_s: list[float] = []
    ocr_start = time.perf_counter()
    for image in images:
        page_start = time.perf_counter()
        _ = ocr_image(image)
        ocr_pages_s.append(time.perf_counter() - page_start)
    ocr_s = time.perf_counter() - ocr_start

    total_s = time.perf_counter() - total_start
    return BenchRun(
        page_count=page_count,
        read_s=read_s,
        open_s=open_s,
        render_s=render_s,
        render_pages_s=render_pages_s,
        ocr_s=ocr_s,
        ocr_pages_s=ocr_pages_s,
        total_s=total_s,
    )


def _print_run(run: BenchRun, *, label: str) -> None:
    print(f"[{label}] pages={run.page_count}")
    print(
        "timing_s:"
        f" read={run.read_s:.3f}"
        f" open={run.open_s:.3f}"
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
    print(f"settings: dpi={dpi} max_edge={max_edge} max_pages={max_pages} max_bytes={max_bytes}")
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
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: Benchmark failed: {exc}", file=sys.stderr)
            return 1
        runs.append(run)
        _print_run(run, label=f"run {i + 1}/{repeat}")
        print("")

    avg = BenchRun(
        page_count=runs[-1].page_count,
        read_s=mean(r.read_s for r in runs),
        open_s=mean(r.open_s for r in runs),
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
                "read_s": r.read_s,
                "open_s": r.open_s,
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

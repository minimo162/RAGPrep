from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual smoke test: PDF -> images -> LightOnOCR -> markdown",
    )
    parser.add_argument("pdf", type=Path, help="Path to a local PDF file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path (default: <pdf>.md)",
    )
    parser.add_argument("--dpi", type=int, default=None, help="Override render DPI")
    parser.add_argument("--max-pages", type=int, default=None, help="Override max pages limit")
    parser.add_argument("--max-bytes", type=int, default=None, help="Override max bytes limit")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set LIGHTONOCR_DRY_RUN=1 (skips real inference)",
    )
    parser.add_argument("--model-id", type=str, default=None, help="Set LIGHTONOCR_MODEL_ID")
    parser.add_argument("--device", type=str, default=None, help="Set LIGHTONOCR_DEVICE")
    parser.add_argument("--dtype", type=str, default=None, help="Set LIGHTONOCR_DTYPE")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Set LIGHTONOCR_MAX_NEW_TOKENS",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return 2
    if not pdf_path.is_file():
        print(f"ERROR: Not a file: {pdf_path}", file=sys.stderr)
        return 2

    out_path = args.out if args.out is not None else pdf_path.with_suffix(".md")
    if out_path.exists() and not args.overwrite:
        print(
            f"ERROR: Output already exists: {out_path} (use --overwrite to replace)",
            file=sys.stderr,
        )
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        os.environ["LIGHTONOCR_DRY_RUN"] = "1"
    if args.model_id:
        os.environ["LIGHTONOCR_MODEL_ID"] = args.model_id
    if args.device:
        os.environ["LIGHTONOCR_DEVICE"] = args.device
    if args.dtype:
        os.environ["LIGHTONOCR_DTYPE"] = args.dtype
    if args.max_new_tokens is not None:
        os.environ["LIGHTONOCR_MAX_NEW_TOKENS"] = str(args.max_new_tokens)

    try:
        pdf_bytes = pdf_path.read_bytes()
    except OSError as exc:
        print(f"ERROR: Failed to read PDF: {exc}", file=sys.stderr)
        return 1

    try:
        from ragprep.ocr.lightonocr import ocr_image
        from ragprep.pdf_render import render_pdf_to_images
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to import app modules: {exc}", file=sys.stderr)
        return 1

    try:
        images = render_pdf_to_images(
            pdf_bytes,
            dpi=args.dpi,
            max_pages=args.max_pages,
            max_bytes=args.max_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to render PDF: {exc}", file=sys.stderr)
        return 1

    page_texts: list[str] = []
    for i, image in enumerate(images, start=1):
        print(f"OCR page {i}/{len(images)}...", file=sys.stderr)
        try:
            text = ocr_image(image)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: OCR failed on page {i}: {exc}", file=sys.stderr)
            return 1
        text = _normalize_newlines(text).strip()
        if text:
            page_texts.append(text)

    markdown = "\n\n".join(page_texts).strip() + "\n"
    try:
        out_path.write_text(markdown, encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"ERROR: Failed to write output: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

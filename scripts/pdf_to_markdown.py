from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF -> Markdown (pymupdf-layout + pymupdf4llm)",
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Path to a local PDF file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path (default: <pdf>.md)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write markdown to stdout instead of a file",
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

    if args.stdout and args.out is not None:
        print("ERROR: Cannot use --stdout with --out", file=sys.stderr)
        return 2

    out_path = args.out if args.out is not None else pdf_path.with_suffix(".md")
    if not args.stdout:
        if out_path.exists() and not args.overwrite:
            print(
                f"ERROR: Output already exists: {out_path} (use --overwrite to replace)",
                file=sys.stderr,
            )
            return 2
        out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pdf_bytes = pdf_path.read_bytes()
    except OSError as exc:
        print(f"ERROR: Failed to read PDF: {exc}", file=sys.stderr)
        return 1

    try:
        from ragprep.pipeline import pdf_to_markdown
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to import app modules: {exc}", file=sys.stderr)
        return 1

    try:
        markdown = pdf_to_markdown(pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to convert PDF: {exc}", file=sys.stderr)
        return 1

    output = markdown + ("\n" if markdown else "")
    if args.stdout:
        sys.stdout.write(output)
        return 0

    try:
        out_path.write_text(output, encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"ERROR: Failed to write output: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


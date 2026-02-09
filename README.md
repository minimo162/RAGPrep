# RAGPrep

RAGPrep converts PDFs to structured HTML through one fixed path:

1. Render PDF pages to images.
2. OCR with `noctrex/LightOnOCR-2-1B-GGUF` (`LightOnOCR-2-1B-IQ4_XS.gguf` + `mmproj-BF16.gguf`).
3. Correct OCR text with the PyMuPDF text layer (`strict` merge policy).
4. Build HTML output.

No fallback OCR/layout path is used.

## Requirements

- Python `>=3.11,<3.13`
- `llama-server` binary from `llama.cpp` available on PATH
  - or set `RAGPREP_LLAMA_SERVER_PATH`
- Network access on first run if model files are not present locally

## Install

```bash
uv sync --dev
```

## Main settings (LightOn only)

- `RAGPREP_PDF_BACKEND=lighton-ocr`
- `RAGPREP_LIGHTON_REPO_ID=noctrex/LightOnOCR-2-1B-GGUF`
- `RAGPREP_LIGHTON_MODEL_FILE=LightOnOCR-2-1B-IQ4_XS.gguf`
- `RAGPREP_LIGHTON_MMPROJ_FILE=mmproj-BF16.gguf`
- `RAGPREP_LIGHTON_MODEL_DIR=~/.ragprep/models/lighton`
- `RAGPREP_LIGHTON_AUTO_DOWNLOAD=1` (auto-download missing files)
- `RAGPREP_LLAMA_SERVER_PATH` (optional explicit `llama-server` path)
- `RAGPREP_LIGHTON_MERGE_POLICY=strict`

Performance defaults are aggressive:

- GPU-first (`RAGPREP_LIGHTON_N_GPU_LAYERS=-1`)
- `RAGPREP_LIGHTON_PARALLEL=4`
- `RAGPREP_LIGHTON_FLASH_ATTN=1`
- page-level OCR concurrency (`RAGPREP_LIGHTON_PAGE_CONCURRENCY=4`)

If GPU startup fails, startup retries in this order:

1. GPU aggressive
2. GPU conservative (`flash-attn` off)
3. CPU fallback (`-ngl 0`, `-np 1`)

## Run

Desktop:

```bash
uv run python -m ragprep.desktop
```

Web:

```bash
uv run uvicorn ragprep.web.app:app --reload
```

Open `http://127.0.0.1:8000`.

CLI (PDF -> HTML):

```bash
uv run python scripts/pdf_to_html.py --pdf .\\path\\to\\input.pdf --out .\\out\\input.html --overwrite
```

## Failure behavior

- If a page OCR call fails, the conversion job fails immediately.
- If model files are missing and auto-download is disabled, conversion fails.
- If `llama-server` is not found, conversion fails with a clear error.

## Quality gate

```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

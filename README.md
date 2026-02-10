# RAGPrep

RAGPrep is a PDF-to-HTML converter focused on financial/IR style documents.
It uses LightOn OCR plus PyMuPDF-based correction to produce structured HTML tables and readable text.

## Quick Start (Windows / PowerShell)

```powershell
cd C:\Users\Administrator\RAGPrep
uv sync --dev
uv run python -m ragprep.desktop
```

Web mode:

```powershell
uv run uvicorn ragprep.web.app:app --reload
```

Open `http://127.0.0.1:8000`.

CLI conversion:

```powershell
uv run python scripts/pdf_to_html.py --pdf .\path\to\input.pdf --out .\out\input.html --overwrite
```

## Runtime Behavior

- PDF pages are rendered to images, OCR'd with LightOn, then corrected with PyMuPDF text.
- Truncated OCR table tails are repaired with a secondary lower-page OCR pass (best effort).
- Table text may be corrected from PyMuPDF word geometry when confidence is high.
- Output uses semantic HTML table tags (`<table>`, `<thead>`, `<tbody>`).

## Automatic Defaults

If not explicitly set, these stable defaults are applied automatically:

- `RAGPREP_LIGHTON_START_TIMEOUT_SECONDS=300`
- `RAGPREP_LIGHTON_REQUEST_TIMEOUT_SECONDS=600`
- `RAGPREP_LIGHTON_PARALLEL=2`
- `RAGPREP_LIGHTON_PAGE_CONCURRENCY=2`
- `RAGPREP_LIGHTON_N_GPU_LAYERS=-1`
- `RAGPREP_LIGHTON_FLASH_ATTN=1`
- `RAGPREP_LIGHTON_FAST_PASS=1`
- `RAGPREP_LIGHTON_FAST_RENDER_DPI=200`
- `RAGPREP_LIGHTON_FAST_RENDER_MAX_EDGE=1100`
- `RAGPREP_LIGHTON_FAST_RETRY=0`
- `RAGPREP_LIGHTON_SECONDARY_TABLE_REPAIR=0`
- `RAGPREP_LIGHTON_RETRY_RENDER_DPI=220`
- `RAGPREP_LIGHTON_RETRY_RENDER_MAX_EDGE=1280`
- `RAGPREP_LIGHTON_RETRY_MIN_QUALITY=0.40`
- `RAGPREP_LIGHTON_RETRY_QUALITY_GAP=0.22`
- `RAGPREP_LIGHTON_RETRY_MIN_PYM_TEXT_LEN=80`
- `RAGPREP_LIGHTON_FAST_NON_TABLE_MAX_EDGE=960`
- `RAGPREP_LIGHTON_FAST_TABLE_LIKELIHOOD_THRESHOLD=0.60`
- `RAGPREP_LIGHTON_FAST_MAX_TOKENS_TEXT=4096`
- `RAGPREP_LIGHTON_FAST_MAX_TOKENS_TABLE=8192`
- `RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE=light` (`full` / `light` / `off`)

## llama-server Auto Setup

RAGPrep resolves `llama-server` in this order:

1. `RAGPREP_LLAMA_SERVER_PATH` (if set)
2. PATH / common local locations
3. Auto-download and install (enabled by default)

Related variables:

- `RAGPREP_LLAMA_SERVER_PATH` (explicit binary path)
- `RAGPREP_LLAMA_SERVER_AUTO_DOWNLOAD=1` (default)
- `RAGPREP_LLAMA_SERVER_INSTALL_DIR` (optional install destination)

## Model and OCR Configuration

- `RAGPREP_PDF_BACKEND=lighton-ocr`
- `RAGPREP_LIGHTON_REPO_ID=noctrex/LightOnOCR-2-1B-GGUF`
- `RAGPREP_LIGHTON_MODEL_FILE=LightOnOCR-2-1B-IQ4_XS.gguf`
- `RAGPREP_LIGHTON_MMPROJ_FILE=mmproj-BF16.gguf`
- `RAGPREP_LIGHTON_MODEL_DIR=~/.ragprep/models/lighton`
- `RAGPREP_LIGHTON_AUTO_DOWNLOAD=1`
- `RAGPREP_LIGHTON_MERGE_POLICY=strict` (`strict` or `aggressive`)
- `RAGPREP_LIGHTON_FAST_PASS=0` to disable fast-pass mode and use legacy single-pass OCR rendering.
- `RAGPREP_LIGHTON_FAST_RETRY=1` to enable high-resolution retry OCR for low-quality pages.
- `RAGPREP_LIGHTON_SECONDARY_TABLE_REPAIR=1` to enable secondary cropped OCR table-tail repair.
- `RAGPREP_LIGHTON_FAST_NON_TABLE_MAX_EDGE=0` to disable non-table downscale in fast-pass.
- `RAGPREP_LIGHTON_FAST_MAX_TOKENS_TEXT` / `RAGPREP_LIGHTON_FAST_MAX_TOKENS_TABLE` for page-type token budget.
- `RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE=off` for maximum throughput (least correction work).

## Failure Model

- If a primary page OCR call fails, the job fails.
- Secondary table-tail OCR repair is best-effort and will not fail the job by itself.
- If models are missing and auto-download is disabled, conversion fails.
- If `llama-server` cannot be resolved (including auto-setup failure), conversion fails with a clear error.

## Development

```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

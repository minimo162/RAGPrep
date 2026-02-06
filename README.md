# RAGPrep

RAGPrep converts PDFs to **structured HTML** by combining:
- **Text layer extraction** (PyMuPDF): uses the PDF's existing text layer (no OCR for text extraction).
- **Layout analysis (required)**: PP-DocLayout-V3 regions are used to structure the extracted text.

## Behavior highlights

- Layout-aware output with improved natural reading order.
- Table rendering supports merge-aware cell attributes (`colspan`/`rowspan`) when detected.
- Web/Desktop `.html` download with in-app success/failure/cancel feedback.
- Running-job partial output shows all processed pages so far (no last-N preview cap).
- Optional startup prewarm to reduce first-request cold start for local layout backend.
- `RAGPREP_LAYOUT_MODE` is defaulted to `local-paddle` (no env required for local mode).
- UI shows prewarm/conversion state and disables `Convert` while prewarm/conversion is in progress.

Outputs:
- Web / Desktop: download `.html`
- CLI: writes an `.html` file

## Quickstart

### 1) Install
```bash
uv sync --dev
```

### 2) Layout mode (default is already enabled)

RAGPrep **requires layout analysis**. By default, it runs with:

```text
RAGPREP_LAYOUT_MODE=local-paddle
RAGPREP_WEB_PREWARM_ON_STARTUP=1
```

Set env vars only when you want to override this behavior.

#### Option A: Local (no Docker) via PaddleOCR (recommended)
Install optional runtime:
```bash
uv pip install paddlepaddle paddleocr "paddlex[ocr]"
```

Set layout mode:
```powershell
$env:RAGPREP_LAYOUT_MODE = "local-paddle"
```
```bash
export RAGPREP_LAYOUT_MODE=local-paddle
```

#### Option B: Server (OpenAI-compatible `/v1/chat/completions`)
Point RAGPrep to a running server:
```powershell
$env:RAGPREP_LAYOUT_MODE = "server"
$env:RAGPREP_LAYOUT_BASE_URL = "http://127.0.0.1:8080"
$env:RAGPREP_LAYOUT_MODEL = "zai-org/GLM-OCR"
# optional:
# $env:RAGPREP_LAYOUT_API_KEY = "..."
```

### 3) Run
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

## Layout settings

- `RAGPREP_LAYOUT_MODE`: `local-paddle` (local PP-DocLayout-V3) or `server` (OpenAI-compatible HTTP)
- `RAGPREP_LAYOUT_BASE_URL`: server base URL (server mode only)
- `RAGPREP_LAYOUT_MODEL`: model name (server mode; kept for parity in local mode)
- `RAGPREP_LAYOUT_API_KEY`: bearer token (optional, server mode)
- `RAGPREP_LAYOUT_TIMEOUT_SECONDS`: request timeout in seconds (server mode; default: `60`)
- `RAGPREP_MODEL_CACHE_DIR`: shared local model cache directory used for Paddle/HuggingFace/Torch assets (default: OS cache dir under `ragprep/model-cache`)
- `RAGPREP_LAYOUT_CONCURRENCY`: number of in-flight layout requests in server mode (default: `1`)
- `RAGPREP_LAYOUT_RENDER_DPI`: DPI used for layout rendering (default: `250`)
- `RAGPREP_LAYOUT_RENDER_MAX_EDGE`: max edge for layout rendering (default: `768`)
- `RAGPREP_LAYOUT_RENDER_AUTO`: enable small-first layout rendering with one higher-res retry on empty results (server mode; default: `0`)
- `RAGPREP_LAYOUT_RENDER_AUTO_SMALL_DPI`: DPI for small-first pass (default: `250`)
- `RAGPREP_LAYOUT_RENDER_AUTO_SMALL_MAX_EDGE`: max edge for small-first pass (default: `768`)
- `RAGPREP_LAYOUT_RETRY_COUNT`: retry count for transient failures (server mode; default: `1`)
- `RAGPREP_LAYOUT_RETRY_BACKOFF_SECONDS`: base backoff seconds between retries (server mode; default: `0.0`)

Fast layout hint (server mode):
- Try `RAGPREP_LAYOUT_CONCURRENCY=2` and `RAGPREP_LAYOUT_RENDER_AUTO=1` first.
- Start with `RAGPREP_LAYOUT_RENDER_MAX_EDGE=768` for layout-only workloads.
- Very small `max_edge` values (for example `640`) can be slower depending on backend/device; if quality or speed regresses, try `1024`.

## Web settings

- `RAGPREP_WEB_PREWARM_ON_STARTUP`: pre-initialize local layout backend at app startup (`1`/`0`, default: `1`)
- `RAGPREP_WEB_PREWARM_EXECUTOR`: prewarm backend executor (`thread` or `process`).
  - default: `process` when `RAGPREP_DESKTOP_MODE=1`, otherwise `thread`
- `RAGPREP_WEB_PREWARM_TIMEOUT_SECONDS`: timeout for process-based prewarm stage2 (default: `120`)
- `RAGPREP_WEB_PREWARM_START_DELAY_SECONDS`: delay before prewarm starts (default: `0.35`)
- `RAGPREP_DESKTOP_MODE`: desktop launcher marker (`1` enables desktop-optimized defaults)
- Startup prewarm prepares cache directories and model artifacts in `RAGPREP_MODEL_CACHE_DIR`.
- Startup prewarm runs in two phases: `stage1` (cache prep) -> `stage2` (layout engine load).
- Partial output always accumulates all processed pages so far.
- Legacy `RAGPREP_WEB_PARTIAL_PREVIEW_PAGES` is no longer used.
- `Convert` button is locked during prewarm and while any conversion job is active.

## Download behavior

- Browser mode: `Download .html` triggers file download and shows status in the page.
- Desktop mode (pywebview): tries saving to Downloads first, then falls back to save dialog.

## Troubleshooting (layout server)

If you see `Layout analysis request timed out`:
- Confirm `RAGPREP_LAYOUT_BASE_URL` is reachable and supports `POST /v1/chat/completions`.
- Increase `RAGPREP_LAYOUT_TIMEOUT_SECONDS` (or reduce PDF pages / image size).
- If server backend is not required, switch to `RAGPREP_LAYOUT_MODE=local-paddle`.

## Troubleshooting (local paddle)

If local layout fails with a Paddle runtime error mentioning `ConvertPirAttribute2RuntimeAttribute` or
`onednn_instruction.cc`, disable OneDNN/MKLDNN and PIR API flags, then restart:

PowerShell:
```powershell
$env:FLAGS_use_mkldnn = "0"
$env:FLAGS_enable_pir_api = "0"
```

bash:
```bash
export FLAGS_use_mkldnn=0
export FLAGS_enable_pir_api=0
```

Or set one switch and restart:
```bash
export RAGPREP_LAYOUT_PADDLE_SAFE_MODE=1
```

## Quality gate
```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

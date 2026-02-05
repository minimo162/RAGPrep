# RAGPrep

RAGPrep converts PDFs to **structured HTML** by combining:
- **Text layer extraction** (PyMuPDF): uses the PDF’s existing text layer (no OCR needed for text)
- **Layout analysis (required)**: PP-DocLayout-V3 regions are used to structure the extracted text

Outputs:
- Web / Desktop: download `.html`
- CLI: writes an `.html` file

## Quickstart

### 1) Install
```bash
uv sync --dev
```

### 2) Enable layout analysis (required)

RAGPrep **requires layout analysis**. Choose one of the following backends.

#### Option A: Local (no Docker) via PaddleOCR (recommended)
Install the optional runtime:
```bash
uv pip install paddlepaddle paddleocr "paddlex[ocr]"
```

Set the layout mode:
```powershell
$env:RAGPREP_LAYOUT_MODE = "local-paddle"
```
```bash
export RAGPREP_LAYOUT_MODE=local-paddle
```

#### Option B: Server (OpenAI-compatible `/v1/chat/completions`)
Point RAGPrep at a running server (for example, GLM-OCR served behind an OpenAI-compatible API):
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

CLI (PDF → HTML):
```bash
uv run python scripts/pdf_to_html.py --pdf .\\path\\to\\input.pdf --out .\\out\\input.html --overwrite
```

## Layout settings

- `RAGPREP_LAYOUT_MODE`: `local-paddle` (local PP-DocLayout-V3) or `server` (OpenAI-compatible HTTP)
- `RAGPREP_LAYOUT_BASE_URL`: server base URL (server mode only)
- `RAGPREP_LAYOUT_MODEL`: model name (server mode; kept for parity in local mode)
- `RAGPREP_LAYOUT_API_KEY`: bearer token (optional, server mode)
- `RAGPREP_LAYOUT_TIMEOUT_SECONDS`: request timeout in seconds (server mode; default: `60`)
- `RAGPREP_LAYOUT_CONCURRENCY`: number of in-flight layout requests in server mode (default: `1`)
- `RAGPREP_LAYOUT_RENDER_DPI`: DPI used for layout rendering (default: `250`)
- `RAGPREP_LAYOUT_RENDER_MAX_EDGE`: max edge for layout rendering (default: `1024`)
- `RAGPREP_LAYOUT_RENDER_AUTO`: enable small-first layout rendering with a single higher-res retry on empty results (server mode; default: `0`)
- `RAGPREP_LAYOUT_RENDER_AUTO_SMALL_DPI`: DPI for the small-first pass (default: `250`)
- `RAGPREP_LAYOUT_RENDER_AUTO_SMALL_MAX_EDGE`: max edge for the small-first pass (default: `1024`)
- `RAGPREP_LAYOUT_RETRY_COUNT`: retry count for transient failures (server mode; default: `1`)
- `RAGPREP_LAYOUT_RETRY_BACKOFF_SECONDS`: base backoff in seconds between retries (server mode; default: `0.0`)

Fast layout (server mode):
- Try `RAGPREP_LAYOUT_CONCURRENCY=2` and `RAGPREP_LAYOUT_RENDER_AUTO=1` first.

## Web settings

- `RAGPREP_WEB_PARTIAL_PREVIEW_PAGES`: limit streaming “partial output” preview to the last N pages (default: `3`)

## Troubleshooting (layout server)

If you see `Layout analysis request timed out`:
- Confirm the server at `RAGPREP_LAYOUT_BASE_URL` is reachable and supports `POST /v1/chat/completions`.
- Try increasing `RAGPREP_LAYOUT_TIMEOUT_SECONDS` (or reducing PDF pages / image size limits).
- If you don't need a server backend, switch to `RAGPREP_LAYOUT_MODE=local-paddle`.

## Troubleshooting (local paddle)

If local layout fails with a Paddle runtime error mentioning `ConvertPirAttribute2RuntimeAttribute` or
`onednn_instruction.cc`, try disabling OneDNN/MKLDNN and PIR API flags and restart:

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

Or set a single switch and restart:
```bash
export RAGPREP_LAYOUT_PADDLE_SAFE_MODE=1
```

## Quality gate
```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

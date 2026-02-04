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
uv pip install paddlepaddle paddleocr
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

## Quality gate
```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

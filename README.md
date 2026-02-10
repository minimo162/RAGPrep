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
- `RAGPREP_LIGHTON_PARALLEL=1`
- `RAGPREP_LIGHTON_PAGE_CONCURRENCY=1`
- `RAGPREP_LIGHTON_N_GPU_LAYERS=-1`
- `RAGPREP_LIGHTON_FLASH_ATTN=1`

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

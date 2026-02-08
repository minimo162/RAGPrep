# RAGPrep

RAGPrep converts PDFs to structured HTML/Markdown with:
- LightOn OCR (OpenAI-compatible server)
- PyMuPDF text layer extraction (for line-level correction)
- Layout-aware rendering

## Default behavior

- Default OCR backend: `lighton-ocr`
- Default layout mode: `lighton`
- No fallback on OCR/layout failure: conversion fails fast with page-specific error

## Install

```bash
uv sync --dev
```

## Configure (LightOn default)

PowerShell:
```powershell
$env:RAGPREP_PDF_BACKEND = "lighton-ocr"
$env:RAGPREP_LAYOUT_MODE = "lighton"
$env:RAGPREP_LIGHTON_BASE_URL = "http://127.0.0.1:8080"
$env:RAGPREP_LIGHTON_MODEL = "noctrex/LightOnOCR-2-1B-GGUF"
```

bash:
```bash
export RAGPREP_PDF_BACKEND=lighton-ocr
export RAGPREP_LAYOUT_MODE=lighton
export RAGPREP_LIGHTON_BASE_URL=http://127.0.0.1:8080
export RAGPREP_LIGHTON_MODEL=noctrex/LightOnOCR-2-1B-GGUF
```

### LightOn response contract

`POST /v1/chat/completions` must return `message.content` containing JSON:

```json
{
  "schema_version": "v1",
  "elements": [
    {
      "page_index": 0,
      "bbox": [0, 0, 100, 100],
      "label": "text",
      "score": 0.9,
      "order": 0
    }
  ],
  "lines": [
    {
      "bbox": [0, 0, 100, 20],
      "text": "example",
      "confidence": 0.95
    }
  ]
}
```

- `elements` and `lines` are required.
- `bbox` must satisfy `x0 < x1` and `y0 < y1`.
- On invalid response, conversion fails immediately.

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

CLI (legacy PDF -> Markdown entrypoint, still supported):
```bash
uv run python scripts/pdf_to_markdown.py --pdf .\\path\\to\\input.pdf --out .\\out\\input.md --overwrite
```

## Main settings

### OCR backend
- `RAGPREP_PDF_BACKEND`: `lighton-ocr` (default) or `glm-ocr`
- `RAGPREP_OCR_BACKEND`: alias of `RAGPREP_PDF_BACKEND`

### LightOn
- `RAGPREP_LIGHTON_BASE_URL` (default: `http://127.0.0.1:8080`)
- `RAGPREP_LIGHTON_MODEL` (default: `noctrex/LightOnOCR-2-1B-GGUF`)
- `RAGPREP_LIGHTON_API_KEY` (optional)
- `RAGPREP_LIGHTON_MAX_TOKENS` (default: `8192`)
- `RAGPREP_LIGHTON_TIMEOUT_SECONDS` (default: `60`)

### Layout
- `RAGPREP_LAYOUT_MODE`: `lighton` (default), `local-fast`, `local-paddle`, or `server`
- `RAGPREP_LAYOUT_BASE_URL`
- `RAGPREP_LAYOUT_MODEL`
- `RAGPREP_LAYOUT_API_KEY`
- `RAGPREP_LAYOUT_MAX_TOKENS`
- `RAGPREP_LAYOUT_TIMEOUT_SECONDS`
- `RAGPREP_LAYOUT_CONCURRENCY` (server mode)
- `RAGPREP_LAYOUT_RENDER_DPI`
- `RAGPREP_LAYOUT_RENDER_MAX_EDGE`
- `RAGPREP_LAYOUT_RENDER_AUTO` (+ small-pass settings)

## Legacy GLM mode

`RAGPREP_GLM_OCR_*` settings are legacy and used only when:
- `RAGPREP_PDF_BACKEND=glm-ocr`, or
- layout mode explicitly uses legacy server/local-paddle flows.

## Standalone helper

`scripts/build-standalone.ps1` now generates:
- `start-lighton-ocr.ps1`
- `start-lighton-ocr.cmd`

It includes a llama.cpp example with:
- `LightOnOCR-2-1B-IQ4_XS.gguf`
- `mmproj-BF16.gguf`

## Quality gate

```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

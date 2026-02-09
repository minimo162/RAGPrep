# RAGPrep

RAGPrep converts PDFs to structured HTML/Markdown with local processing.

- PyMuPDF text layer extraction
- Local layout analysis (`local-fast` by default)
- Local OCR path for Markdown via GLM-OCR Transformers

## Default behavior

- `RAGPREP_PDF_BACKEND=glm-ocr`
- `RAGPREP_GLM_OCR_MODE=transformers`
- `RAGPREP_LAYOUT_MODE=local-fast`
- No external OCR/layout server required

## Install

```bash
uv sync --dev
```

## Configure (local default)

PowerShell:

```powershell
$env:RAGPREP_PDF_BACKEND = "glm-ocr"
$env:RAGPREP_GLM_OCR_MODE = "transformers"
$env:RAGPREP_LAYOUT_MODE = "local-fast"
```

bash:

```bash
export RAGPREP_PDF_BACKEND=glm-ocr
export RAGPREP_GLM_OCR_MODE=transformers
export RAGPREP_LAYOUT_MODE=local-fast
```

Optional local layout model path:

```powershell
$env:RAGPREP_LAYOUT_MODE = "local-paddle"
```

```bash
export RAGPREP_LAYOUT_MODE=local-paddle
```

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

CLI (PDF -> Markdown):

```bash
uv run python scripts/pdf_to_markdown.py --pdf .\\path\\to\\input.pdf --out .\\out\\input.md --overwrite
```

## Main settings

### OCR backend

- `RAGPREP_PDF_BACKEND`: `glm-ocr` (default)
- `RAGPREP_OCR_BACKEND`: alias of `RAGPREP_PDF_BACKEND`
- `RAGPREP_GLM_OCR_MODE`: `transformers` (default)
- `RAGPREP_GLM_OCR_MODEL`: model id (default: `zai-org/GLM-OCR`)
- `RAGPREP_GLM_OCR_MAX_TOKENS` (default: `8192`)
- `RAGPREP_GLM_OCR_TIMEOUT_SECONDS` (default: `60`)

### Layout

- `RAGPREP_LAYOUT_MODE`: `local-fast` (default), `local-paddle`, `transformers` (alias of `local-fast`)
- `RAGPREP_LAYOUT_RENDER_DPI` (default: `250`)
- `RAGPREP_LAYOUT_RENDER_MAX_EDGE` (default: `1024`)
- `RAGPREP_LAYOUT_RENDER_AUTO` (+ small-pass settings)

## Quality gate

```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

# RAGPrep

PDF to Markdown app scaffold (FastAPI + htmx) backed by `lightonai/LightOnOCR-2-1B`.

## Dev setup
- Install `uv`: https://docs.astral.sh/uv/
- Install deps: `uv sync --dev`
- Run server: `uv run uvicorn ragprep.web.app:app --reload`
- Open: `http://127.0.0.1:8000`

## Real OCR verification (manual)
Runs PDF -> images -> LightOnOCR -> markdown and writes an output file. The first run downloads model weights.

- Run: `uv run python scripts/smoke-real-ocr.py path/to/file.pdf`
- Output: `path/to/file.md` (use `--out ...` and `--overwrite`)

Environment variables (optional):
- `LIGHTONOCR_MODEL_ID` (default: `lightonai/LightOnOCR-2-1B`)
- `LIGHTONOCR_DEVICE` (`cpu|cuda|mps|auto`, default: `cpu`)
- `LIGHTONOCR_DTYPE` (`float32|bfloat16|float16`, optional)
- `LIGHTONOCR_MAX_NEW_TOKENS` (default: `1024`)
- `LIGHTONOCR_DRY_RUN` (truthy to skip real inference)
- `HF_HOME` (controls the Hugging Face cache location; standalone defaults to `dist/standalone/data/hf` unless overridden)

## Standalone distribution (Windows reference)
Builds a self-contained folder using `python-build-standalone` (no system Python required for end users).

### Prereqs (build machine)
- Windows x86_64
- `uv` on PATH
- `git` on PATH (required for the `transformers` Git dependency)
- `tar` on PATH (Windows 10+ includes it)
- Internet access (downloads python runtime + wheels + model weights unless skipped)

### Build
- `powershell -ExecutionPolicy Bypass -File scripts/build-standalone.ps1 -Clean`

Optional parameters:
- `-PythonVersion 3.11.14`
- `-TargetTriple x86_64-pc-windows-msvc`
- `-PbsRelease latest` (or a tag like `20260114` for repeatability)
- `-PipTempRoot <path>` (defaults to `%LOCALAPPDATA%\\t`; use a very short path if you see path-length failures)
- `-ModelId lightonai/LightOnOCR-2-1B` (prefetch this model into the standalone cache)
- `-SkipModelPrefetch` (do not download model weights during build; they will download on first use at runtime)

Outputs to `dist/standalone/`:
- `python/` (runtime)
- `site-packages/` (deps)
- `app/` (source)
- `data/hf/` (Hugging Face cache; includes model weights when prefetched)
- `run.ps1` / `run.cmd` (launcher)

### Package (zip)
- `powershell -ExecutionPolicy Bypass -File scripts/package-standalone.ps1 -Force`
- Output: `dist/ragprep-standalone-<git-sha>.zip` (includes `BUILD_INFO.txt` at the zip root)

### Third-party notices
- Repo: `THIRD_PARTY_NOTICES.md`
- Standalone output / zip: `THIRD_PARTY_NOTICES.md`

### Run (smoke test)
- `powershell -ExecutionPolicy Bypass -File dist/standalone/run.ps1`
- Visit `http://127.0.0.1:8000` and upload a PDF

### Troubleshooting
- If build fails on the `transformers` line, confirm `git` is installed and on PATH.
- If you see Windows path-length errors (e.g. `Filename too long`), set `-PipTempRoot` to a shorter path.
- If `tar` is missing, install a recent Windows build or provide bsdtar.
- By default, model weights are prefetched during `scripts/build-standalone.ps1` into `dist/standalone/data/hf`. Set `HF_HOME` to override the cache location, or pass `-SkipModelPrefetch` to download on first use instead.

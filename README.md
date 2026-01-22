# RAGPrep

PDF → Markdown app scaffold (FastAPI + htmx) backed by `lightonai/LightOnOCR-2-1B`.

## Dev setup
- Install `uv`: https://docs.astral.sh/uv/
- Install deps: `uv sync --dev`
- Run server: `uv run uvicorn ragprep.web.app:app --reload`
- Open: `http://127.0.0.1:8000`

## Standalone distribution (Windows reference)
Builds a self-contained folder using `python-build-standalone` (no system Python required for end users).

### Prereqs (build machine)
- Windows x86_64
- `uv` on PATH
- `git` on PATH (required for the `transformers` Git dependency)
- `tar` on PATH (Windows 10+ includes it)
- Internet access (downloads python runtime + wheels)

### Build
- `pwsh -File scripts/build-standalone.ps1 -Clean`

Optional parameters:
- `-PythonVersion 3.11.14`
- `-TargetTriple x86_64-pc-windows-msvc`
- `-PbsRelease latest` (or a tag like `20260114` for repeatability)
- `-PipTempRoot <path>` (defaults to `%LOCALAPPDATA%\\t`; use a very short path if you see path-length failures)

Outputs to `dist/standalone/`:
- `python/` (runtime)
- `site-packages/` (deps)
- `app/` (source)
- `run.ps1` / `run.cmd` (launcher)

### Run (smoke test)
- `pwsh -File dist/standalone/run.ps1`
- Visit `http://127.0.0.1:8000` and upload a PDF

### Troubleshooting
- If build fails on the `transformers` line, confirm `git` is installed and on PATH.
- If you see Windows path-length errors (e.g. `Filename too long`), set `-PipTempRoot` to a shorter path.
- If `tar` is missing, install a recent Windows build or provide bsdtar.
- Model weights download on first use; set `HF_HOME` to control the cache location.

# RAGPrep

PDF をページ単位で OCR し、Markdown に変換するツールです（Web UI / Desktop UI / CLI）。

## セットアップ
```bash
cd C:\Users\Administrator\RAGPrep
uv sync --dev
```

## 実行

### Desktop（推奨）
```bash
uv run python -m ragprep.desktop
```

バインド先を変える場合:
```bash
uv run python -m ragprep.desktop --host 127.0.0.1 --port 8000
```

### Web
```bash
uv run uvicorn ragprep.web.app:app --reload
```
ブラウザで `http://127.0.0.1:8000` を開きます。

### CLI（PDF → Markdown）
```bash
uv run python scripts/pdf_to_markdown.py --pdf .\path\to\input.pdf --out .\out\input.md --overwrite
```

標準出力へ:
```bash
uv run python scripts/pdf_to_markdown.py --pdf .\path\to\input.pdf --stdout
```

## OCR バックエンド（GLM-OCR）
RAGPrep は `zai-org/GLM-OCR` を **OpenAI 互換 API**（`/v1/chat/completions`）経由で呼び出します。

RAGPrep 側のデフォルト:
- `RAGPREP_GLM_OCR_BASE_URL=http://127.0.0.1:8080`
- `RAGPREP_GLM_OCR_MODEL=zai-org/GLM-OCR`

疎通確認（例）:
```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/v1/models
```

### vLLM（例）
```bash
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
pip install git+https://github.com/huggingface/transformers.git
vllm serve zai-org/GLM-OCR --allowed-local-media-path / --port 8080
```

### SGLang（例）
```bash
pip install git+https://github.com/sgl-project/sglang.git#subdirectory=python
pip install git+https://github.com/huggingface/transformers.git
python -m sglang.launch_server --model zai-org/GLM-OCR --port 8080
```

## 環境変数

### 共通
- `RAGPREP_MAX_UPLOAD_BYTES`（デフォルト: 10MB）
- `RAGPREP_MAX_PAGES`（デフォルト: 50）
- `RAGPREP_MAX_CONCURRENCY`（デフォルト: 1）
- `RAGPREP_RENDER_DPI`（デフォルト: 400）
- `RAGPREP_RENDER_MAX_EDGE`（デフォルト: 1540）

### GLM-OCR
- `RAGPREP_GLM_OCR_BASE_URL`（デフォルト: `http://127.0.0.1:8080`）
- `RAGPREP_GLM_OCR_MODEL`（デフォルト: `zai-org/GLM-OCR`）
- `RAGPREP_GLM_OCR_API_KEY`（任意: `Authorization: Bearer ...`）
- `RAGPREP_GLM_OCR_MAX_TOKENS`（デフォルト: 8192）
- `RAGPREP_GLM_OCR_TIMEOUT_SECONDS`（デフォルト: 60）

## スタンドアロン配布（Windows）
`scripts/build-standalone.ps1` は、Python runtime + `site-packages` + `app` を `dist/standalone/` にまとめます。

注意:
- OCR は外部の GLM-OCR サーバに依存します（スタンドアロンにモデルは同梱しません）。
- `dist/standalone/run.ps1` / `run.cmd` は起動時に `RAGPREP_GLM_OCR_BASE_URL/v1/models` を疎通確認し、到達できない場合はエラーで終了します。

### ビルド
```powershell
cd C:\Users\Administrator\RAGPrep
.\scripts\build-standalone.ps1 -Clean
```

### 起動
```powershell
.\dist\standalone\run.ps1
```

別ポートの GLM-OCR サーバを使う場合（例）:
```powershell
$env:RAGPREP_GLM_OCR_BASE_URL = "http://127.0.0.1:18080"
.\dist\standalone\run.ps1
```

## 開発用コマンド
```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

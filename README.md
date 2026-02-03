# RAGPrep

PDF（および画像）をページ単位で OCR し、Markdown として取り出すツールです。

- 実装: FastAPI（Web UI） + pywebview（Desktop UI）
- 出力: Markdown（`document.md`）

## バックエンド（OCR）
RAGPrep は PDF を画像にレンダリングし、各ページを OCR バックエンドに渡します。

- `glm-ocr`（デフォルト）: `zai-org/GLM-OCR` を OpenAI 互換 API（`/v1/chat/completions`）経由で呼び出す
- `lightonocr`（フォールバック）: llama.cpp CLI + GGUF（スタンドアロン配布に同梱可能）

バックエンドは環境変数で切り替えます（後述）。

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

オプション:
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

## GLM-OCR（デフォルト / OpenAI互換API）
`RAGPREP_PDF_BACKEND=glm-ocr`（デフォルト）は、ローカルで起動した GLM-OCR サーバ（vLLM / SGLang など）へ
OpenAI 互換の `chat.completions` API を呼び出して OCR を行います。

### サーバ起動（vLLM 例）
```bash
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
pip install git+https://github.com/huggingface/transformers.git
vllm serve zai-org/GLM-OCR --allowed-local-media-path / --port 8080
```

### サーバ起動（SGLang 例）
```bash
pip install git+https://github.com/sgl-project/sglang.git#subdirectory=python
pip install git+https://github.com/huggingface/transformers.git
python -m sglang.launch_server --model zai-org/GLM-OCR --port 8080
```

### RAGPrep 側の設定（例）
```powershell
$env:RAGPREP_PDF_BACKEND = "glm-ocr"
$env:RAGPREP_GLM_OCR_BASE_URL = "http://127.0.0.1:8080"
$env:RAGPREP_GLM_OCR_MODEL = "zai-org/GLM-OCR"
```

## LightOnOCR（フォールバック / GGUF）
`RAGPREP_PDF_BACKEND=lightonocr` は、llama.cpp CLI（`llama-mtmd-cli` / `llava-cli` 系）+ GGUF を使うローカル OCR です。

### 推奨パラメータ（llama.cpp CLI）
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--temp` | 0.2 | recommended temperature |
| `--top-p` | 0.9 | sampling top_p |
| `--repeat-penalty` | 1.15 | prevents repetition |
| `--repeat-last-n` | 128 | tokens to consider for penalty |
| `-n` | 1000 | max output tokens |
| `-ngl` | 99 | GPU layers |

### 必要な環境変数（LightOnOCR）
- `LIGHTONOCR_GGUF_MODEL_PATH`: `.gguf` モデルへのパス
- `LIGHTONOCR_GGUF_MMPROJ_PATH`: `.gguf` mmproj へのパス
- `LIGHTONOCR_LLAVA_CLI_PATH`: 任意（CLI のパス。未指定なら PATH を探索）

## 主要な環境変数
### 共通
- `RAGPREP_PDF_BACKEND`: `glm-ocr`（デフォルト） / `lightonocr`
- `RAGPREP_MAX_UPLOAD_BYTES`: デフォルト 10MB
- `RAGPREP_MAX_PAGES`: デフォルト 50
- `RAGPREP_MAX_CONCURRENCY`: デフォルト 1
- `RAGPREP_RENDER_DPI`: デフォルト 400
- `RAGPREP_RENDER_MAX_EDGE`: デフォルト 1540

### GLM-OCR
- `RAGPREP_GLM_OCR_BASE_URL`: デフォルト `http://127.0.0.1:8080`
- `RAGPREP_GLM_OCR_MODEL`: デフォルト `zai-org/GLM-OCR`
- `RAGPREP_GLM_OCR_API_KEY`: 任意（`Authorization: Bearer ...`）
- `RAGPREP_GLM_OCR_MAX_TOKENS`: デフォルト 8192
- `RAGPREP_GLM_OCR_TIMEOUT_SECONDS`: デフォルト 60

### LightOnOCR
- `LIGHTONOCR_GGUF_MODEL_PATH` / `LIGHTONOCR_GGUF_MMPROJ_PATH`
- `LIGHTONOCR_LLAVA_CLI_PATH`

## スタンドアロン配布（Windows）
スタンドアロンは llama.cpp（Vulkan/AVX2）と LightOnOCR GGUF を同梱できます。
GLM-OCR を使う場合は別途サーバ起動が必要です（スタンドアロンにモデルは同梱しません）。

スタンドアロンの `run.ps1` / `run.cmd` は **デフォルトで `lightonocr`** を選びます（オフラインで動作させるため）。
GLM-OCR を使う場合は、GLM-OCR サーバを起動した上で `RAGPREP_PDF_BACKEND=glm-ocr` を設定してください。

### 前提
- Windows + PowerShell
- `uv`（依存: `scripts/build-standalone.ps1`）
- `tar`（依存: `scripts/build-standalone.ps1`）
- `7z`（依存: `scripts/package-standalone.ps1`）

### ビルド
```powershell
cd C:\Users\Administrator\RAGPrep
.\scripts\build-standalone.ps1 -Clean
```

### 起動
```powershell
.\dist\standalone\run.ps1
```

LightOnOCR を使う場合（例）:
```powershell
$env:RAGPREP_PDF_BACKEND = "lightonocr"
.\dist\standalone\run.ps1
```

GLM-OCR を使う場合（例）:
```powershell
$env:RAGPREP_PDF_BACKEND = "glm-ocr"
$env:RAGPREP_GLM_OCR_BASE_URL = "http://127.0.0.1:8080"
.\dist\standalone\run.ps1
```

## 開発用コマンド
```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

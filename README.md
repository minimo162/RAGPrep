# RAGPrep

PDF を OCR して Markdown に変換するツール（Web / Desktop / CLI）。

OCR は GLM-OCR（`zai-org/GLM-OCR`）を使います。

## 最短で動かす（推奨: Transformers / ローカル実行）

### 1) セットアップ
```bash
cd C:\Users\Administrator\RAGPrep
uv sync --dev
```

### 2) Transformers 依存を入れる
```bash
pip install git+https://github.com/huggingface/transformers.git
pip install torch
```

もし Transformers のモデル読み込みでエラーになる場合は、`RAGPREP_GLM_OCR_MODE=server` に切り替えるのが確実です。

### 3) 起動
Desktop:
```bash
uv run python -m ragprep.desktop
```

Web:
```bash
uv run uvicorn ragprep.web.app:app --reload
```
`http://127.0.0.1:8000` を開きます。

CLI（PDF → Markdown）:
```bash
uv run python scripts/pdf_to_markdown.py --pdf .\path\to\input.pdf --out .\out\input.md --overwrite
```

## OCR モード（環境変数）

デフォルトはローカル実行です:
- `RAGPREP_GLM_OCR_MODE=transformers`（default）
- `RAGPREP_GLM_OCR_MODEL=zai-org/GLM-OCR`（default）

OpenAI 互換サーバーに接続したい場合:
- `RAGPREP_GLM_OCR_MODE=server`
- `RAGPREP_GLM_OCR_BASE_URL=http://127.0.0.1:8080`（default）
- `RAGPREP_GLM_OCR_API_KEY`（optional）

### server モード: サーバー起動（Windows 推奨: Docker）
```bash
docker run --rm -it -p 8080:8080 vllm/vllm-openai:nightly vllm serve zai-org/GLM-OCR --host 0.0.0.0 --port 8080
```

疎通確認（200 なら OK）:
```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/v1/models
```

## Standalone（Windows）

### ビルド
```powershell
cd C:\Users\Administrator\RAGPrep
.\scripts\build-standalone.ps1 -Clean
```

### 起動
Transformers（デフォルト）:
```powershell
.\dist\standalone\run.ps1
```

server モード（サーバーが必要）:
```powershell
$env:RAGPREP_GLM_OCR_MODE = "server"
.\dist\standalone\start-glm-ocr.ps1
.\dist\standalone\run.ps1
```

## 開発者向け（品質ゲート）
```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```


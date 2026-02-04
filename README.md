# RAGPrep

PDF のテキストレイヤー（PyMuPDF）を使って、PDF を構造化された HTML に変換するツール（Web / Desktop / CLI）。

レイアウト解析は GLM-OCR（`zai-org/GLM-OCR`）のサーバーモード（OpenAI 互換）で利用できます。
サーバーがない場合も、フォールバック（ページ全体を text として扱う）で HTML を出力します。

## 最短で動かす

### 1) セットアップ
```bash
cd C:\Users\Administrator\RAGPrep
uv sync --dev
```

### 2) 起動
Desktop:
```bash
uv run python -m ragprep.desktop
```

Web:
```bash
uv run uvicorn ragprep.web.app:app --reload
```
`http://127.0.0.1:8000` を開きます。

CLI（PDF → HTML）:
```bash
uv run python scripts/pdf_to_html.py --pdf .\path\to\input.pdf --out .\out\input.html --overwrite
```

## レイアウト解析（環境変数）

デフォルトは `transformers`（= レイアウト解析はフォールバック）です:
- `RAGPREP_GLM_OCR_MODE=transformers`（default）

GLM-OCR サーバーでレイアウト解析を有効にする場合:
- `RAGPREP_GLM_OCR_MODE=server`
- `RAGPREP_GLM_OCR_BASE_URL=http://127.0.0.1:8080`（default）
- `RAGPREP_GLM_OCR_MODEL=zai-org/GLM-OCR`（default）
- `RAGPREP_GLM_OCR_API_KEY`（optional）

### server モード: サーバー起動（Windows 推奨: Docker）
```bash
docker run --rm -it -p 8080:8080 vllm/vllm-openai:nightly vllm serve zai-org/GLM-OCR --host 0.0.0.0 --port 8080
```

疎通確認（200 なら OK）:
```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/v1/models
```

## （任意）OCR で Markdown を作りたい場合
OCR ベースの Markdown 変換も残しています（レイアウト解析とは別）。
```bash
uv run python scripts/pdf_to_markdown.py --pdf .\path\to\input.pdf --out .\out\input.md --overwrite
```
Transformers で GLM-OCR を動かす場合は `torch` / `transformers` を別途インストールしてください。

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


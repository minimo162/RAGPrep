# RAGPrep

RAGPrep は、財務・IR系ドキュメント向けの PDF-to-HTML 変換ツールです。
現在のパイプラインは LightOn OCR（`lighton-ocr`）と PyMuPDF ベースの補正を組み合わせ、構造化された HTML を生成します。

## 動作要件

- Python `>=3.11,<3.13`
- `uv`
- `llama-server`（手動インストールまたは自動ダウンロード）

## クイックスタート（Windows / PowerShell）

```powershell
cd <repo-root>
uv sync --dev
uv run python -m ragprep.desktop
```

Web モード:

```powershell
uv run uvicorn ragprep.web.app:app --reload
```

`http://127.0.0.1:8000` を開きます。

CLI 変換:

```powershell
uv run python scripts/pdf_to_html.py --pdf .\path\to\input.pdf --out .\out\input.html --overwrite
```

CLI オプション:

- `--stdout`: HTML を標準出力に出力
- `--fragment`: HTML フラグメントを出力（`<html>` ラッパーなし）
- `--stdout` と `--out` は同時指定不可

## 実行時の動作

- PDF 各ページを画像化し、LightOn OCR を実行します。
- OCR テキストは PyMuPDF のテキスト情報と表構造情報で補正されます。
- Fast-pass モードは既定で有効です（`RAGPREP_LIGHTON_FAST_PASS=1`）。
- 高解像度リトライと表末尾の二次 OCR 修復は任意で、ベストエフォートです。
- Web/Desktop モードでは同時に 1 ジョブのみ変換します。
- 出力はセマンティックな HTML テーブルタグ（`<table>`, `<thead>`, `<tbody>`）を使用します。

## 設定

基本制限とバックエンド:

- `RAGPREP_MAX_UPLOAD_BYTES=104857600` (100 MB)
- `RAGPREP_MAX_PAGES=200`
- `RAGPREP_MAX_CONCURRENCY=1`
- `RAGPREP_PDF_BACKEND=lighton-ocr`（対応バックエンドはこれのみ）
- `RAGPREP_MODEL_CACHE_DIR=~/.ragprep/model-cache`

LightOn モデル設定:

- `RAGPREP_LIGHTON_REPO_ID=noctrex/LightOnOCR-2-1B-GGUF`
- `RAGPREP_LIGHTON_MODEL_FILE=LightOnOCR-2-1B-IQ4_XS.gguf`
- `RAGPREP_LIGHTON_MMPROJ_FILE=mmproj-BF16.gguf`
- `RAGPREP_LIGHTON_MODEL_DIR=~/.ragprep/models/lighton`
- `RAGPREP_LIGHTON_AUTO_DOWNLOAD=1`

LightOn OCR/サーバー既定値:

- `RAGPREP_LIGHTON_PROFILE=balanced`（対応プロファイルはこれのみ）
- `RAGPREP_LIGHTON_SERVER_HOST=127.0.0.1`
- `RAGPREP_LIGHTON_SERVER_PORT=8080`
- `RAGPREP_LIGHTON_START_TIMEOUT_SECONDS=300`
- `RAGPREP_LIGHTON_REQUEST_TIMEOUT_SECONDS=600`
- `RAGPREP_LIGHTON_CTX_SIZE=8192`
- `RAGPREP_LIGHTON_N_GPU_LAYERS=-1`
- `RAGPREP_LIGHTON_PARALLEL=2`
- `RAGPREP_LIGHTON_THREADS=<cpu_count>`
- `RAGPREP_LIGHTON_THREADS_BATCH=<cpu_count>`
- `RAGPREP_LIGHTON_FLASH_ATTN=1`
- `RAGPREP_LIGHTON_MAX_TOKENS=8192`
- `RAGPREP_LIGHTON_PAGE_CONCURRENCY=2`
- `RAGPREP_LIGHTON_RENDER_DPI=220`
- `RAGPREP_LIGHTON_RENDER_MAX_EDGE=1280`
- `RAGPREP_LIGHTON_MERGE_POLICY=strict` (`strict` or `aggressive`)
- `RAGPREP_LIGHTON_FAST_PASS=1`
- `RAGPREP_LIGHTON_FAST_RENDER_DPI=200`
- `RAGPREP_LIGHTON_FAST_RENDER_MAX_EDGE=1100`
- `RAGPREP_LIGHTON_FAST_RETRY=0`
- `RAGPREP_LIGHTON_SECONDARY_TABLE_REPAIR=0`
- `RAGPREP_LIGHTON_RETRY_RENDER_DPI=220`
- `RAGPREP_LIGHTON_RETRY_RENDER_MAX_EDGE=1280`
- `RAGPREP_LIGHTON_RETRY_MIN_QUALITY=0.40`
- `RAGPREP_LIGHTON_RETRY_QUALITY_GAP=0.22`
- `RAGPREP_LIGHTON_RETRY_MIN_PYM_TEXT_LEN=80`
- `RAGPREP_LIGHTON_FAST_NON_TABLE_MAX_EDGE=960`
- `RAGPREP_LIGHTON_FAST_TABLE_LIKELIHOOD_THRESHOLD=0.60`
- `RAGPREP_LIGHTON_FAST_MAX_TOKENS_TEXT=4096`
- `RAGPREP_LIGHTON_FAST_MAX_TOKENS_TABLE=8192`
- `RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE=light` (`full`, `light`, `off`)
- `RAGPREP_LIGHTON_PYMUPDF_PAGE_FALLBACK_MODE=repeat` (`off`, `repeat`, `aggressive`)

`llama-server` の解決順:

1. `RAGPREP_LLAMA_SERVER_PATH`（明示パス）
2. PATH / 一般的なローカル配置先
3. `ggml-org/llama.cpp` の最新リリースから自動ダウンロード（既定で有効）

関連環境変数:

- `RAGPREP_LLAMA_SERVER_PATH`
- `RAGPREP_LLAMA_SERVER_AUTO_DOWNLOAD=1`
- `RAGPREP_LLAMA_SERVER_INSTALL_DIR`（既定: LightOn モデルディレクトリの兄弟）

Web 起動時 prewarm:

- `RAGPREP_WEB_PREWARM_ON_STARTUP=1`
- `RAGPREP_WEB_PREWARM_START_DELAY_SECONDS=0.35`
- `RAGPREP_WEB_PREWARM_EXECUTOR=thread`（desktop モードでは `process` が自動選択）
- `RAGPREP_WEB_PREWARM_TIMEOUT_SECONDS=120`

## 失敗時の挙動

- 1 次 OCR のページ処理が失敗した場合、ジョブは失敗します。
- 表末尾の二次修復や retry OCR はベストエフォートであり、それ自体ではジョブ失敗にしません。
- LightOn モデルファイルが不足していて `RAGPREP_LIGHTON_AUTO_DOWNLOAD=0` の場合、変換は失敗します。
- `llama-server` を検出できない、または起動できない場合、明確なエラーで失敗します。

## Standalone ビルドスクリプト（Windows）

```powershell
.\scripts\build-standalone.ps1
.\scripts\verify-standalone.ps1
.\scripts\package-standalone.ps1
```

## 開発

```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

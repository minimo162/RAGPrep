# RAGPrep

PyMuPDF Layout（`pymupdf-layout`）+ PyMuPDF4LLM（`pymupdf4llm`）を使って、PDF を **全ページ一括**で Markdown に変換するツールです。

## 前提・制約
- **OCR は廃止**しています（スキャンPDF等の画像文字は変換できません）。
- 入力PDFは **テキストレイヤー付き**である必要があります。

## セットアップ
```bash
cd C:\Users\Administrator\RAGPrep
uv sync --dev
```

## 実行（Web）
```bash
cd C:\Users\Administrator\RAGPrep
uv run uvicorn ragprep.web.app:app --reload
```

ブラウザで `http://127.0.0.1:8000` を開き、PDF をアップロードしてください。

## 実行（CLI: PDF → Markdown）
```bash
cd C:\Users\Administrator\RAGPrep
uv run python scripts/pdf_to_markdown.py --pdf .\path\to\input.pdf --out .\out\input.md --overwrite
```

標準出力へ出す場合:
```bash
uv run python scripts/pdf_to_markdown.py --pdf .\path\to\input.pdf --stdout
```

## ベンチマーク（変換全体時間）
```bash
cd C:\Users\Administrator\RAGPrep
uv run python scripts/bench_pdf_to_markdown.py --synthetic-pages 3
uv run python scripts/bench_pdf_to_markdown.py --pdf .\path\to\input.pdf --repeat 3
```

## 主要な環境変数
- `RAGPREP_MAX_UPLOAD_BYTES`（デフォルト: 10MB）
- `RAGPREP_MAX_PAGES`（デフォルト: 50）
- `RAGPREP_MAX_CONCURRENCY`（デフォルト: 1）

## 開発（品質ゲート）
```bash
cd C:\Users\Administrator\RAGPrep
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

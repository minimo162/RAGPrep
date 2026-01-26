# RAGPrep

PyMuPDF Layout（`pymupdf-layout`）+ PyMuPDF4LLM（`pymupdf4llm`）を使って、PDF を **全ページ一括**で JSON に変換するツールです。
JSON 出力ではページの header/footer を除外します。

## 前提・制約
- **OCR は廃止**しています（スキャンPDF等の画像文字は変換できません）。
- 入力PDFは **テキストレイヤー付き**である必要があります。

## PDFテキストの読み順（仕様）
RAGPrep は「できるだけ自然な読み順」を目標に、ページ内レイアウトを簡易推定して抽出順を調整します。

### 原則
- **単一カラム**: 上→下（行単位、同一行は左→右）
- **右サイドバー/コールアウト**: 本文の近傍（y位置）に挿入
- **2カラム以上（主要カラム）**: column-major（左カラムを上→下、次に右カラム…）

### 既知の制約
- テキストボックスが **カラム境界を跨ぐ** / **重なりが強い自由配置** は、完全な読み順再現が定義できない場合があります。
  - 重複や欠落を避けるため、該当ページは whole-page sort にフォールバックすることがあります（column-major が崩れる場合あり）。
- PDF内部のテキストブロック分割のされ方（フォント/描画方法）により、行結合や順序が変わることがあります。

## セットアップ
```bash
cd C:\Users\Administrator\RAGPrep
uv sync --dev
```

## 実行（GUI）
```bash
cd C:\Users\Administrator\RAGPrep
uv run python -m ragprep.desktop
```

起動すると GUI が自動で開きます。GUI を閉じるとアプリも終了します。

変換が完了したら、結果画面の `Download .json` をクリックして JSON を保存します。
- **ファイル名**: 元のPDFファイル名の拡張子を `.json` に変更（例: `foo.pdf` → `foo.json`）
- **GUI**: 保存ダイアログが開き、任意の場所に保存
- **Web**: ブラウザのダウンロードとして保存

### トラブルシュート（GUI）
- GUI が開かない場合: Microsoft Edge WebView2 Runtime が必要です（未導入の場合はインストールしてください）。
- 起動直後に終了する場合: ポート競合の可能性があります。`--port 8001` などでポートを変更してください。
- 詳細ログ: PowerShell/CMD から起動するとエラーメッセージが表示されます。

## 実行（Web）
```bash
cd C:\Users\Administrator\RAGPrep
uv run uvicorn ragprep.web.app:app --reload
```

ブラウザで `http://127.0.0.1:8000` を開き、PDF をアップロードしてください。

## 実行（CLI: PDF → JSON）
```bash
cd C:\Users\Administrator\RAGPrep
uv run python scripts/pdf_to_json.py --pdf .\path\to\input.pdf --out .\out\input.json --overwrite
```

標準出力へ出す場合:
```bash
uv run python scripts/pdf_to_json.py --pdf .\path\to\input.pdf --stdout
```

## Legacy（Markdown 出力）
Markdown が必要な場合は、既存の CLI を利用してください。
```bash
cd C:\Users\Administrator\RAGPrep
uv run python scripts/pdf_to_markdown.py --pdf .\path\to\input.pdf --out .\out\input.md --overwrite
```

## ベンチマーク（変換全体時間 / Legacy: Markdown）
```bash
cd C:\Users\Administrator\RAGPrep
uv run python scripts/bench_pdf_to_markdown.py --synthetic-pages 3
uv run python scripts/bench_pdf_to_markdown.py --pdf .\path\to\input.pdf --repeat 3
```

## 主要な環境変数
- `RAGPREP_MAX_UPLOAD_BYTES`（デフォルト: 10MB）
- `RAGPREP_MAX_PAGES`（デフォルト: 50）
- `RAGPREP_MAX_CONCURRENCY`（デフォルト: 1）

## スタンドアロン配布（Windows）

### 前提
- Windows + PowerShell
- `uv`（依存: `scripts/build-standalone.ps1`）
- `tar`（依存: `scripts/build-standalone.ps1`）
- `7z`（依存: `scripts/package-standalone.ps1`）
- ネットワーク接続（Python runtime / 依存 / llama.cpp 等の取得に使用）

### ビルド
```powershell
cd C:\Users\Administrator\RAGPrep
.\scripts\build-standalone.ps1 -Clean
```

GGUF prefetch をスキップする場合:
```powershell
.\scripts\build-standalone.ps1 -SkipGgufPrefetch -Clean
```

### パッケージ（zip）
```powershell
.\scripts\package-standalone.ps1 -Force
```

`7z.exe` が PATH に無い場合:
```powershell
.\scripts\package-standalone.ps1 -SevenZipPath "C:\Program Files\7-Zip\7z.exe" -Force
```

### 起動
PowerShell:
```powershell
.\dist\standalone\run.ps1
.\dist\standalone\run.ps1 -BindHost 0.0.0.0 -Port 8001
```

起動すると GUI が自動で開きます。GUI を閉じるとアプリも終了します。

CMD:
```bat
dist\standalone\run.cmd
dist\standalone\run.cmd 0.0.0.0 8001
```

`run.cmd` は環境変数 `RAGPREP_BIND_HOST` / `RAGPREP_PORT` でも指定できます。

### よくある失敗（standalone）
- `uv` / `tar` / `7z` が見つからない: インストールして PATH を通す（または `-SevenZipPath` を指定）
- `GGUF prefetch failed`: `-SkipGgufPrefetch` を付けて再実行
- `AccessDenied` / ファイルロック: Explorer で `dist/standalone` を開いている場合は閉じる、ウイルス対策/インデクサのロックが疑わしい場合は少し待って再実行（`-Clean` 推奨）

## 開発（品質ゲート）
```bash
cd C:\Users\Administrator\RAGPrep
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

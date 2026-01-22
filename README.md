# RAGPrep

`lightonai/LightOnOCR-2-1B` を使った PDF -> Markdown 変換アプリ（FastAPI + htmx）のスキャフォールドです。

## 開発環境
- `uv` のインストール: https://docs.astral.sh/uv/
- 依存関係のインストール: `uv sync --dev`
- サーバ起動: `uv run uvicorn ragprep.web.app:app --reload`
- ブラウザで開く: `http://127.0.0.1:8000`

## 実機OCR検証（手動）
PDF -> 画像 -> LightOnOCR -> Markdown を実行し、出力ファイルを書き込みます。初回実行時はモデル重みをダウンロードします。

- 実行: `uv run python scripts/smoke-real-ocr.py path/to/file.pdf`
- 出力: `path/to/file.md`（`--out ...` と `--overwrite` を使用）

環境変数（任意）:
- `LIGHTONOCR_MODEL_ID`（デフォルト: `lightonai/LightOnOCR-2-1B`）
- `LIGHTONOCR_DEVICE`（`cpu|cuda|mps|auto`, デフォルト: `cpu`）
- `LIGHTONOCR_DTYPE`（`float32|bfloat16|float16`, 任意）
- `LIGHTONOCR_MAX_NEW_TOKENS`（デフォルト: `1024`）
- `LIGHTONOCR_DRY_RUN`（truthy で実推論をスキップ）
- `HF_HOME`（Hugging Face のキャッシュ場所。スタンドアロンは未指定時に `dist/standalone/data/hf` を使用）

## スタンドアロン配布（Windows向け）
`python-build-standalone` を使って自己完結したフォルダを作成します（エンドユーザはシステムPython不要）。

### 前提（ビルドマシン）
- Windows x86_64
- `uv` が PATH 上にあること
- `git` が PATH 上にあること（`transformers` の Git 依存のため）
- `tar` が PATH 上にあること（Windows 10+ で同梱）
- `7z` が PATH 上にあること（7-Zip; パッケージ作成に必要）
- インターネット接続（Python runtime + wheels をダウンロード。`-SkipModelPrefetch` を指定しない場合はモデル重みもダウンロード）

### ビルド
- `powershell -ExecutionPolicy Bypass -File scripts/build-standalone.ps1 -Clean`

オプションパラメータ:
- `-PythonVersion 3.11.14`
- `-TargetTriple x86_64-pc-windows-msvc`
- `-PbsRelease latest`（再現性のために `20260114` のような tag も指定可能）
- `-PipTempRoot <path>`（デフォルト: `%LOCALAPPDATA%\\t`; パス長エラーが出る場合は短いパスを指定）
- `-ModelId lightonai/LightOnOCR-2-1B`（このモデルをスタンドアロンのキャッシュへプリフェッチ）
- `-SkipModelPrefetch`（ビルド時にモデル重みをダウンロードしない。実行時の初回利用でダウンロード）

`dist/standalone/` への出力:
- `python/`（ランタイム）
- `site-packages/`（依存関係）
- `app/`（ソース）
- `data/hf/`（Hugging Face キャッシュ。プリフェッチ時はモデル重みを含む）
- `run.ps1` / `run.cmd`（ランチャ）

### パッケージ（zip）
- 7-Zip のインストール: https://www.7-zip.org/
- 実行: `powershell -ExecutionPolicy Bypass -File scripts/package-standalone.ps1 -Force`
- `7z` が PATH 上にない場合: `powershell -ExecutionPolicy Bypass -File scripts/package-standalone.ps1 -SevenZipPath "C:\\Program Files\\7-Zip\\7z.exe" -Force`
- 出力: `dist/ragprep-standalone-<git-sha>.zip`（zip ルートに `BUILD_INFO.txt` を含む）

### サードパーティ通知
- リポジトリ: `THIRD_PARTY_NOTICES.md`
- スタンドアロン出力 / zip: `THIRD_PARTY_NOTICES.md`

### 実行（スモークテスト）
- `powershell -ExecutionPolicy Bypass -File dist/standalone/run.ps1`
- `http://127.0.0.1:8000` にアクセスして PDF をアップロード

### トラブルシューティング
- `transformers` の行でビルドが失敗する場合は、`git` がインストール済みで PATH にあることを確認してください。
- Windows のパス長エラー（例: `Filename too long`）が出る場合は、`-PipTempRoot` に短いパスを指定してください。
- `tar` が見つからない場合は、最近の Windows を利用するか bsdtar を用意してください。
- デフォルトでは `scripts/build-standalone.ps1` 実行時にモデル重みを `dist/standalone/data/hf` にプリフェッチします。キャッシュ場所を変える場合は `HF_HOME` を設定するか、`-SkipModelPrefetch` を指定して初回利用時のダウンロードに切り替えてください。

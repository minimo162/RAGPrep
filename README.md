# RAGPrep

## Standalone llama.cpp CLI
- Standalone bundles llama.cpp binaries under `dist/standalone/bin/llama.cpp/` (includes `llava-cli.exe`).
- `dist/standalone/run.ps1` / `run.cmd` set `LIGHTONOCR_LLAVA_CLI_PATH` automatically unless already set.

LightOnOCR-2-1B（GGUF + llama.cpp）を使った PDF -> Markdown 変換アプリ（FastAPI + htmx）のスキャフォールドです。

## 開発環境
- `uv` のインストール: https://docs.astral.sh/uv/
- 依存関係のインストール: `uv sync --dev`
- サーバ起動: `uv run uvicorn ragprep.web.app:app --reload`
- ブラウザで開く: `http://127.0.0.1:8000`

## 実機OCR検証（手動）
PDF -> 画像 -> LightOnOCR（GGUF + llama.cpp）-> Markdown を実行し、出力ファイルを書き込みます。
事前に GGUF ファイルをローカルに用意し、`LIGHTONOCR_GGUF_MODEL_PATH` / `LIGHTONOCR_GGUF_MMPROJ_PATH` を設定してください。

- 実行: `uv run python scripts/smoke-real-ocr.py path/to/file.pdf`
- 出力: `path/to/file.md`（`--out ...` と `--overwrite` を使用）

環境変数:
- `LIGHTONOCR_MAX_NEW_TOKENS`（デフォルト: `1024`）
- `LIGHTONOCR_DRY_RUN`（truthy で実推論をスキップ）
- `LIGHTONOCR_GGUF_MODEL_PATH`（必須: `LightOnOCR-2-1B-Q4_K_M.gguf`）
- `LIGHTONOCR_GGUF_MMPROJ_PATH`（必須: `LightOnOCR-2-1B-mmproj-f16.gguf`）
- `LIGHTONOCR_LLAMA_N_CTX`（任意: int）
- `LIGHTONOCR_LLAMA_N_THREADS`（任意: int）
- `LIGHTONOCR_LLAMA_N_GPU_LAYERS`（任意: int）
- `HF_HOME`（Hugging Face のキャッシュ場所。スタンドアロンは未指定時に `dist/standalone/data/hf` を使用）

### GGUF + llama.cpp（既定）
GGUF 量子化済モデルを llama.cpp で実行します。

必須モデル（Hugging Face）:
- `wangjazz/LightOnOCR-2-1B-gguf/LightOnOCR-2-1B-Q4_K_M.gguf`
- `wangjazz/LightOnOCR-2-1B-gguf/LightOnOCR-2-1B-mmproj-f16.gguf`

セットアップ（開発環境例）:
1) 依存関係のインストール: `uv sync --dev`
2) 環境変数を設定して実行:
   - `LIGHTONOCR_GGUF_MODEL_PATH=<...\\LightOnOCR-2-1B-Q4_K_M.gguf>`
   - `LIGHTONOCR_GGUF_MMPROJ_PATH=<...\\LightOnOCR-2-1B-mmproj-f16.gguf>`

スタンドアロンで `-SkipGgufPrefetch` を指定しない場合は、GGUF は `dist/standalone/data/models/lightonocr-gguf/` にステージされ、`dist/standalone/run.ps1` / `run.cmd` がデフォルトで `LIGHTONOCR_GGUF_MODEL_PATH` / `LIGHTONOCR_GGUF_MMPROJ_PATH` を設定します（未設定の場合）。

## スタンドアロン配布（Windows向け）
`python-build-standalone` を使って自己完結したフォルダを作成します（エンドユーザはシステムPython不要）。

### 前提（ビルドマシン）
- Windows x86_64
- `uv` が PATH 上にあること
- `tar` が PATH 上にあること（Windows 10+ で同梱）
- `7z` が PATH 上にあること（7-Zip; パッケージ作成に必要）
- インターネット接続（Python runtime + wheels をダウンロード。`-SkipGgufPrefetch` を指定しない場合は GGUF + mmproj もダウンロード）

### ビルド
- `powershell -ExecutionPolicy Bypass -File scripts/build-standalone.ps1 -Clean`

オプションパラメータ:
- `-PythonVersion 3.11.14`
- `-TargetTriple x86_64-pc-windows-msvc`
- `-PbsRelease latest`（再現性のために `20260114` のような tag も指定可能）
- `-PipTempRoot <path>`（デフォルト: `%LOCALAPPDATA%\\t`; パス長エラーが出る場合は短いパスを指定）
- `-GgufRepoId wangjazz/LightOnOCR-2-1B-gguf`（GGUF の取得元リポジトリ）
- `-GgufModelFile LightOnOCR-2-1B-Q4_K_M.gguf`（量子化済 LLM）
- `-GgufMmprojFile LightOnOCR-2-1B-mmproj-f16.gguf`（Vision encoder + projector）
- `-SkipGgufPrefetch`（ビルド時に GGUF + mmproj をステージしない）

`dist/standalone/` への出力:
- `python/`（ランタイム）
- `site-packages/`（依存関係）
- `app/`（ソース）
- `data/hf/`（Hugging Face キャッシュ）
- `data/models/lightonocr-gguf/`（GGUF + mmproj をステージしたファイル。`run.ps1` / `run.cmd` が `LIGHTONOCR_GGUF_*` を設定）
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
- `Move-Item` が `dist/standalone/_extract/python` で `アクセスが拒否されました` になる場合は、`-Clean` で再実行し、`dist/standalone` を開いている Explorer/プロセスを閉じてください（ウイルス対策/インデクサの一時ロックなら少し待って再実行。スクリプトは診断ログとリトライ/フォールバックで回避を試みます）。
- `llama-cpp-python` は PyPI に wheel が無い版があり、`nmake` が見つからない等のエラーは「sdist をビルドしようとしている」状態です。Visual Studio Build Tools を導入するか、prebuilt wheels を利用してください（開発環境は `pyproject.toml` の `[tool.uv]` で CPU wheels を参照します）。スタンドアロンビルドで pip が失敗する場合は、`PIP_EXTRA_INDEX_URL=https://abetlen.github.io/llama-cpp-python/whl/cpu` を設定して再実行してください。
- Windows のパス長エラー（例: `Filename too long`）が出る場合は、`-PipTempRoot` に短いパスを指定してください。
- `tar` が見つからない場合は、最近の Windows を利用するか bsdtar を用意してください。
- デフォルトでは `scripts/build-standalone.ps1` 実行時に GGUF + mmproj を `dist/standalone/data/models/lightonocr-gguf` にステージします。不要なら `-SkipGgufPrefetch` を指定してください。

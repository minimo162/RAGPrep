# RAGPrep

## Standalone llama.cpp CLI
- Standalone bundles llama.cpp binaries under `dist/standalone/bin/llama.cpp/` (includes `llava-cli.exe`).
- `dist/standalone/run.ps1` / `run.cmd` set `LIGHTONOCR_LLAVA_CLI_PATH` automatically unless already set.

## Troubleshooting: `llava-cli failed with exit code 1`

Quick self-check (no network):
- `powershell -ExecutionPolicy Bypass -File scripts/selfcheck-llava-cli-runtime.ps1`

Required env vars (real OCR):
- `LIGHTONOCR_GGUF_MODEL_PATH` (model .gguf)
- `LIGHTONOCR_GGUF_MMPROJ_PATH` (mmproj .gguf)

Optional env vars:
- `LIGHTONOCR_BACKEND` (default: `cli`; `cli` = subprocess CLI per page, `python` = in-process llama-cpp-python runtime)
- `LIGHTONOCR_LLAVA_CLI_PATH` (path to a working `llama-mtmd-cli(.exe)`, `llava-cli(.exe)` or `llama-llava-cli(.exe)`)
- `LIGHTONOCR_IMAGE_TMP_DIR` (override temp dir for OCR images; helps with non-ASCII temp paths)
- `LIGHTONOCR_MAX_NEW_TOKENS` (default: `1000`)
- `LIGHTONOCR_LLAMA_N_CTX` (default: `4096`)
- `LIGHTONOCR_LLAMA_N_THREADS` (optional)
- `LIGHTONOCR_LLAMA_N_GPU_LAYERS` (default: `99`)
- `LIGHTONOCR_LLAMA_TEMP` (default: `0.2`)
- `LIGHTONOCR_LLAMA_REPEAT_PENALTY` (default: `1.15`)
- `LIGHTONOCR_LLAMA_REPEAT_LAST_N` (default: `128`)
- `LIGHTONOCR_DRY_RUN=1` (verify end-to-end flow without inference)

Optional app env vars:
- `RAGPREP_WARMUP_ON_START=1` (warm up the OCR runtime on startup; effective with `LIGHTONOCR_BACKEND=python`)

## Troubleshooting: `The filename, directory name, or volume label syntax is incorrect.`

If you see this message right before Uvicorn starts when launching the standalone via `dist/standalone/run.cmd`,
your `run.cmd` is likely outdated.

Fix:
- Rebuild the standalone (`scripts/build-standalone.ps1`) to regenerate `dist/standalone/run.cmd`.
- Or edit `dist/standalone/run.cmd` and replace `if not exist "%HF_HOME%" mkdir "%HF_HOME%"` with
  `if not exist "%ROOT%data\hf" mkdir "%ROOT%data\hf"`.

## Performance benchmark (conversion total)

Examples (PowerShell):
- Synthetic (no file I/O): `uv run python scripts/bench_pdf_to_markdown.py --synthetic-pages 3`
- Local PDF: `uv run python scripts/bench_pdf_to_markdown.py --pdf .\\path\\to\\input.pdf`

Convert a PDF to Markdown:
- `uv run python scripts/pdf_to_markdown.py --pdf .\\path\\to\\input.pdf --out .\\out\\input.md --overwrite`

Speed-first recommended env preset (PowerShell):

```powershell
# Render (quality vs speed tradeoff)
$env:RAGPREP_RENDER_DPI='120'
$env:RAGPREP_RENDER_MAX_EDGE='1280'

# OCR runtime (reduce per-page overhead)
$env:LIGHTONOCR_BACKEND='python'  # fallback: 'cli'
$env:LIGHTONOCR_MAX_NEW_TOKENS='1000'
$env:LIGHTONOCR_LLAMA_N_THREADS=([Environment]::ProcessorCount).ToString()
$env:LIGHTONOCR_LLAMA_N_GPU_LAYERS='99'  # set 0 for CPU-only

# Optional: warm up on startup (effective with LIGHTONOCR_BACKEND=python)
$env:RAGPREP_WARMUP_ON_START='1'
```

Targets (baseline comparison, same machine/inputs):
- Total time: -30%
- Time-to-first-page: -50%

Record baseline results (fill this table):
| date | pdf | pages | pages_ocr | pages_skipped | pages_table | pages_image | backend | dpi | max_edge | render_s | ocr_s | total_s | notes |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|

If the error includes a deprecation warning like:

```
WARNING: The binary 'llava-cli.exe' is deprecated.
Please use 'llama-mtmd-cli' instead.
```

Then you are pointing at a deprecated `llava-cli` shim. Fix options:
- Point `LIGHTONOCR_LLAVA_CLI_PATH` to `llama-mtmd-cli.exe` from your llama.cpp release (recommended for newer llama.cpp).
- Point `LIGHTONOCR_LLAVA_CLI_PATH` to `llama-llava-cli.exe` from your llama.cpp release (this project already searches for it), or install/upgrade llama.cpp so `llava-cli` works.
- For the standalone bundle, ensure the actual llama.cpp CLI + required DLLs are present under `dist/standalone/bin/llama.cpp/` (the self-check prints what it resolved).

granite-docling-258M（GGUF + llama.cpp）を使った PDF -> Markdown 変換アプリ（FastAPI + htmx）のスキャフォールドです。

## 推奨パラメータ（llama-mtmd-cli）
このプロジェクトは `ragprep/ocr/lightonocr.py` から llama.cpp CLI（`llama-mtmd-cli`）を呼び出します。
以下は intent の推奨値で、環境変数未設定時のデフォルトとして適用されます（必要に応じて env で上書き可能）。

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--temp` | 0.2 | Official recommended temperature |
| `--repeat-penalty` | 1.15 | Prevents repetition (1.1-1.2 optimal) |
| `--repeat-last-n` | 128 | Tokens to consider for penalty |
| `-n` | 1000 | Max output tokens (avoid >1500) |
| `-ngl` | 99 | GPU layers (use all for best speed) |

Parameter notes:
- **repeat-penalty**: 1.2 を超えると OCR 品質が落ちる可能性があります
- **-n (max tokens)**: 末尾の繰り返しを避けるため、`~1000` 程度に制限するのがおすすめです

環境変数の対応（抜粋）:
- `LIGHTONOCR_MAX_NEW_TOKENS` -> `-n`
- `LIGHTONOCR_LLAMA_N_CTX` -> `-c`
- `LIGHTONOCR_LLAMA_N_THREADS` -> `-t`
- `LIGHTONOCR_LLAMA_N_GPU_LAYERS` -> `-ngl`
- `LIGHTONOCR_LLAMA_TEMP` -> `--temp`
- `LIGHTONOCR_LLAMA_REPEAT_PENALTY` -> `--repeat-penalty`
- `LIGHTONOCR_LLAMA_REPEAT_LAST_N` -> `--repeat-last-n`

## 画像前処理（PDF -> 画像）
PDF は PyMuPDF でレンダリングした後、最終的に「最長辺 N px」になるようリサイズされます。
PNG への一時変換は行わず、ピクセルバッファから直接 PIL.Image を作ります。
- `RAGPREP_RENDER_DPI`（デフォルト: `200`）
- `RAGPREP_RENDER_MAX_EDGE`（デフォルト: `1540`）

## GUIの進捗表示
Web UI（`/`）は HTMX のポーリング（1秒間隔）で進捗を更新し、以下を表示します。
- Phase: `starting` / `rendering` / `ocr` / `done` / `error`
- Page: `x / y`
- progress bar（ページ総数が確定するまでは indeterminate）

## 開発環境
- `uv` のインストール: https://docs.astral.sh/uv/
- 依存関係のインストール: `uv sync --dev`
- サーバ起動: `uv run uvicorn ragprep.web.app:app --reload`
- ブラウザで開く: `http://127.0.0.1:8000`

## 実機OCR検証（手動）
PDF -> 画像 -> LightOnOCR（GGUF + llama.cpp）-> Markdown を実行し、出力ファイルを書き込みます。
事前に GGUF ファイルをローカルに用意し、`LIGHTONOCR_GGUF_MODEL_PATH` / `LIGHTONOCR_GGUF_MMPROJ_PATH` を設定してください。

- 実行: `uv run python scripts/pdf_to_markdown.py --pdf path/to/file.pdf --out path/to/file.md --overwrite`
- 出力: `path/to/file.md`（`--out ...` と `--overwrite` を使用）

環境変数:
- `LIGHTONOCR_MAX_NEW_TOKENS`（デフォルト: `1000`）
- `LIGHTONOCR_DRY_RUN`（truthy で実推論をスキップ）
- `LIGHTONOCR_GGUF_MODEL_PATH`（必須: `granite-docling-258M-Q4_K_M.gguf`）
- `LIGHTONOCR_GGUF_MMPROJ_PATH`（必須: `mmproj-model-f16.gguf`）
- `LIGHTONOCR_LLAMA_N_CTX`（任意: int、未設定時は `4096`）
- `LIGHTONOCR_LLAMA_N_THREADS`（任意: int）
- `LIGHTONOCR_LLAMA_N_GPU_LAYERS`（任意: int、未設定時は `99`）
- `LIGHTONOCR_LLAMA_TEMP`（任意: float、未設定時は `0.2`）
- `LIGHTONOCR_LLAMA_REPEAT_PENALTY`（任意: float、未設定時は `1.15`）
- `LIGHTONOCR_LLAMA_REPEAT_LAST_N`（任意: int、未設定時は `128`）
- `HF_HOME`（Hugging Face のキャッシュ場所。スタンドアロンは未指定時に `dist/standalone/data/hf` を使用）

### GGUF + llama.cpp（既定）
GGUF 量子化済モデルを llama.cpp で実行します。

必須モデル（Hugging Face）:
- `mrutkows/granite-docling-258M-GGUF/granite-docling-258M-Q4_K_M.gguf`
- `mrutkows/granite-docling-258M-GGUF/mmproj-model-f16.gguf`

セットアップ（開発環境例）:
1) 依存関係のインストール: `uv sync --dev`
2) 環境変数を設定して実行:
   - `LIGHTONOCR_GGUF_MODEL_PATH=<...\\granite-docling-258M-Q4_K_M.gguf>`
   - `LIGHTONOCR_GGUF_MMPROJ_PATH=<...\\mmproj-model-f16.gguf>`

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
- `-GgufRepoId mrutkows/granite-docling-258M-GGUF`（GGUF の取得元リポジトリ）
- `-GgufModelFile granite-docling-258M-Q4_K_M.gguf`（量子化済 LLM）
- `-GgufMmprojFile mmproj-model-f16.gguf`（Vision encoder + projector）
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

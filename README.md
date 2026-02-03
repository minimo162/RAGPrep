# RAGPrep

PyMuPDF Layout（`pymupdf-layout`）+ PyMuPDF4LLM（`pymupdf4llm`）を使って、PDF を **ページ単位**で処理し、部分出力をストリーミング表示できる JSON 変換ツールです。
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

## PDFレンダリング（PyMuPDF）
- PDFのレンダリングには PyMuPDF（`pymupdf`）を使用します。
- 画像前処理は以下を前提とします。
  - PDFは PNG または JPEG にレンダリングする
  - 最長辺が 1540px になるようにリサイズする
  - アスペクト比は維持してテキストの幾何を保つ
  - 1ページ = 1画像で扱う（vLLM 側でのバッチ処理に対応）

## GLM-OCR（デフォルト / OpenAI互換API）
`RAGPREP_PDF_BACKEND=glm-ocr`（デフォルト）は、ローカルで起動した GLM-OCR サーバ（vLLM / SGLang）に対して
OpenAI互換の `chat.completions` API を呼び出して OCR を行います。

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

### RAGPrep 側の設定（例）
- `RAGPREP_PDF_BACKEND=glm-ocr`
- `RAGPREP_GLM_OCR_BASE_URL=http://127.0.0.1:8080`
- `RAGPREP_GLM_OCR_MODEL=zai-org/GLM-OCR`

## LightOnOCR推奨パラメータ（llama.cpp CLI）
LightOnOCR を llama.cpp CLI で使う場合は、以下のパラメータを推奨します。

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--temp` | 0.2 | Official recommended temperature |
| `--top-p` | 0.9 | Sampling top_p |
| `--repeat-penalty` | 1.15 | Prevents repetition (1.1-1.2 optimal) |
| `--repeat-last-n` | 128 | Tokens to consider for penalty |
| `-n` | 1000 | Max output tokens (avoid >1500) |
| `-ngl` | 99 | GPU layers (use all for best speed) |

### Parameter Notes
- **repeat-penalty**: 1.2 を超えると OCR 品質が落ちる場合があります
- **-n (max tokens)**: 長い文書の末尾での繰り返しを防ぐため、~1000 を推奨します
- **Image preprocessing**: PDF を最長辺 1540px の PNG としてレンダリングします

## ページ単位ストリーミング出力
- PDFは1ページずつ処理し、部分出力をストリーミング表示します。
- ストリーミング出力のテキストは、テキスト選択でコピーできます。
- スクロール挙動:
  - 表示の一番下にいる場合は、新しい出力に自動で追従します。
  - 途中までスクロールしている場合は、スクロール位置を維持します（勝手に下へ飛びません）。

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

変換が完了したら、結果画面の `Download .json` または `Download .md` をクリックして保存します。
- **ファイル名**:
  - `.json`: 元のPDFファイル名の拡張子を `.json` に変更（例: `foo.pdf` → `foo.json`）
  - `.md`: 元のPDFファイル名の拡張子を `.md` に変更（例: `foo.pdf` → `foo.md`）
- **Markdown の内容**: ページごとの `markdown` を順に結合し、ページ間は空行で区切ります。
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
- `RAGPREP_PDF_BACKEND`: `glm-ocr`（デフォルト） / `lightonocr`（ローカルGGUF）
- `RAGPREP_GLM_OCR_BASE_URL`: GLM-OCR サーバURL（デフォルト: `http://127.0.0.1:8080`）
- `RAGPREP_GLM_OCR_MODEL`: モデル名（デフォルト: `zai-org/GLM-OCR`）
- `RAGPREP_GLM_OCR_API_KEY`: 任意（OpenAI互換の `Authorization: Bearer ...`）
- `RAGPREP_GLM_OCR_MAX_TOKENS`: デフォルト 8192
- `RAGPREP_GLM_OCR_TIMEOUT_SECONDS`: デフォルト 60
- `LIGHTONOCR_GGUF_MODEL_PATH` / `LIGHTONOCR_GGUF_MMPROJ_PATH`: `RAGPREP_PDF_BACKEND=lightonocr` のときに使用
- `LIGHTONOCR_LLAVA_CLI_PATH`: 任意（llama.cpp CLI のパス）

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

#### GGUF 同梱と検証（重要）
- 既定の `.\scripts\build-standalone.ps1 -Clean` は GGUF を取得し、最後に `scripts/verify-standalone.ps1` で必須ファイルを検証します。
- `-SkipGgufPrefetch` を使うと GGUF が無いままになり、検証は失敗します。配布/実行する場合は次のいずれかが必要です。
  - `dist\standalone\data\models\lightonocr-gguf\` に以下 2 ファイルを配置する  
    - `LightOnOCR-2-1B-Q6_K.gguf`  
    - `mmproj-BF16.gguf`
  - 実行前に環境変数で実在パスを上書きする  
    - `LIGHTONOCR_GGUF_MODEL_PATH`  
    - `LIGHTONOCR_GGUF_MMPROJ_PATH`
- 配布前の確認コマンド（手動検証）:
```powershell
.\scripts\verify-standalone.ps1 -OutputDir dist/standalone
```
- `run.ps1` / `run.cmd` は起動前に GGUF の存在を検証し、欠落時は理由と想定配置を表示して停止します。

#### llama.cpp 同梱（Vulkan + AVX2）
- `scripts/build-standalone.ps1` は llama.cpp の **Vulkan 版**と **AVX2（CPU）版**の両方を取得し、同梱します。
- 同梱構造（抜粋）:
```text
dist\standalone\bin\llama.cpp\
  vulkan\llama-mtmd-cli.exe
  avx2\llama-mtmd-cli.exe
  llama-mtmd-cli.exe  # 互換用（avx2 のコピー）
```
- 実行時の自動選択:
  - まず Vulkan 版を優先して使用します。
  - Vulkan 版の起動に失敗した場合は、AVX2 版に自動でフォールバックします。
- 明示的に固定したい場合は `LIGHTONOCR_LLAVA_CLI_PATH` を指定してください（未指定時は自動選択）。
  - 例（PowerShell）:
```powershell
$env:LIGHTONOCR_LLAVA_CLI_PATH = "dist\\standalone\\bin\\llama.cpp\\avx2\\llama-mtmd-cli.exe"
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

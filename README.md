# RAGPrep

PDF を OCR して Markdown に変換するツールです（Web / Desktop / CLI）。

RAGPrep は **GLM-OCR (`zai-org/GLM-OCR`)** を使って OCR します。

デフォルトは **Transformers（ローカル実行）** です（`RAGPREP_GLM_OCR_MODE=transformers`）。  
必要なら OpenAI 互換サーバー（`/v1/chat/completions`）に接続する **server モード**（`RAGPREP_GLM_OCR_MODE=server`）も使えます。

## セットアップ
```bash
cd C:\Users\Administrator\RAGPrep
uv sync --dev
```

## 使い方

### Desktop
```bash
uv run python -m ragprep.desktop
```
バインド先を変える場合:
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
標準出力:
```bash
uv run python scripts/pdf_to_markdown.py --pdf .\path\to\input.pdf --stdout
```

## GLM-OCR（必須）

### 必要な環境変数（デフォルトあり）
- `RAGPREP_GLM_OCR_MODE`（default: `transformers` / `server`）
- `RAGPREP_GLM_OCR_BASE_URL`（default: `http://127.0.0.1:8080`）
- `RAGPREP_GLM_OCR_MODEL`（default: `zai-org/GLM-OCR`）
- `RAGPREP_GLM_OCR_API_KEY`（optional: `Authorization: Bearer ...` を付与）

### Transformers（デフォルト）: ローカル実行
```bash
pip install git+https://github.com/huggingface/transformers.git
pip install torch
```

（参考）単体実行の最小例:
```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

MODEL_PATH = "zai-org/GLM-OCR"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "test_image.png"},
            {"type": "text", "text": "Text Recognition:"},
        ],
    }
]
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
inputs.pop("token_type_ids", None)
generated_ids = model.generate(**inputs, max_new_tokens=8192)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
print(output_text)
```

### server モード: OpenAI 互換サーバーに接続
server モードにする場合:
```powershell
$env:RAGPREP_GLM_OCR_MODE = "server"
```

疎通確認（200 が返れば OK）:
```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/v1/models
```

### Windows（推奨）: Docker Desktop 経由で起動
`vllm` は **Windows ネイティブの wheels が無い**ため、`pip install vllm` は通常失敗します。  
Windows では Docker Desktop（WSL2 backend）での起動を推奨します。

```bash
docker run --rm -it -p 8080:8080 vllm/vllm-openai:nightly vllm serve zai-org/GLM-OCR --host 0.0.0.0 --port 8080
```

### Linux / WSL2: vLLM で起動
```bash
pip install -U vllm
pip install git+https://github.com/huggingface/transformers.git
vllm serve zai-org/GLM-OCR --host 0.0.0.0 --port 8080
```

## Standalone（Windows）
`scripts/build-standalone.ps1` は python runtime + `site-packages` + `app` を `dist/standalone/` にまとめます。  
Standalone はデフォルトで transformers を使います（重い依存が必要）。server モードを使う場合は外部サーバーが必要です。

### ビルド
```powershell
cd C:\Users\Administrator\RAGPrep
.\scripts\build-standalone.ps1 -Clean
```

### 起動（推奨）
1) （server モードのみ）GLM-OCR サーバー起動（Docker があれば自動で Docker を使います）
```powershell
.\dist\standalone\start-glm-ocr.ps1
```
2) RAGPrep 起動
```powershell
.\dist\standalone\run.ps1
```

別ポートで起動したい場合:
```powershell
.\dist\standalone\start-glm-ocr.ps1 -Port 18080
$env:RAGPREP_GLM_OCR_BASE_URL = "http://127.0.0.1:18080"
.\dist\standalone\run.ps1
```

## 開発者向け（品質ゲート）
```bash
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

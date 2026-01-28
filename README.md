# RAGPrep

PyMuPDF Layout・・pymupdf-layout`・・ PyMuPDF4LLM・・pymupdf4llm`・峨ｒ菴ｿ縺｣縺ｦ縲￣DF 繧・**繝壹・繧ｸ蜊倅ｽ・*縺ｧ蜃ｦ逅・＠縲・Κ蛻・・蜉帙ｒ繧ｹ繝医Μ繝ｼ繝溘Φ繧ｰ陦ｨ遉ｺ縺ｧ縺阪ｋ JSON 螟画鋤繝・・繝ｫ縺ｧ縺吶・
JSON 蜃ｺ蜉帙〒縺ｯ繝壹・繧ｸ縺ｮ header/footer 繧帝勁螟悶＠縺ｾ縺吶・

## 蜑肴署繝ｻ蛻ｶ邏・
- **OCR 縺ｯ蟒・ｭ｢**縺励※縺・∪縺呻ｼ医せ繧ｭ繝｣繝ｳPDF遲峨・逕ｻ蜒乗枚蟄励・螟画鋤縺ｧ縺阪∪縺帙ｓ・峨・
- 蜈･蜉娜DF縺ｯ **繝・く繧ｹ繝医Ξ繧､繝､繝ｼ莉倥″**縺ｧ縺ゅｋ蠢・ｦ√′縺ゅｊ縺ｾ縺吶・

## PDF繝・く繧ｹ繝医・隱ｭ縺ｿ鬆・ｼ井ｻ墓ｧ假ｼ・
RAGPrep 縺ｯ縲後〒縺阪ｋ縺縺題・辟ｶ縺ｪ隱ｭ縺ｿ鬆・阪ｒ逶ｮ讓吶↓縲√・繝ｼ繧ｸ蜀・Ξ繧､繧｢繧ｦ繝医ｒ邁｡譏捺耳螳壹＠縺ｦ謚ｽ蜃ｺ鬆・ｒ隱ｿ謨ｴ縺励∪縺吶・

### 蜴溷援
- **蜊倅ｸ繧ｫ繝ｩ繝**: 荳岩・荳具ｼ郁｡悟腰菴阪∝酔荳陦後・蟾ｦ竊貞承・・
- **蜿ｳ繧ｵ繧､繝峨ヰ繝ｼ/繧ｳ繝ｼ繝ｫ繧｢繧ｦ繝・*: 譛ｬ譁・・霑大ｍ・・菴咲ｽｮ・峨↓謖ｿ蜈･
- **2繧ｫ繝ｩ繝莉･荳奇ｼ井ｸｻ隕√き繝ｩ繝・・*: column-major・亥ｷｦ繧ｫ繝ｩ繝繧剃ｸ岩・荳九∵ｬ｡縺ｫ蜿ｳ繧ｫ繝ｩ繝窶ｦ・・

### 譌｢遏･縺ｮ蛻ｶ邏・
- 繝・く繧ｹ繝医・繝・け繧ｹ縺・**繧ｫ繝ｩ繝蠅・阜繧定ｷｨ縺・* / **驥阪↑繧翫′蠑ｷ縺・・逕ｱ驟咲ｽｮ** 縺ｯ縲∝ｮ悟・縺ｪ隱ｭ縺ｿ鬆・・迴ｾ縺悟ｮ夂ｾｩ縺ｧ縺阪↑縺・ｴ蜷医′縺ゅｊ縺ｾ縺吶・
  - 驥崎､・ｄ谺關ｽ繧帝∩縺代ｋ縺溘ａ縲∬ｩｲ蠖薙・繝ｼ繧ｸ縺ｯ whole-page sort 縺ｫ繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ縺吶ｋ縺薙→縺後≠繧翫∪縺呻ｼ・olumn-major 縺悟ｴｩ繧後ｋ蝣ｴ蜷医≠繧奇ｼ峨・
- PDF蜀・Κ縺ｮ繝・く繧ｹ繝医ヶ繝ｭ繝・け蛻・牡縺ｮ縺輔ｌ譁ｹ・医ヵ繧ｩ繝ｳ繝・謠冗判譁ｹ豕包ｼ峨↓繧医ｊ縲∬｡檎ｵ仙粋繧・・ｺ上′螟峨ｏ繧九％縺ｨ縺後≠繧翫∪縺吶・

## PDF繝ｬ繝ｳ繝繝ｪ繝ｳ繧ｰ・・yMuPDF・・
- PDF縺ｮ繝ｬ繝ｳ繝繝ｪ繝ｳ繧ｰ縺ｫ縺ｯ PyMuPDF・・pymupdf`・峨ｒ菴ｿ逕ｨ縺励∪縺吶・
- 逕ｻ蜒丞燕蜃ｦ逅・・莉･荳九ｒ蜑肴署縺ｨ縺励∪縺吶・
  - PDF縺ｯ PNG 縺ｾ縺溘・ JPEG 縺ｫ繝ｬ繝ｳ繝繝ｪ繝ｳ繧ｰ縺吶ｋ
  - 譛髟ｷ霎ｺ縺・1540px 縺ｫ縺ｪ繧九ｈ縺・↓繝ｪ繧ｵ繧､繧ｺ縺吶ｋ
  - 繧｢繧ｹ繝壹け繝域ｯ斐・邯ｭ謖√＠縺ｦ繝・く繧ｹ繝医・蟷ｾ菴輔ｒ菫昴▽
  - 1繝壹・繧ｸ = 1逕ｻ蜒上〒謇ｱ縺・ｼ・LLM 蛛ｴ縺ｧ縺ｮ繝舌ャ繝∝・逅・↓蟇ｾ蠢懶ｼ・

## LightOnOCR・・lama-server・・
LightOnOCR 縺ｯ OpenAI莠呈鋤API邨檎罰縺ｧ llama-server 縺ｫ騾∽ｿ｡縺励∪縺吶・

### 蠢・医・迺ｰ蠅・､画焚
- `LIGHTONOCR_BACKEND`・亥崋螳・ `llama-server`・・
- `LIGHTONOCR_LLAMA_SERVER_URL`・医ョ繝輔か繝ｫ繝・ `http://127.0.0.1:8080`・・
- `LIGHTONOCR_MODEL`・・/v1/models` 縺ｮ id・・
- `LIGHTONOCR_REQUEST_TIMEOUT_SECONDS`・医ョ繝輔か繝ｫ繝・ `120`・・

### 莉ｻ諢上・隱ｿ謨ｴ・・lama-server・・
- `LIGHTONOCR_TEMPERATURE`・医ョ繝輔か繝ｫ繝・ `0.2`・・
- `LIGHTONOCR_MAX_NEW_TOKENS`・医ョ繝輔か繝ｫ繝・ `1000`・・

### 繝医Λ繝悶Ν繧ｷ繝･繝ｼ繝茨ｼ医ち繧､繝繧｢繧ｦ繝茨ｼ・
- 蛻晏屓縺ｯ繝｢繝・Ν隱ｭ縺ｿ霎ｼ縺ｿ縺ｧ譎る俣縺後°縺九ｋ縺溘ａ縲～LIGHTONOCR_REQUEST_TIMEOUT_SECONDS` 繧貞｢励ｄ縺励※縺上□縺輔＞・・tandalone 縺ｮ run 繧ｹ繧ｯ繝ｪ繝励ヨ譌｢螳壹・ 120 遘抵ｼ峨・
- `llama-server` 縺瑚ｵｷ蜍輔＠縺ｦ縺・ｋ縺九～LIGHTONOCR_LLAMA_SERVER_URL` 縺ｫ繧｢繧ｯ繧ｻ繧ｹ縺ｧ縺阪ｋ縺狗｢ｺ隱阪＠縺ｦ縺上□縺輔＞縲・

## 繝壹・繧ｸ蜊倅ｽ阪せ繝医Μ繝ｼ繝溘Φ繧ｰ蜃ｺ蜉・
- PDF縺ｯ1繝壹・繧ｸ縺壹▽蜃ｦ逅・＠縲・Κ蛻・・蜉帙ｒ繧ｹ繝医Μ繝ｼ繝溘Φ繧ｰ陦ｨ遉ｺ縺励∪縺吶・
- 繧ｹ繝医Μ繝ｼ繝溘Φ繧ｰ蜃ｺ蜉帙・繝・く繧ｹ繝医・縲√ユ繧ｭ繧ｹ繝磯∈謚槭〒繧ｳ繝斐・縺ｧ縺阪∪縺吶・
- 繧ｹ繧ｯ繝ｭ繝ｼ繝ｫ謖吝虚:
  - 陦ｨ遉ｺ縺ｮ荳逡ｪ荳九↓縺・ｋ蝣ｴ蜷医・縲∵眠縺励＞蜃ｺ蜉帙↓閾ｪ蜍輔〒霑ｽ蠕薙＠縺ｾ縺吶・
  - 騾比ｸｭ縺ｾ縺ｧ繧ｹ繧ｯ繝ｭ繝ｼ繝ｫ縺励※縺・ｋ蝣ｴ蜷医・縲√せ繧ｯ繝ｭ繝ｼ繝ｫ菴咲ｽｮ繧堤ｶｭ謖√＠縺ｾ縺呻ｼ亥享謇九↓荳九∈鬟帙・縺ｾ縺帙ｓ・峨・

## 繧ｻ繝・ヨ繧｢繝・・
```bash
cd C:\Users\Administrator\RAGPrep
uv sync --dev
```

## 螳溯｡鯉ｼ・UI・・
```bash
cd C:\Users\Administrator\RAGPrep
uv run python -m ragprep.desktop
```

襍ｷ蜍輔☆繧九→ GUI 縺瑚・蜍輔〒髢九″縺ｾ縺吶・UI 繧帝哩縺倥ｋ縺ｨ繧｢繝励Μ繧らｵゆｺ・＠縺ｾ縺吶・

螟画鋤縺悟ｮ御ｺ・＠縺溘ｉ縲∫ｵ先棡逕ｻ髱｢縺ｮ `Download .json` 縺ｾ縺溘・ `Download .md` 繧偵け繝ｪ繝・け縺励※菫晏ｭ倥＠縺ｾ縺吶・
- **繝輔ぃ繧､繝ｫ蜷・*:
  - `.json`: 蜈・・PDF繝輔ぃ繧､繝ｫ蜷阪・諡｡蠑ｵ蟄舌ｒ `.json` 縺ｫ螟画峩・井ｾ・ `foo.pdf` 竊・`foo.json`・・
  - `.md`: 蜈・・PDF繝輔ぃ繧､繝ｫ蜷阪・諡｡蠑ｵ蟄舌ｒ `.md` 縺ｫ螟画峩・井ｾ・ `foo.pdf` 竊・`foo.md`・・
- **Markdown 縺ｮ蜀・ｮｹ**: 繝壹・繧ｸ縺斐→縺ｮ `markdown` 繧帝・↓邨仙粋縺励√・繝ｼ繧ｸ髢薙・遨ｺ陦後〒蛹ｺ蛻・ｊ縺ｾ縺吶・
- **GUI**: 菫晏ｭ倥ム繧､繧｢繝ｭ繧ｰ縺碁幕縺阪∽ｻｻ諢上・蝣ｴ謇縺ｫ菫晏ｭ・
- **Web**: 繝悶Λ繧ｦ繧ｶ縺ｮ繝繧ｦ繝ｳ繝ｭ繝ｼ繝峨→縺励※菫晏ｭ・

### 繝医Λ繝悶Ν繧ｷ繝･繝ｼ繝茨ｼ・UI・・
- GUI 縺碁幕縺九↑縺・ｴ蜷・ Microsoft Edge WebView2 Runtime 縺悟ｿ・ｦ√〒縺呻ｼ域悴蟆主・縺ｮ蝣ｴ蜷医・繧､繝ｳ繧ｹ繝医・繝ｫ縺励※縺上□縺輔＞・峨・
- 襍ｷ蜍慕峩蠕後↓邨ゆｺ・☆繧句ｴ蜷・ 繝昴・繝育ｫｶ蜷医・蜿ｯ閭ｽ諤ｧ縺後≠繧翫∪縺吶Ａ--port 8001` 縺ｪ縺ｩ縺ｧ繝昴・繝医ｒ螟画峩縺励※縺上□縺輔＞縲・
- 隧ｳ邏ｰ繝ｭ繧ｰ: PowerShell/CMD 縺九ｉ襍ｷ蜍輔☆繧九→繧ｨ繝ｩ繝ｼ繝｡繝・そ繝ｼ繧ｸ縺瑚｡ｨ遉ｺ縺輔ｌ縺ｾ縺吶・

## 螳溯｡鯉ｼ・eb・・
```bash
cd C:\Users\Administrator\RAGPrep
uv run uvicorn ragprep.web.app:app --reload
```

繝悶Λ繧ｦ繧ｶ縺ｧ `http://127.0.0.1:8000` 繧帝幕縺阪￣DF 繧偵い繝・・繝ｭ繝ｼ繝峨＠縺ｦ縺上□縺輔＞縲・

## 螳溯｡鯉ｼ・LI: PDF 竊・JSON・・
```bash
cd C:\Users\Administrator\RAGPrep
uv run python scripts/pdf_to_json.py --pdf .\path\to\input.pdf --out .\out\input.json --overwrite
```

讓呎ｺ門・蜉帙∈蜃ｺ縺吝ｴ蜷・
```bash
uv run python scripts/pdf_to_json.py --pdf .\path\to\input.pdf --stdout
```

## Legacy・・arkdown 蜃ｺ蜉幢ｼ・
Markdown 縺悟ｿ・ｦ√↑蝣ｴ蜷医・縲∵里蟄倥・ CLI 繧貞茜逕ｨ縺励※縺上□縺輔＞縲・
```bash
cd C:\Users\Administrator\RAGPrep
uv run python scripts/pdf_to_markdown.py --pdf .\path\to\input.pdf --out .\out\input.md --overwrite
```

## 繝吶Φ繝√・繝ｼ繧ｯ・亥､画鋤蜈ｨ菴捺凾髢・/ Legacy: Markdown・・
```bash
cd C:\Users\Administrator\RAGPrep
uv run python scripts/bench_pdf_to_markdown.py --synthetic-pages 3
uv run python scripts/bench_pdf_to_markdown.py --pdf .\path\to\input.pdf --repeat 3
```

## 荳ｻ隕√↑迺ｰ蠅・､画焚
- `RAGPREP_MAX_UPLOAD_BYTES`・医ョ繝輔か繝ｫ繝・ 10MB・・
- `RAGPREP_MAX_PAGES`・医ョ繝輔か繝ｫ繝・ 50・・
- `RAGPREP_MAX_CONCURRENCY`・医ョ繝輔か繝ｫ繝・ 1・・
- LightOnOCR・・lama-server・・
  - `LIGHTONOCR_BACKEND`・亥崋螳・ `llama-server`・・
  - `LIGHTONOCR_LLAMA_SERVER_URL`・医ョ繝輔か繝ｫ繝・ `http://127.0.0.1:8080`・・
  - `LIGHTONOCR_MODEL`・・/v1/models` 縺ｮ id・・
  - `LIGHTONOCR_REQUEST_TIMEOUT_SECONDS`・医ョ繝輔か繝ｫ繝・ `120`・・
  - `LIGHTONOCR_MAX_NEW_TOKENS`・医ョ繝輔か繝ｫ繝・ `1000`・・
  - `LIGHTONOCR_TEMPERATURE`・医ョ繝輔か繝ｫ繝・ `0.2`・・

## 繧ｹ繧ｿ繝ｳ繝峨い繝ｭ繝ｳ驟榊ｸ・ｼ・indows・・

### 蜑肴署
- Windows + PowerShell
- `uv`・井ｾ晏ｭ・ `scripts/build-standalone.ps1`・・
- `tar`・井ｾ晏ｭ・ `scripts/build-standalone.ps1`・・
- `7z`・井ｾ晏ｭ・ `scripts/package-standalone.ps1`・・
- 繝阪ャ繝医Ρ繝ｼ繧ｯ謗･邯夲ｼ・ython runtime / 萓晏ｭ・/ llama.cpp 遲峨・蜿門ｾ励↓菴ｿ逕ｨ・・

### 繝薙Ν繝・
```powershell
cd C:\Users\Administrator\RAGPrep
.\scripts\build-standalone.ps1 -Clean
```

GGUF prefetch 繧偵せ繧ｭ繝・・縺吶ｋ蝣ｴ蜷・
```powershell
.\scripts\build-standalone.ps1 -SkipGgufPrefetch -Clean
```

#### GGUF 蜷梧｢ｱ縺ｨ讀懆ｨｼ・磯㍾隕・ｼ・
- 譌｢螳壹・ `.\scripts\build-standalone.ps1 -Clean` 縺ｯ GGUF 繧貞叙蠕励＠縲∵怙蠕後↓ `scripts/verify-standalone.ps1` 縺ｧ蠢・医ヵ繧｡繧､繝ｫ繧呈､懆ｨｼ縺励∪縺吶・
- `-SkipGgufPrefetch` 繧剃ｽｿ縺・→ GGUF 縺檎┌縺・∪縺ｾ縺ｫ縺ｪ繧翫∵､懆ｨｼ縺ｯ螟ｱ謨励＠縺ｾ縺吶る・蟶・螳溯｡後☆繧句ｴ蜷医・谺｡縺ｮ縺・★繧後°縺悟ｿ・ｦ√〒縺吶・
  - `dist\standalone\data\models\lightonocr-gguf\` 縺ｫ莉･荳・2 繝輔ぃ繧､繝ｫ繧帝・鄂ｮ縺吶ｋ  
    - `LightOnOCR-2-1B-IQ4_XS.gguf`  
    - `mmproj-F32.gguf`
  - 螳溯｡悟燕縺ｫ迺ｰ蠅・､画焚縺ｧ螳溷惠繝代せ繧剃ｸ頑嶌縺阪☆繧・ 
    - `LIGHTONOCR_GGUF_MODEL_PATH`  
    - `LIGHTONOCR_GGUF_MMPROJ_PATH`
- 驟榊ｸ・燕縺ｮ遒ｺ隱阪さ繝槭Φ繝会ｼ域焔蜍墓､懆ｨｼ・・
```powershell
.\scripts\verify-standalone.ps1 -OutputDir dist/standalone
- 起動待機が足りない場合は `-ServerStartupTimeoutSeconds` を指定（例: `120` / `180`）。
- `LIGHTONOCR_REQUEST_TIMEOUT_SECONDS` を設定している場合、verify はその値を待機時間として使用します。
```
- `run.ps1` / `run.cmd` 縺ｯ襍ｷ蜍募燕縺ｫ GGUF 縺ｮ蟄伜惠繧呈､懆ｨｼ縺励∵ｬ關ｽ譎ゅ・逅・罰縺ｨ諠ｳ螳夐・鄂ｮ繧定｡ｨ遉ｺ縺励※蛛懈ｭ｢縺励∪縺吶・

#### llama.cpp 蜷梧｢ｱ・・lama-server / Vulkan + AVX2・・
- `scripts/build-standalone.ps1` 縺ｯ llama.cpp 縺ｮ **Vulkan 迚・*縺ｨ **AVX2・・PU・臥沿**縺ｮ荳｡譁ｹ繧貞叙蠕励＠縲∝酔譴ｱ縺励∪縺吶・
- 蜷梧｢ｱ讒矩・域栢邊具ｼ・
```text
dist\standalone\bin\llama.cpp\
  vulkan\llama-server.exe
  avx2\llama-server.exe
  llama-server.exe  # 莠呈鋤逕ｨ・・vx2 縺ｮ繧ｳ繝斐・・・
```
- 螳溯｡梧凾縺ｮ閾ｪ蜍暮∈謚・
  - 縺ｾ縺・Vulkan 迚医・ llama-server 繧貞━蜈医＠縺ｦ菴ｿ逕ｨ縺励∪縺吶・
  - Vulkan 迚医・襍ｷ蜍輔↓螟ｱ謨励＠縺溷ｴ蜷医・縲、VX2 迚医↓閾ｪ蜍輔〒繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ縺励∪縺吶・

### 繝代ャ繧ｱ繝ｼ繧ｸ・・ip・・
```powershell
.\scripts\package-standalone.ps1 -Force
```

`7z.exe` 縺・PATH 縺ｫ辟｡縺・ｴ蜷・
```powershell
.\scripts\package-standalone.ps1 -SevenZipPath "C:\Program Files\7-Zip\7z.exe" -Force
```

### 襍ｷ蜍・
PowerShell:
```powershell
.\dist\standalone\run.ps1
.\dist\standalone\run.ps1 -BindHost 0.0.0.0 -Port 8001
```

襍ｷ蜍輔☆繧九→ GUI 縺瑚・蜍輔〒髢九″縺ｾ縺吶・UI 繧帝哩縺倥ｋ縺ｨ繧｢繝励Μ繧らｵゆｺ・＠縺ｾ縺吶・

CMD:
```bat
dist\standalone\run.cmd
dist\standalone\run.cmd 0.0.0.0 8001
```

`run.cmd` 縺ｯ迺ｰ蠅・､画焚 `RAGPREP_BIND_HOST` / `RAGPREP_PORT` 縺ｧ繧よ欠螳壹〒縺阪∪縺吶・

### 繧医￥縺ゅｋ螟ｱ謨暦ｼ・tandalone・・
- `uv` / `tar` / `7z` 縺瑚ｦ九▽縺九ｉ縺ｪ縺・ 繧､繝ｳ繧ｹ繝医・繝ｫ縺励※ PATH 繧帝壹☆・医∪縺溘・ `-SevenZipPath` 繧呈欠螳夲ｼ・
- `GGUF prefetch failed`: `-SkipGgufPrefetch` 繧剃ｻ倥￠縺ｦ蜀榊ｮ溯｡・
- `AccessDenied` / 繝輔ぃ繧､繝ｫ繝ｭ繝・け: Explorer 縺ｧ `dist/standalone` 繧帝幕縺・※縺・ｋ蝣ｴ蜷医・髢峨§繧九√え繧､繝ｫ繧ｹ蟇ｾ遲・繧､繝ｳ繝・け繧ｵ縺ｮ繝ｭ繝・け縺檎桝繧上＠縺・ｴ蜷医・蟆代＠蠕・▲縺ｦ蜀榊ｮ溯｡鯉ｼ・-Clean` 謗ｨ螂ｨ・・

## 髢狗匱・亥刀雉ｪ繧ｲ繝ｼ繝茨ｼ・
```bash
cd C:\Users\Administrator\RAGPrep
uv run ruff check .
uv run mypy ragprep tests
uv run pytest
```

[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$StandaloneDir = "",
    [switch]$SkipStandaloneCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Self-check for GGUF-only runtime prerequisites. No network access required.
#
# Run:
#   powershell -ExecutionPolicy Bypass -File scripts/selfcheck-gguf-runtime.ps1
#
# Optional:
#   powershell -ExecutionPolicy Bypass -File scripts/selfcheck-gguf-runtime.ps1 -StandaloneDir dist/standalone

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        $PythonExe = $venvPython
    }
    else {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd -and -not [string]::IsNullOrWhiteSpace($cmd.Source)) {
            $PythonExe = $cmd.Source
        }
        else {
            throw "Python not found. Run 'uv sync --dev' to create .venv, or install Python and ensure it's on PATH."
        }
    }
}

$PythonExe = [System.IO.Path]::GetFullPath($PythonExe)
Assert-True (Test-Path -LiteralPath $PythonExe) "PythonExe not found: $PythonExe"
Write-Host "Python: $PythonExe" -ForegroundColor Cyan

$envModelPathRaw = $env:LIGHTONOCR_GGUF_MODEL_PATH
$envMmprojPathRaw = $env:LIGHTONOCR_GGUF_MMPROJ_PATH

if (-not [string]::IsNullOrWhiteSpace($envModelPathRaw)) {
    $modelPath = $envModelPathRaw.Trim()
    if (-not [System.IO.Path]::IsPathRooted($modelPath)) {
        $modelPath = Join-Path $repoRoot $modelPath
    }
    $modelPath = [System.IO.Path]::GetFullPath($modelPath)
    Assert-True (Test-Path -LiteralPath $modelPath -PathType Leaf) "GGUF model file not found: $modelPath"
    Write-Host "OK: LIGHTONOCR_GGUF_MODEL_PATH -> $modelPath" -ForegroundColor Green
}
else {
    Write-Host "NOTE: LIGHTONOCR_GGUF_MODEL_PATH is not set (required for runtime unless a launcher sets it)." -ForegroundColor Yellow
}

if (-not [string]::IsNullOrWhiteSpace($envMmprojPathRaw)) {
    $mmprojPath = $envMmprojPathRaw.Trim()
    if (-not [System.IO.Path]::IsPathRooted($mmprojPath)) {
        $mmprojPath = Join-Path $repoRoot $mmprojPath
    }
    $mmprojPath = [System.IO.Path]::GetFullPath($mmprojPath)
    Assert-True (Test-Path -LiteralPath $mmprojPath -PathType Leaf) "GGUF mmproj file not found: $mmprojPath"
    Write-Host "OK: LIGHTONOCR_GGUF_MMPROJ_PATH -> $mmprojPath" -ForegroundColor Green
}
else {
    Write-Host "NOTE: LIGHTONOCR_GGUF_MMPROJ_PATH is not set (required for runtime unless a launcher sets it)." -ForegroundColor Yellow
}

$tempRoot = Join-Path $env:TEMP ("ragprep-gguf-runtime-selfcheck-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
try {
    $pyPath = Join-Path $tempRoot "llama_cpp_import_check.py"
    $py = @'
import importlib
import sys

try:
    import llama_cpp  # noqa: F401
except Exception as exc:
    print("ERROR: failed to import llama_cpp (llama-cpp-python).", file=sys.stderr)
    print("  Fix: run `uv sync --dev` in the repo, or install llama-cpp-python.", file=sys.stderr)
    print("  Note: some versions have no Windows wheels on PyPI.", file=sys.stderr)
    print("        Consider using the upstream CPU wheel index:", file=sys.stderr)
    print("        PIP_EXTRA_INDEX_URL=https://abetlen.github.io/llama-cpp-python/whl/cpu", file=sys.stderr)
    raise

mod = None
for name in ("llama_cpp.llama_chat_format", "llama_cpp.llava"):
    try:
        mod = importlib.import_module(name)
        break
    except Exception:
        continue
if mod is None:
    raise RuntimeError(
        "llama-cpp-python multimodal helpers not found. "
        "Expected llama_cpp.llama_chat_format or llama_cpp.llava."
    )

handler = getattr(mod, "Llava15ChatHandler", None) or getattr(mod, "Llava16ChatHandler", None)
if handler is None:
    raise RuntimeError("Llava15ChatHandler/Llava16ChatHandler not found; multimodal GGUF cannot run.")

print("OK: llama_cpp import and multimodal helpers available.")
'@
    Set-Content -Path $pyPath -Value $py -Encoding UTF8

    $output = & $PythonExe $pyPath 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw ("llama_cpp import self-check failed with exit code ${exitCode}:`n" + ($output | Out-String))
    }
    Write-Host ($output | Out-String).TrimEnd() -ForegroundColor Green
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not $SkipStandaloneCheck) {
    if ([string]::IsNullOrWhiteSpace($StandaloneDir)) {
        $StandaloneDir = Join-Path $repoRoot "dist\\standalone"
    }
    if (-not [System.IO.Path]::IsPathRooted($StandaloneDir)) {
        $StandaloneDir = Join-Path $repoRoot $StandaloneDir
    }
    $StandaloneDir = [System.IO.Path]::GetFullPath($StandaloneDir)

    if (Test-Path -LiteralPath $StandaloneDir -PathType Container) {
        if ([string]::IsNullOrWhiteSpace($envModelPathRaw) -and [string]::IsNullOrWhiteSpace($envMmprojPathRaw)) {
            $stagedModel = Join-Path $StandaloneDir "data\\models\\lightonocr-gguf\\LightOnOCR-2-1B-Q4_K_M.gguf"
            $stagedMmproj = Join-Path $StandaloneDir "data\\models\\lightonocr-gguf\\LightOnOCR-2-1B-mmproj-f16.gguf"
            Assert-True (Test-Path -LiteralPath $stagedModel -PathType Leaf) "Missing staged GGUF model: $stagedModel"
            Assert-True (Test-Path -LiteralPath $stagedMmproj -PathType Leaf) "Missing staged GGUF mmproj: $stagedMmproj"
            Write-Host "OK: Standalone staged GGUF files found under $StandaloneDir" -ForegroundColor Green
        }
        else {
            Write-Host "SKIP: Standalone staged-file check (env vars are set)." -ForegroundColor DarkGray
        }
    }
    else {
        Write-Host "SKIP: Standalone folder not found: $StandaloneDir" -ForegroundColor DarkGray
    }
}

Write-Host "OK: GGUF runtime self-check completed." -ForegroundColor Green

[CmdletBinding()]
param(
    [string]$StandaloneDir = "",
    [switch]$SkipStandaloneFallback
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Self-check for the llava-cli based OCR runtime prerequisites. No network access required.
#
# Run:
#   powershell -ExecutionPolicy Bypass -File scripts/selfcheck-llava-cli-runtime.ps1
#
# Optional:
#   powershell -ExecutionPolicy Bypass -File scripts/selfcheck-llava-cli-runtime.ps1 -StandaloneDir dist/standalone
#
# Notes:
# - Respects the app env vars:
#   - LIGHTONOCR_GGUF_MODEL_PATH
#   - LIGHTONOCR_GGUF_MMPROJ_PATH
#   - LIGHTONOCR_LLAVA_CLI_PATH
# - If the GGUF env vars are unset, and a standalone dir is available, this script falls back to:
#   - <standalone>/data/models/lightonocr-gguf/LightOnOCR-2-1B-Q4_K_M.gguf
#   - <standalone>/data/models/lightonocr-gguf/LightOnOCR-2-1B-mmproj-f16.gguf

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

function Strip-Wrappers {
    param(
        [Parameter(Mandatory = $true)][string]$Value
    )

    $raw = $Value.Trim()
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return ""
    }

    $wrappers = @(
        @('"', '"'),
        @("'", "'"),
        @("<", ">")
    )
    foreach ($w in $wrappers) {
        $start = $w[0]
        $end = $w[1]
        if ($raw.StartsWith($start) -and $raw.EndsWith($end) -and $raw.Length -ge 2) {
            $raw = $raw.Substring(1, $raw.Length - 2).Trim()
        }
    }

    return $raw
}

function Normalize-EnvPath {
    param(
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ""
    }

    $raw = Strip-Wrappers -Value $Value

    if ($raw -match '^(?i)file://') {
        try {
            $uri = [System.Uri]$raw
            if ($uri.Scheme -eq "file") {
                $raw = $uri.LocalPath
            }
        }
        catch {
            # ignore parse errors; keep raw
        }
    }

    return $raw.Trim()
}

function Resolve-PathFromRepo {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [AllowEmptyString()][string]$PathValue = ""
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return ""
    }

    $p = $PathValue.Trim()
    if (-not [System.IO.Path]::IsPathRooted($p)) {
        $p = Join-Path $RepoRoot $p
    }
    return [System.IO.Path]::GetFullPath($p)
}

function Resolve-LlavaCli {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$StandaloneDirFull,
        [AllowEmptyString()][string]$EnvLlavaCliPath = ""
    )

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($EnvLlavaCliPath)) {
        $candidates += $EnvLlavaCliPath
    }
    else {
        $bundled = Join-Path $StandaloneDirFull "bin\llama.cpp\llava-cli.exe"
        if (Test-Path -LiteralPath $bundled -PathType Leaf) {
            $candidates += $bundled
        }
    }

    $candidates += @("llava-cli", "llava-cli.exe", "llama-llava-cli", "llama-llava-cli.exe")

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }

        $candidateTrimmed = $candidate.Trim()
        $candidatePath = Resolve-PathFromRepo -RepoRoot $RepoRoot -PathValue $candidateTrimmed
        if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
            return $candidatePath
        }

        $cmd = Get-Command $candidateTrimmed -ErrorAction SilentlyContinue
        if ($cmd -and -not [string]::IsNullOrWhiteSpace($cmd.Source)) {
            return $cmd.Source
        }
    }

    throw (
        "llava-cli executable not found.`n" +
        "Fix:`n" +
        "  - Set LIGHTONOCR_LLAVA_CLI_PATH to the full path of llava-cli(.exe) (or llama-llava-cli(.exe))`n" +
        "  - Or ensure it is available on PATH.`n" +
        "  - If using the standalone build, confirm <standalone>\\bin\\llama.cpp\\llava-cli.exe exists."
    )
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if ([string]::IsNullOrWhiteSpace($StandaloneDir)) {
    $StandaloneDir = Join-Path $repoRoot "dist\\standalone"
}
if (-not [System.IO.Path]::IsPathRooted($StandaloneDir)) {
    $StandaloneDir = Join-Path $repoRoot $StandaloneDir
}
$StandaloneDir = [System.IO.Path]::GetFullPath($StandaloneDir)

Write-Host "Repo:          $repoRoot" -ForegroundColor Cyan
Write-Host "StandaloneDir: $StandaloneDir" -ForegroundColor Cyan

$envModelRaw = Normalize-EnvPath -Value $env:LIGHTONOCR_GGUF_MODEL_PATH
$envMmprojRaw = Normalize-EnvPath -Value $env:LIGHTONOCR_GGUF_MMPROJ_PATH
$envLlavaRaw = Normalize-EnvPath -Value $env:LIGHTONOCR_LLAVA_CLI_PATH

$modelPath = Resolve-PathFromRepo -RepoRoot $repoRoot -PathValue $envModelRaw
$mmprojPath = Resolve-PathFromRepo -RepoRoot $repoRoot -PathValue $envMmprojRaw

if ([string]::IsNullOrWhiteSpace($envModelRaw) -or [string]::IsNullOrWhiteSpace($envMmprojRaw)) {
    if (-not $SkipStandaloneFallback -and (Test-Path -LiteralPath $StandaloneDir -PathType Container)) {
        $fallbackModel = Join-Path $StandaloneDir "data\models\lightonocr-gguf\LightOnOCR-2-1B-Q4_K_M.gguf"
        $fallbackMmproj = Join-Path $StandaloneDir "data\models\lightonocr-gguf\LightOnOCR-2-1B-mmproj-f16.gguf"
        if ((Test-Path -LiteralPath $fallbackModel -PathType Leaf) -and (Test-Path -LiteralPath $fallbackMmproj -PathType Leaf)) {
            $modelPath = $fallbackModel
            $mmprojPath = $fallbackMmproj
            Write-Host "Using standalone staged GGUF paths (env vars are unset)." -ForegroundColor Yellow
        }
    }
}

Assert-True (-not [string]::IsNullOrWhiteSpace($modelPath)) "Missing LIGHTONOCR_GGUF_MODEL_PATH (and no standalone fallback found)."
Assert-True (-not [string]::IsNullOrWhiteSpace($mmprojPath)) "Missing LIGHTONOCR_GGUF_MMPROJ_PATH (and no standalone fallback found)."
Assert-True (Test-Path -LiteralPath $modelPath -PathType Leaf) "GGUF model file not found: $modelPath"
Assert-True (Test-Path -LiteralPath $mmprojPath -PathType Leaf) "GGUF mmproj file not found: $mmprojPath"

Write-Host "Model:  $modelPath" -ForegroundColor Green
Write-Host "Mmproj: $mmprojPath" -ForegroundColor Green

$llavaCli = Resolve-LlavaCli -RepoRoot $repoRoot -StandaloneDirFull $StandaloneDir -EnvLlavaCliPath $envLlavaRaw
Write-Host "llava-cli: $llavaCli" -ForegroundColor Green

Write-Host ""  # spacer
Write-Host "Running: $llavaCli --help" -ForegroundColor Cyan
$helpOutput = & $llavaCli --help 2>&1
$helpExitCode = $LASTEXITCODE
Write-Host "exit_code: $helpExitCode" -ForegroundColor Cyan

if ($helpOutput) {
    ($helpOutput | ForEach-Object { $_.ToString() }) | Write-Host
}

if ($helpExitCode -ne 0) {
    $combined = ""
    if ($helpOutput) {
        $combined = ($helpOutput | ForEach-Object { $_.ToString() }) -join "`n"
    }

    if ($combined -match '(?i)deprecated' -and $combined -match '(?i)llama-mtmd-cli') {
        Write-Warning "Detected a deprecated llava-cli shim. Your bundle may require llama-mtmd-cli instead."
        $mtmdSibling = Join-Path (Split-Path -Parent $llavaCli) "llama-mtmd-cli.exe"
        if (Test-Path -LiteralPath $mtmdSibling -PathType Leaf) {
            Write-Host "Found llama-mtmd-cli next to llava-cli: $mtmdSibling" -ForegroundColor Yellow
        }
        else {
            $mtmdCmd = Get-Command "llama-mtmd-cli" -ErrorAction SilentlyContinue
            if ($mtmdCmd -and -not [string]::IsNullOrWhiteSpace($mtmdCmd.Source)) {
                Write-Host "Found llama-mtmd-cli on PATH: $($mtmdCmd.Source)" -ForegroundColor Yellow
            }
            else {
                Write-Host "llama-mtmd-cli not found next to llava-cli and not on PATH." -ForegroundColor Yellow
            }
        }
    }

    Write-Host ""  # spacer
    Write-Host "Fix:" -ForegroundColor Yellow
    Write-Host "  - Ensure the resolved llava-cli path points to a working executable."
    Write-Host "  - If using the standalone build, ensure required llama.cpp DLLs are next to the executable under <standalone>\\bin\\llama.cpp\\."
    Write-Host "  - If you see a deprecation warning telling you to use llama-mtmd-cli, update the bundle/runtime to use llama-mtmd-cli."

    Write-Error "llava-cli --help failed with exit code $helpExitCode"
    exit $helpExitCode
}

Write-Host ""  # spacer
Write-Host "OK: llava-cli launches and GGUF paths are present." -ForegroundColor Green
Write-Host "OK: llava-cli runtime self-check completed." -ForegroundColor Green

[CmdletBinding()]
param(
    [string]$StandaloneDir = "",
    [switch]$SkipStandaloneFallback
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Self-check for the llama.cpp multimodal CLI based OCR runtime prerequisites. No network access required.
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
#   - LIGHTONOCR_LLAVA_CLI_PATH (can point to llama-mtmd-cli.exe / llava-cli.exe / llama-llava-cli.exe)
# - If the GGUF env vars are unset, and a standalone dir is available, this script falls back to:
#   - <standalone>/data/models/lightonocr-gguf/granite-docling-258M-Q4_K_M.gguf
#   - <standalone>/data/models/lightonocr-gguf/mmproj-model-f16.gguf

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

function Resolve-MultimodalCli {
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
        $bundledCandidates = @(
            (Join-Path $StandaloneDirFull "bin\llama.cpp\llama-mtmd-cli.exe"),
            (Join-Path $StandaloneDirFull "bin\llama.cpp\llava-cli.exe"),
            (Join-Path $StandaloneDirFull "bin\llama.cpp\llama-llava-cli.exe")
        )
        foreach ($bundled in $bundledCandidates) {
            if (Test-Path -LiteralPath $bundled -PathType Leaf) {
                $candidates += $bundled
            }
        }
    }

    $candidates += @(
        "llama-mtmd-cli",
        "llama-mtmd-cli.exe",
        "llava-cli",
        "llava-cli.exe",
        "llama-llava-cli",
        "llama-llava-cli.exe"
    )

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
        "llama.cpp multimodal CLI executable not found.`n" +
        "Fix:`n" +
        "  - Set LIGHTONOCR_LLAVA_CLI_PATH to the full path of llama-mtmd-cli(.exe)`n" +
        "    (or llava-cli(.exe) / llama-llava-cli(.exe))`n" +
        "  - Or ensure it is available on PATH.`n" +
        "  - If using the standalone build, confirm <standalone>\\bin\\llama.cpp\\llama-mtmd-cli.exe exists."
    )
}

function Invoke-CliHelp {
    param(
        [Parameter(Mandatory = $true)][string]$ExePath
    )

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $helpOutput = & $ExePath --help 2>&1
        $helpExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }

    $lines = @()
    if ($helpOutput) {
        $lines = ($helpOutput | ForEach-Object { $_.ToString() })
    }

    $combined = ""
    if ($lines) {
        $combined = ($lines -join "`n").Trim()
    }

    return [pscustomobject]@{
        ExitCode = $helpExitCode
        Lines    = $lines
        Text     = $combined
    }
}

function Test-DeprecatedLlavaShim {
    param(
        [AllowEmptyString()][string]$Text = ""
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $false
    }
    return ($Text -match '(?i)deprecated' -and $Text -match '(?i)llama-mtmd-cli')
}

function Test-HelpLooksValid {
    param(
        [AllowEmptyString()][string]$Text = ""
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $false
    }
    return ($Text -match '(?m)^Usage:' -or $Text -match '(?i)--mmproj' -or $Text -match '(?i)--image')
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
        $fallbackModel = Join-Path $StandaloneDir "data\models\lightonocr-gguf\granite-docling-258M-Q4_K_M.gguf"
        $fallbackMmproj = Join-Path $StandaloneDir "data\models\lightonocr-gguf\mmproj-model-f16.gguf"
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

$cli = Resolve-MultimodalCli -RepoRoot $repoRoot -StandaloneDirFull $StandaloneDir -EnvLlavaCliPath $envLlavaRaw
Write-Host "multimodal-cli: $cli" -ForegroundColor Green

Write-Host ""  # spacer
Write-Host "Running: $cli --help" -ForegroundColor Cyan
$result = Invoke-CliHelp -ExePath $cli
Write-Host "exit_code: $($result.ExitCode)" -ForegroundColor Cyan

if ($result.Lines) {
    ($result.Lines | ForEach-Object { $_.ToString() }) | Write-Host
}

if (Test-DeprecatedLlavaShim -Text $result.Text) {
    Write-Warning "Detected a deprecated llava-cli shim. Switching to llama-mtmd-cli for re-check."

    $mtmdSibling = Join-Path (Split-Path -Parent $cli) "llama-mtmd-cli.exe"
    $mtmdCandidate = ""
    if (Test-Path -LiteralPath $mtmdSibling -PathType Leaf) {
        $mtmdCandidate = $mtmdSibling
        Write-Host "Found llama-mtmd-cli next to the shim: $mtmdCandidate" -ForegroundColor Yellow
    }
    else {
        $mtmdCmd = Get-Command "llama-mtmd-cli" -ErrorAction SilentlyContinue
        if ($mtmdCmd -and -not [string]::IsNullOrWhiteSpace($mtmdCmd.Source)) {
            $mtmdCandidate = $mtmdCmd.Source
            Write-Host "Found llama-mtmd-cli on PATH: $mtmdCandidate" -ForegroundColor Yellow
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($mtmdCandidate)) {
        Write-Host ""  # spacer
        Write-Host "Re-check: $mtmdCandidate --help" -ForegroundColor Cyan
        $result = Invoke-CliHelp -ExePath $mtmdCandidate
        Write-Host "exit_code: $($result.ExitCode)" -ForegroundColor Cyan
        if ($result.Lines) {
            ($result.Lines | ForEach-Object { $_.ToString() }) | Write-Host
        }
        $cli = $mtmdCandidate
    }
    else {
        Write-Host "llama-mtmd-cli not found next to the shim and not on PATH." -ForegroundColor Yellow
    }
}

$helpValid = Test-HelpLooksValid -Text $result.Text
if (-not $helpValid) {
    Write-Host ""  # spacer
    Write-Host "Fix:" -ForegroundColor Yellow
    Write-Host "  - Ensure LIGHTONOCR_LLAVA_CLI_PATH points to a working llama.cpp multimodal CLI (recommended: llama-mtmd-cli.exe)."
    Write-Host "  - If using the standalone build, ensure required llama.cpp DLLs are next to the executable under <standalone>\\bin\\llama.cpp\\."
    Write-Host "  - If you see a deprecation warning telling you to use llama-mtmd-cli, rebuild the standalone bundle (or install llama.cpp and set LIGHTONOCR_LLAVA_CLI_PATH)."

    Write-Error "multimodal CLI --help did not look valid (exit_code=$($result.ExitCode))."
    exit 1
}

Write-Host ""  # spacer
Write-Host "OK: multimodal CLI launches and GGUF paths are present." -ForegroundColor Green
Write-Host "OK: multimodal CLI runtime self-check completed." -ForegroundColor Green

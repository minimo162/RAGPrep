[CmdletBinding()]
param(
    [string]$ServerUrl = "http://127.0.0.1:18080",
    [string]$StandaloneDir = "",
    [ValidateRange(1, 3600)]
    [int]$ServerStartupTimeoutSeconds = 180,
    [ValidateRange(1, 100)]
    [int]$Requests = 3,
    [ValidateRange(64, 6000)]
    [int]$ImageSize = 1540,
    [switch]$UseDefaultSizes,
    [string]$LogDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Repro harness for "unknown hard error" crashes during LightOnOCR (llama-server).
#
# This script:
# 1) Starts llama-server (from dist/standalone if available).
# 2) Polls /v1/models until ready.
# 3) Runs a few OCR requests via ragprep.ocr.lightonocr (uv-managed env).
# 4) Prints where logs are and whether llama-server exited unexpectedly.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/repro-unknown-hard-error.ps1
# Optional:
#   powershell -ExecutionPolicy Bypass -File scripts/repro-unknown-hard-error.ps1 -ServerUrl http://127.0.0.1:8080 -Requests 10 -ImageSize 1540

function Normalize-EnvValue {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return "" }
    return $Value.Trim().Trim('"').Trim("'").Trim("<").Trim(">")
}

function Resolve-StandaloneDir {
    param([string]$RepoRoot, [string]$StandaloneDir)
    if (-not [string]::IsNullOrWhiteSpace($StandaloneDir)) {
        if ([System.IO.Path]::IsPathRooted($StandaloneDir)) {
            return [System.IO.Path]::GetFullPath($StandaloneDir)
        }
        return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $StandaloneDir))
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "dist\\standalone"))
}

function Resolve-ServerExe {
    param([string]$StandaloneRoot)

    $candidates = @(
        (Join-Path $StandaloneRoot "bin\\llama.cpp\\vulkan\\llama-server.exe"),
        (Join-Path $StandaloneRoot "bin\\llama.cpp\\avx2\\llama-server.exe"),
        (Join-Path $StandaloneRoot "bin\\llama.cpp\\llama-server.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    $extractRoot = Join-Path $StandaloneRoot "_extract"
    if (Test-Path -LiteralPath $extractRoot -PathType Container) {
        $found = Get-ChildItem -LiteralPath $extractRoot -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) {
            return $found.FullName
        }
    }

    throw "llama-server.exe not found under: $StandaloneRoot"
}

function Resolve-ModelFiles {
    param([string]$StandaloneRoot)

    $ggufDir = Join-Path $StandaloneRoot "data\\models\\lightonocr-gguf"
    if (-not (Test-Path -LiteralPath $ggufDir -PathType Container)) {
        throw "GGUF directory not found: $ggufDir"
    }

    $mmproj = Get-ChildItem -LiteralPath $ggufDir -Filter "mmproj-*.gguf" -ErrorAction SilentlyContinue |
        Sort-Object Name |
        Select-Object -First 1
    if (-not $mmproj) {
        throw "mmproj gguf not found under: $ggufDir"
    }

    $model = Get-ChildItem -LiteralPath $ggufDir -Filter "LightOnOCR-*.gguf" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch "(?i)^mmproj-" } |
        Sort-Object Name |
        Select-Object -First 1
    if (-not $model) {
        $model = Get-ChildItem -LiteralPath $ggufDir -Filter "*.gguf" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notmatch "(?i)^mmproj-" } |
            Sort-Object Name |
            Select-Object -First 1
    }
    if (-not $model) {
        throw "model gguf not found under: $ggufDir"
    }

    return [pscustomobject]@{
        GgufDir   = $ggufDir
        ModelPath = $model.FullName
        MmprojPath = $mmproj.FullName
    }
}

function Wait-ForServerReady {
    param(
        [Parameter(Mandatory = $true)][string]$BaseUrl,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-RestMethod -Uri ($BaseUrl.TrimEnd("/") + "/v1/models") -TimeoutSec 2
            if ($resp) {
                return $resp
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $null
}

function Get-ModelId {
    param([Parameter(Mandatory = $true)]$ModelsResponse)

    if ($ModelsResponse.data -and $ModelsResponse.data[0] -and $ModelsResponse.data[0].id) {
        return $ModelsResponse.data[0].id.ToString()
    }
    if ($ModelsResponse.models -and $ModelsResponse.models[0] -and $ModelsResponse.models[0].name) {
        return $ModelsResponse.models[0].name.ToString()
    }
    throw "Could not determine model id from /v1/models response."
}

function Read-LogTail {
    param([string]$Path, [int]$Lines = 120)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    try {
        return (Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction SilentlyContinue) -join [Environment]::NewLine
    }
    catch {
        return $null
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$standaloneRoot = Resolve-StandaloneDir -RepoRoot $repoRoot -StandaloneDir $StandaloneDir
$standaloneInfo = Resolve-ModelFiles -StandaloneRoot $standaloneRoot
$serverExe = Resolve-ServerExe -StandaloneRoot $standaloneRoot

$serverUri = [Uri]$ServerUrl
$baseUrl = $serverUri.GetLeftPart([System.UriPartial]::Authority).TrimEnd("/")

if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $LogDir = Join-Path $env:TEMP "RAGPrep_repro_unknown_hard_error_$stamp"
}
$LogDir = [System.IO.Path]::GetFullPath($LogDir)
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$stdoutLog = Join-Path $LogDir "llama-server.stdout.log"
$stderrLog = Join-Path $LogDir "llama-server.stderr.log"
$metaLog = Join-Path $LogDir "llama-server.meta.log"
Remove-Item -LiteralPath $stdoutLog, $stderrLog, $metaLog -ErrorAction SilentlyContinue

$bindHost = if ([string]::IsNullOrWhiteSpace($serverUri.Host)) { "127.0.0.1" } else { $serverUri.Host }
$port = if ($serverUri.Port -gt 0) { $serverUri.Port } else { 18080 }

$serverArgs = @(
    "-m", $standaloneInfo.ModelPath,
    "--mmproj", $standaloneInfo.MmprojPath,
    "--host", $bindHost,
    "--port", $port
)

@(
    "start: $(Get-Date -Format o)"
    "repoRoot: $repoRoot"
    "standaloneRoot: $standaloneRoot"
    "serverExe: $serverExe"
    "serverUrl: $ServerUrl"
    "bindHost: $bindHost"
    "port: $port"
    "args: $($serverArgs -join ' ')"
    "model: $($standaloneInfo.ModelPath)"
    "mmproj: $($standaloneInfo.MmprojPath)"
    "logDir: $LogDir"
) | Set-Content -LiteralPath $metaLog -Encoding UTF8

Write-Host "ServerExe:  $serverExe" -ForegroundColor Cyan
Write-Host "Model:      $($standaloneInfo.ModelPath)" -ForegroundColor Cyan
Write-Host "Mmproj:     $($standaloneInfo.MmprojPath)" -ForegroundColor Cyan
Write-Host "ServerUrl:  $ServerUrl" -ForegroundColor Cyan
Write-Host "Logs:       $LogDir" -ForegroundColor Cyan

$serverProcess = $null
try {
    $serverProcess = Start-Process `
        -FilePath $serverExe `
        -ArgumentList $serverArgs `
        -PassThru `
        -WindowStyle Minimized `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog

    Write-Host "llama-server PID: $($serverProcess.Id)" -ForegroundColor Green

    $modelsResponse = Wait-ForServerReady -BaseUrl $baseUrl -TimeoutSeconds $ServerStartupTimeoutSeconds
    if (-not $modelsResponse) {
        $stderrTail = Read-LogTail -Path $stderrLog -Lines 120
        $stdoutTail = if (-not $stderrTail) { Read-LogTail -Path $stdoutLog -Lines 120 } else { $null }
        $detail = @("llama-server did not become ready: $baseUrl/v1/models")
        if ($serverProcess.HasExited) {
            $detail += "exit: exited (ExitCode=$($serverProcess.ExitCode))"
        }
        else {
            $detail += "exit: not exited within wait window"
        }
        $detail += "stdout: $stdoutLog"
        $detail += "stderr: $stderrLog"
        $detail += "meta:   $metaLog"
        if ($stderrTail) { $detail += ""; $detail += "stderr tail:"; $detail += $stderrTail }
        elseif ($stdoutTail) { $detail += ""; $detail += "stdout tail:"; $detail += $stdoutTail }
        throw ($detail -join [Environment]::NewLine)
    }

    $modelId = Get-ModelId -ModelsResponse $modelsResponse
    Write-Host "ModelId:    $modelId" -ForegroundColor Green

    # Run OCR requests through the actual app code path.
    $env:LIGHTONOCR_BACKEND = "llama-server"
    $env:LIGHTONOCR_LLAMA_SERVER_URL = $baseUrl
    $env:LIGHTONOCR_MODEL = $modelId
    if ([string]::IsNullOrWhiteSpace((Normalize-EnvValue -Value $env:LIGHTONOCR_REQUEST_TIMEOUT_SECONDS))) {
        $env:LIGHTONOCR_REQUEST_TIMEOUT_SECONDS = "240"
    }

    $sizes = if ($UseDefaultSizes) { @(800, 1540, 3000) } else { @($ImageSize) }
    $sizesJson = ($sizes | ConvertTo-Json -Compress)

    @"
from __future__ import annotations

from PIL import Image, ImageDraw
from ragprep.ocr import lightonocr

sizes = $sizesJson
requests = int($Requests)

for req_index in range(1, requests + 1):
    for size in sizes:
        img = Image.new("RGB", (int(size), int(size)), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"REQ={req_index} SIZE={size}x{size}", fill="black")
        text = lightonocr.ocr_image(img)
        text = (text or "").strip().replace("\\r\\n", "\\n").replace("\\r", "\\n")
        head = text[:200]
        print(f"[ok] req={req_index} size={size} head={head!r}")
"@ | uv run python -

    if ($serverProcess.HasExited) {
        $stderrTail = Read-LogTail -Path $stderrLog -Lines 200
        throw "llama-server exited unexpectedly (ExitCode=$($serverProcess.ExitCode)).`n`n$stderrTail"
    }

    Write-Host "OK: llama-server stayed alive after OCR requests." -ForegroundColor Green
}
finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
}

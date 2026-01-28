[CmdletBinding()]
param(
    [string]$StandaloneDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Self-check for the LightOnOCR runtime (llama-server only).
#
# Run:
#   powershell -ExecutionPolicy Bypass -File scripts/selfcheck-llava-cli-runtime.ps1
#
# Optional:
#   powershell -ExecutionPolicy Bypass -File scripts/selfcheck-llava-cli-runtime.ps1 -StandaloneDir dist/standalone
#
# Notes:
# - Respects the app env vars:
#   - LIGHTONOCR_BACKEND (must be llama-server)
#   - LIGHTONOCR_LLAMA_SERVER_URL (default: http://127.0.0.1:8080)
#   - LIGHTONOCR_MODEL (required)
#   - LIGHTONOCR_REQUEST_TIMEOUT_SECONDS (default: 60)

function Normalize-EnvValue {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ""
    }
    return $Value.Trim().Trim('"').Trim("'").Trim("<").Trim(">")
}

function Get-PositiveInt {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$DefaultValue
    )

    $raw = Normalize-EnvValue -Value (Get-Item "Env:$Name" -ErrorAction SilentlyContinue).Value
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $DefaultValue
    }

    $parsed = 0
    if (-not [int]::TryParse($raw, [ref]$parsed)) {
        throw "$Name must be an int, got: $raw"
    }
    if ($parsed -le 0) {
        throw "$Name must be > 0, got: $raw"
    }

    return $parsed
}

function Normalize-ServerUrl {
    param(
        [AllowEmptyString()][string]$Value = "",
        [Parameter(Mandatory = $true)][string]$DefaultValue
    )

    $raw = Normalize-EnvValue -Value $Value
    if ([string]::IsNullOrWhiteSpace($raw)) {
        $raw = $DefaultValue
    }
    return $raw.TrimEnd("/")
}

function Get-ModelsFromServer {
    param(
        [Parameter(Mandatory = $true)][string]$ModelsUrl,
        [Parameter(Mandatory = $true)][int]$TimeoutSec
    )

    try {
        $response = Invoke-RestMethod -Uri $ModelsUrl -Method Get -TimeoutSec $TimeoutSec
    }
    catch {
        throw "Failed to reach llama-server at $ModelsUrl. $_"
    }

    if ($null -eq $response) {
        throw "Empty response from llama-server."
    }

    $data = $response.data
    if (-not $data) {
        throw "No models returned from llama-server."
    }

    $models = @()
    foreach ($item in @($data)) {
        if ($item -and $item.id) {
            $models += $item.id.ToString()
        }
    }

    if (-not $models) {
        throw "No model ids returned from llama-server."
    }

    return $models
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

$backendRaw = Normalize-EnvValue -Value $env:LIGHTONOCR_BACKEND
if ([string]::IsNullOrWhiteSpace($backendRaw)) {
    $env:LIGHTONOCR_BACKEND = "llama-server"
    $backendRaw = "llama-server"
    Write-Warning "LIGHTONOCR_BACKEND is not set. Using llama-server."
}
$backend = $backendRaw.ToLower().Replace("_", "-")
if ($backend -ne "llama-server") {
    throw "LIGHTONOCR_BACKEND must be llama-server (got: $backendRaw)."
}

$serverUrl = Normalize-ServerUrl -Value $env:LIGHTONOCR_LLAMA_SERVER_URL -DefaultValue "http://127.0.0.1:8080"
$timeoutSec = Get-PositiveInt -Name "LIGHTONOCR_REQUEST_TIMEOUT_SECONDS" -DefaultValue 60
$modelName = Normalize-EnvValue -Value $env:LIGHTONOCR_MODEL

Write-Host "Backend:       llama-server" -ForegroundColor Green
Write-Host "Server URL:    $serverUrl" -ForegroundColor Green

$modelsUrl = if ($serverUrl.ToLower().EndsWith("/v1")) { "$serverUrl/models" } else { "$serverUrl/v1/models" }
Write-Host "Checking:      $modelsUrl" -ForegroundColor Cyan
$models = Get-ModelsFromServer -ModelsUrl $modelsUrl -TimeoutSec $timeoutSec

if ([string]::IsNullOrWhiteSpace($modelName)) {
    Write-Host "Available models:" -ForegroundColor Yellow
    $models | ForEach-Object { Write-Host "  - $_" }
    throw "Missing LIGHTONOCR_MODEL (required for llama-server)."
}

if ($models -notcontains $modelName) {
    Write-Host "Available models:" -ForegroundColor Yellow
    $models | ForEach-Object { Write-Host "  - $_" }
    throw "Model not found on llama-server: $modelName"
}

Write-Host "Model:         $modelName" -ForegroundColor Green
Write-Host "OK: llama-server reachable and model is available." -ForegroundColor Green

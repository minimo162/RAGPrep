[CmdletBinding()]
param(
    [string]$OutputDir = "dist/standalone",
    [string]$GgufModelFile = "LightOnOCR-2-1B-Q6_K.gguf",
    [string]$GgufMmprojFile = "mmproj-BF16.gguf"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Resolve-OutputDir {
    param([string]$Dir, [string]$Root)

    if ([System.IO.Path]::IsPathRooted($Dir)) {
        return [System.IO.Path]::GetFullPath($Dir)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $Dir))
}

function Assert-Directory {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Missing $Label directory: $Path"
    }
}

function Assert-File {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Label file: $Path"
    }
    $size = (Get-Item -LiteralPath $Path).Length
    if ($size -le 0) {
        throw "$Label file is empty: $Path"
    }
}

$resolvedOutputDir = Resolve-OutputDir -Dir $OutputDir -Root $repoRoot

$pythonExe = Join-Path $resolvedOutputDir "python/python.exe"
$appDir = Join-Path $resolvedOutputDir "app"
$sitePackagesDir = Join-Path $resolvedOutputDir "site-packages"
$llamaCppDir = Join-Path $resolvedOutputDir "bin/llama.cpp"
$llamaCppAvx2Dir = Join-Path $llamaCppDir "avx2"
$llamaCppVulkanDir = Join-Path $llamaCppDir "vulkan"
$llamaServerExe = Join-Path $llamaCppDir "llama-server.exe"
$llamaServerAvx2Exe = Join-Path $llamaCppAvx2Dir "llama-server.exe"
$llamaServerVulkanExe = Join-Path $llamaCppVulkanDir "llama-server.exe"
$ggufDir = Join-Path $resolvedOutputDir "data/models/lightonocr-gguf"
$ggufModelPath = Join-Path $ggufDir $GgufModelFile
$ggufMmprojPath = Join-Path $ggufDir $GgufMmprojFile

Assert-Directory -Path $resolvedOutputDir -Label "standalone output"
Assert-File -Path $pythonExe -Label "python.exe"
Assert-Directory -Path $appDir -Label "app"
Assert-Directory -Path $sitePackagesDir -Label "site-packages"
Assert-Directory -Path $llamaCppDir -Label "llama.cpp bin"
Assert-Directory -Path $llamaCppAvx2Dir -Label "llama.cpp avx2 bin"
Assert-Directory -Path $llamaCppVulkanDir -Label "llama.cpp vulkan bin"
Assert-File -Path $llamaServerExe -Label "llama-server.exe"
Assert-File -Path $llamaServerAvx2Exe -Label "llama-server.exe (avx2)"
Assert-File -Path $llamaServerVulkanExe -Label "llama-server.exe (vulkan)"
Assert-Directory -Path $ggufDir -Label "GGUF directory"
Assert-File -Path $ggufModelPath -Label $GgufModelFile
Assert-File -Path $ggufMmprojPath -Label $GgufMmprojFile

function Resolve-LlamaServerExe {
    param([string]$RootExe, [string]$Avx2Exe, [string]$VulkanExe)
    if (Test-Path -LiteralPath $VulkanExe -PathType Leaf) {
        return $VulkanExe
    }
    if (Test-Path -LiteralPath $Avx2Exe -PathType Leaf) {
        return $Avx2Exe
    }
    return $RootExe
}

function Test-LlamaServer {
    param([string]$BaseUrl)
    try {
        $uri = $BaseUrl.TrimEnd("/") + "/v1/models"
        Invoke-RestMethod -Uri $uri -TimeoutSec 3 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

$serverUrl = "http://127.0.0.1:8080"
$serverExe = Resolve-LlamaServerExe `
    -RootExe $llamaServerExe `
    -Avx2Exe $llamaServerAvx2Exe `
    -VulkanExe $llamaServerVulkanExe

$serverProcess = $null
$startedServer = $false
try {
    if (-not (Test-LlamaServer -BaseUrl $serverUrl)) {
        $serverUri = [Uri]$serverUrl
        $serverArgs = @(
            "-m", $ggufModelPath,
            "--mmproj", $ggufMmprojPath,
            "--host", $serverUri.Host,
            "--port", $serverUri.Port
        )
        $serverProcess = Start-Process -FilePath $serverExe -ArgumentList $serverArgs -PassThru -WindowStyle Minimized
        $startedServer = $true

        $ready = $false
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Milliseconds 500
            if (Test-LlamaServer -BaseUrl $serverUrl) {
                $ready = $true
                break
            }
        }
        if (-not $ready) {
            throw "llama-server failed to start: $serverUrl"
        }
    }
    Write-Host "llama-server health check passed." -ForegroundColor Green
}
finally {
    if ($startedServer -and $serverProcess) {
        Stop-Process -Id $serverProcess.Id -Force
    }
}

Write-Host "Standalone verification passed." -ForegroundColor Green
Write-Host "  output: $resolvedOutputDir" -ForegroundColor DarkGray
Write-Host "  gguf:   $ggufDir" -ForegroundColor DarkGray

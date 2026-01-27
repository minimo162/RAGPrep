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
$mtmdCliExe = Join-Path $llamaCppDir "llama-mtmd-cli.exe"
$ggufDir = Join-Path $resolvedOutputDir "data/models/lightonocr-gguf"
$ggufModelPath = Join-Path $ggufDir $GgufModelFile
$ggufMmprojPath = Join-Path $ggufDir $GgufMmprojFile

Assert-Directory -Path $resolvedOutputDir -Label "standalone output"
Assert-File -Path $pythonExe -Label "python.exe"
Assert-Directory -Path $appDir -Label "app"
Assert-Directory -Path $sitePackagesDir -Label "site-packages"
Assert-Directory -Path $llamaCppDir -Label "llama.cpp bin"
Assert-File -Path $mtmdCliExe -Label "llama-mtmd-cli.exe"
Assert-Directory -Path $ggufDir -Label "GGUF directory"
Assert-File -Path $ggufModelPath -Label $GgufModelFile
Assert-File -Path $ggufMmprojPath -Label $GgufMmprojFile

Write-Host "Standalone verification passed." -ForegroundColor Green
Write-Host "  output: $resolvedOutputDir" -ForegroundColor DarkGray
Write-Host "  gguf:   $ggufDir" -ForegroundColor DarkGray

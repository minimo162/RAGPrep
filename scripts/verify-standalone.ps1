[CmdletBinding()]
param(
    [string]$OutputDir = "dist/standalone"
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
$runPs1 = Join-Path $resolvedOutputDir "run.ps1"
$runCmd = Join-Path $resolvedOutputDir "run.cmd"

Assert-Directory -Path $resolvedOutputDir -Label "standalone output"
Assert-File -Path $pythonExe -Label "python.exe"
Assert-Directory -Path $appDir -Label "app"
Assert-Directory -Path $sitePackagesDir -Label "site-packages"
Assert-File -Path $runPs1 -Label "run.ps1"
Assert-File -Path $runCmd -Label "run.cmd"

Write-Host "Standalone verification passed." -ForegroundColor Green
Write-Host "  output: $resolvedOutputDir" -ForegroundColor DarkGray

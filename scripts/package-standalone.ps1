[CmdletBinding()]
param(
    [string]$StandaloneDir = "",
    [string]$ZipPath = "",
    [string]$MetadataFileName = "BUILD_INFO.txt",
    [string]$SevenZipPath = "",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Resolve-SevenZip {
    param(
        [Parameter(Mandatory = $true)][string]$ExplicitPath
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        $resolved = [System.IO.Path]::GetFullPath($ExplicitPath)
        if (-not (Test-Path $resolved)) {
            throw "7-Zip executable not found at: $resolved"
        }
        return $resolved
    }

    $cmd = Get-Command 7z -ErrorAction SilentlyContinue
    if ($cmd -and -not [string]::IsNullOrWhiteSpace($cmd.Source)) {
        return $cmd.Source
    }

    throw "7z not found on PATH. Install 7-Zip (https://www.7-zip.org/) and ensure 7z.exe is on PATH, or pass -SevenZipPath."
}

function Get-GitCommitShort {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        return "unknown"
    }
    try {
        return (git rev-parse --short HEAD).Trim()
    }
    catch {
        return "unknown"
    }
}

function Get-PythonVersion {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe
    )

    $version = (& $PythonExe -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($version)) {
        throw "Failed to read python version from: $PythonExe"
    }
    return $version
}

function Parse-PbsAssetName {
    param(
        [Parameter(Mandatory = $true)][string]$Name
    )

    $info = [ordered]@{
        pbs_tag     = "unknown"
        pbs_pyver   = "unknown"
        pbs_triple  = "unknown"
        pbs_asset   = $Name
    }

    if ($Name -match "^cpython-(?<pyver>[^+]+)\\+(?<tag>[^-]+)-(?<triple>.+)-install_only\\.tar\\.gz$") {
        $info.pbs_pyver = $Matches.pyver
        $info.pbs_tag = $Matches.tag
        $info.pbs_triple = $Matches.triple
    }

    return [pscustomobject]$info
}

Push-Location $repoRoot
try {
    if ([string]::IsNullOrWhiteSpace($StandaloneDir)) {
        $StandaloneDir = "dist/standalone"
    }
    $StandaloneDir = [System.IO.Path]::GetFullPath($StandaloneDir)
    if (-not (Test-Path $StandaloneDir)) {
        throw "Standalone dir not found: $StandaloneDir. Run scripts/build-standalone.ps1 first."
    }

    $pythonExe = Join-Path $StandaloneDir "python/python.exe"
    if (-not (Test-Path $pythonExe)) {
        $pythonExe = (Get-ChildItem -Path (Join-Path $StandaloneDir "python") -Recurse -File -Filter python.exe |
                Select-Object -First 1).FullName
    }
    if (-not (Test-Path $pythonExe)) {
        throw "python.exe not found under: $StandaloneDir/python"
    }

    $noticesSource = Join-Path $repoRoot "THIRD_PARTY_NOTICES.md"
    if (-not (Test-Path $noticesSource)) {
        throw "Missing THIRD_PARTY_NOTICES.md at repo root: $noticesSource"
    }
    Copy-Item -Force $noticesSource (Join-Path $StandaloneDir "THIRD_PARTY_NOTICES.md")

    $commit = Get-GitCommitShort
    $builtAt = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
    $pythonVersion = Get-PythonVersion -PythonExe $pythonExe

    $pbsTag = "unknown"
    $pbsAsset = "unknown"
    $pbsPyver = "unknown"
    $pbsTriple = "unknown"
    $cacheDir = Join-Path $StandaloneDir "_cache"
    if (Test-Path $cacheDir) {
        $asset = Get-ChildItem -Path $cacheDir -File |
            Where-Object { $_.Name -like "cpython-*-install_only.tar.gz" } |
            Sort-Object Name |
            Select-Object -First 1
        if ($asset) {
            $pbs = Parse-PbsAssetName -Name $asset.Name
            $pbsTag = $pbs.pbs_tag
            $pbsAsset = $pbs.pbs_asset
            $pbsPyver = $pbs.pbs_pyver
            $pbsTriple = $pbs.pbs_triple
        }
    }

    $metadataPath = Join-Path $StandaloneDir $MetadataFileName
    $metadata = @(
        "ragprep standalone build info"
        "commit: $commit"
        "built_at: $builtAt"
        "python_version: $pythonVersion"
        "python_build_standalone_tag: $pbsTag"
        "python_build_standalone_asset: $pbsAsset"
        "python_build_standalone_python: $pbsPyver"
        "python_build_standalone_triple: $pbsTriple"
    )
    Set-Content -Path $metadataPath -Value ($metadata -join "`r`n") -Encoding UTF8

    if ([string]::IsNullOrWhiteSpace($ZipPath)) {
        $zipName = if ($commit -ne "unknown") { "ragprep-standalone-$commit.zip" } else { "ragprep-standalone.zip" }
        $ZipPath = Join-Path (Join-Path $repoRoot "dist") $zipName
    }
    $ZipPath = [System.IO.Path]::GetFullPath($ZipPath)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ZipPath) | Out-Null

    if ((Test-Path $ZipPath) -and (-not $Force)) {
        throw "Zip already exists: $ZipPath (use -Force to overwrite)"
    }
    if (Test-Path $ZipPath) {
        Remove-Item -Force $ZipPath
    }

    $items = @(
        "python"
        "site-packages"
        "app"
        "data"
        "run.ps1"
        "run.cmd"
        "requirements.txt"
        "THIRD_PARTY_NOTICES.md"
        $MetadataFileName
    )

    Push-Location $StandaloneDir
    try {
        $missing = @($items | Where-Object { -not (Test-Path $_) })
        if ($missing.Count -gt 0) {
            Write-Warning "Missing items in standalone dir: $($missing -join ', ')"
        }
        $existing = @($items | Where-Object { Test-Path $_ })
        if ($existing.Count -eq 0) {
            throw "No expected standalone items found in: $StandaloneDir"
        }

        $sevenZipExe = Resolve-SevenZip -ExplicitPath $SevenZipPath
        & $sevenZipExe a -tzip -mx=9 $ZipPath $existing | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "7z failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }

    Write-Host "Wrote metadata: $metadataPath" -ForegroundColor Cyan
    Write-Host "Wrote zip:      $ZipPath" -ForegroundColor Green
}
finally {
    Pop-Location
}

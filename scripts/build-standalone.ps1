[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$PythonVersion = "3.11.14",
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [string]$PbsRelease = "latest",
    [string]$PipTempRoot = "",
    [string]$ModelId = "lightonai/LightOnOCR-2-1B",
    [switch]$SkipModelPrefetch,
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Assert-LastExitCode {
    param(
        [Parameter(Mandatory = $true)][string]$Step
    )

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

Push-Location $repoRoot
try {
    if ([string]::IsNullOrWhiteSpace($OutputDir)) {
        $OutputDir = "dist/standalone"
    }
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)

    if ($Clean -and (Test-Path $OutputDir)) {
        Remove-Item -Recurse -Force $OutputDir
    }

    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

    $cacheDir = Join-Path $OutputDir "_cache"
    $extractDir = Join-Path $OutputDir "_extract"
    $pythonDir = Join-Path $OutputDir "python"
    $sitePackagesDir = Join-Path $OutputDir "site-packages"
    $appDir = Join-Path $OutputDir "app"
    $requirementsPath = Join-Path $OutputDir "requirements.txt"

    foreach ($dir in @($cacheDir, $extractDir, $sitePackagesDir, $appDir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv not found on PATH. Install from https://docs.astral.sh/uv/"
    }
    if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
        throw "tar not found on PATH. Install bsdtar or use Windows 10+ built-in tar."
    }

    $headers = @{ "User-Agent" = "ragprep-standalone-build" }
    if ($PbsRelease -eq "latest") {
        $release = Invoke-RestMethod `
            -Uri "https://api.github.com/repos/indygreg/python-build-standalone/releases/latest" `
            -Headers $headers
    }
    else {
        $release = Invoke-RestMethod `
            -Uri "https://api.github.com/repos/indygreg/python-build-standalone/releases/tags/$PbsRelease" `
            -Headers $headers
    }

    $assetPattern = "cpython-$PythonVersion+*-$TargetTriple-install_only.tar.gz"
    $asset = $release.assets | Where-Object { $_.name -like $assetPattern } | Select-Object -First 1
    if (-not $asset) {
        $assetNames = ($release.assets | Select-Object -ExpandProperty name) -join "`n"
        throw "No asset matched '$assetPattern' in release '$($release.tag_name)'. Available assets:`n$assetNames"
    }

    $archivePath = Join-Path $cacheDir $asset.name
    if (-not (Test-Path $archivePath)) {
        Write-Host "Downloading $($asset.name)..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $archivePath
    }
    else {
        Write-Host "Using cached $($asset.name)" -ForegroundColor Cyan
    }

    if (Test-Path $extractDir) {
        Remove-Item -Recurse -Force $extractDir
    }
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

    Write-Host "Extracting Python runtime..." -ForegroundColor Cyan
    & tar -xf $archivePath -C $extractDir
    Assert-LastExitCode "tar extract"

    $pythonExeCandidate = Get-ChildItem -Path $extractDir -Recurse -File -Filter python.exe |
        Where-Object {
            $dll = Get-ChildItem -Path $_.Directory.FullName -File -Filter "python*.dll" -ErrorAction SilentlyContinue |
                Select-Object -First 1
            $null -ne $dll
        } |
        Select-Object -First 1
    if (-not $pythonExeCandidate) {
        throw "Could not locate python.exe in extracted archive."
    }

    $pythonSourceDir = $pythonExeCandidate.Directory.FullName
    $pythonSourceExe = Join-Path $pythonSourceDir "python.exe"

    Write-Host "Relocating extracted Python runtime..." -ForegroundColor Cyan
    Write-Host "  source: $pythonSourceDir" -ForegroundColor DarkGray
    Write-Host "  dest:   $pythonDir" -ForegroundColor DarkGray

    if (-not (Test-Path $pythonSourceExe)) {
        throw "Expected python.exe missing before Move-Item: $pythonSourceExe"
    }

    Write-Host "Source directory (top-level):" -ForegroundColor DarkGray
    Get-ChildItem -Path $pythonSourceDir -Force |
        Sort-Object Name |
        Select-Object Mode, Length, LastWriteTime, Name |
        Format-Table -AutoSize

    try {
        if (Test-Path $pythonDir) {
            Remove-Item -Recurse -Force $pythonDir
        }
        Move-Item -Path $pythonSourceDir -Destination $pythonDir
    }
    catch {
        $msg = $_.Exception.Message
        $isAccessDenied =
        ($_.Exception -is [System.UnauthorizedAccessException]) -or
        (($_.Exception -is [System.IO.IOException]) -and (($msg -match "(?i)access is denied") -or ($msg -match "アクセス.*拒否")))

        Write-Warning "Failed to move extracted Python runtime."
        Write-Warning "  source: $pythonSourceDir"
        Write-Warning "  dest:   $pythonDir"
        Write-Warning "  error:  $msg"

        try {
            $sourceItem = Get-Item -LiteralPath $pythonSourceDir -Force
            Write-Warning "  source_attrs: $($sourceItem.Attributes)"
        }
        catch {
            Write-Warning "  source_attrs: (failed to read)"
        }
        try {
            $acl = Get-Acl -LiteralPath $pythonSourceDir
            Write-Warning "  source_owner: $($acl.Owner)"
        }
        catch {
            Write-Warning "  source_owner: (failed to read)"
        }

        if ($isAccessDenied) {
            Write-Warning "AccessDenied remediation hints:"
            Write-Warning "  - Close any process using '$OutputDir' (Explorer panes can lock files)."
            Write-Warning "  - Antivirus/indexer may temporarily lock newly extracted files; retry after a short wait."
            Write-Warning "  - Re-run with -Clean."
        }

        throw
    }
    finally {
        if (Test-Path $extractDir) {
            try {
                Remove-Item -Recurse -Force $extractDir
            }
            catch {
                Write-Warning "Failed to clean up extract dir (best-effort): $extractDir"
            }
        }
    }

    $pythonExe = Join-Path $pythonDir "python.exe"
    if (-not (Test-Path $pythonExe)) {
        $pythonExe = (Get-ChildItem -Path $pythonDir -Recurse -File -Filter python.exe | Select-Object -First 1).FullName
    }
    if (-not (Test-Path $pythonExe)) {
        throw "python.exe missing under $pythonDir after extraction."
    }

    Write-Host "Exporting locked requirements..." -ForegroundColor Cyan
    uv export --format requirements.txt --no-dev --frozen --no-hashes --no-emit-project -o $requirementsPath | Out-Null
    Assert-LastExitCode "uv export"

    Write-Host "Bootstrapping pip..." -ForegroundColor Cyan
    try {
        & $pythonExe -m pip --version | Out-Null
    }
    catch {
        & $pythonExe -m ensurepip --upgrade | Out-Null
        Assert-LastExitCode "ensurepip"
    }
    & $pythonExe -m pip install --upgrade pip | Out-Null
    Assert-LastExitCode "pip bootstrap"

    Write-Host "Installing dependencies (this can take a while)..." -ForegroundColor Cyan
    if (Test-Path $sitePackagesDir) {
        Remove-Item -Recurse -Force $sitePackagesDir
    }
    New-Item -ItemType Directory -Force -Path $sitePackagesDir | Out-Null

    $origTemp = $env:TEMP
    $origTmp = $env:TMP
    $origGitConfig = $env:GIT_CONFIG_PARAMETERS
    $origPipVersionCheck = $env:PIP_DISABLE_PIP_VERSION_CHECK
    $pipTempRoot = $PipTempRoot
    if ([string]::IsNullOrWhiteSpace($pipTempRoot)) {
        $pipTempRoot = Join-Path $env:LOCALAPPDATA "t"
    }
    $pipTempName = "p" + [guid]::NewGuid().ToString("N").Substring(0, 8)
    $pipTemp = Join-Path $pipTempRoot $pipTempName
    New-Item -ItemType Directory -Force -Path $pipTempRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $pipTemp | Out-Null

    $env:TEMP = $pipTemp
    $env:TMP = $pipTemp
    $env:GIT_CONFIG_PARAMETERS = "'core.longpaths=true'"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    try {
        & $pythonExe -m pip install --no-deps --target $sitePackagesDir -r $requirementsPath
        Assert-LastExitCode "pip install"
    }
    finally {
        $env:TEMP = $origTemp
        $env:TMP = $origTmp
        $env:GIT_CONFIG_PARAMETERS = $origGitConfig
        $env:PIP_DISABLE_PIP_VERSION_CHECK = $origPipVersionCheck
        Remove-Item -Recurse -Force $pipTemp -ErrorAction SilentlyContinue
    }

    Write-Host "Copying app source..." -ForegroundColor Cyan
    if (Test-Path $appDir) {
        Remove-Item -Recurse -Force $appDir
    }
    New-Item -ItemType Directory -Force -Path $appDir | Out-Null
    Copy-Item -Recurse -Force (Join-Path $repoRoot "ragprep") $appDir

    $noticesPath = Join-Path $repoRoot "THIRD_PARTY_NOTICES.md"
    if (Test-Path $noticesPath) {
        Copy-Item -Force $noticesPath (Join-Path $OutputDir "THIRD_PARTY_NOTICES.md")
    }
    else {
        Write-Warning "THIRD_PARTY_NOTICES.md not found at repo root; standalone output will not include notices."
    }

    $dataDir = Join-Path $OutputDir "data"
    $hfHomeDir = Join-Path $dataDir "hf"
    New-Item -ItemType Directory -Force -Path $hfHomeDir | Out-Null

    if (-not $SkipModelPrefetch) {
        $modelIdTrimmed = $ModelId.Trim()
        if ([string]::IsNullOrWhiteSpace($modelIdTrimmed)) {
            throw "ModelId must be non-empty (got: $ModelId)"
        }

        Write-Host "Prefetching LightOnOCR model (this can take a while)..." -ForegroundColor Cyan
        Write-Host "  model: $modelIdTrimmed" -ForegroundColor DarkGray
        Write-Host "  HF_HOME: $hfHomeDir" -ForegroundColor DarkGray

        $origHfHome = $env:HF_HOME
        $origModelId = $env:LIGHTONOCR_MODEL_ID
        $origPythonPath = $env:PYTHONPATH
        $origNoUserSite = $env:PYTHONNOUSERSITE
        $origPythonUtf8 = $env:PYTHONUTF8
        try {
            $env:HF_HOME = $hfHomeDir
            $env:LIGHTONOCR_MODEL_ID = $modelIdTrimmed
            $env:PYTHONNOUSERSITE = "1"
            $env:PYTHONUTF8 = "1"
            $env:PYTHONPATH = (Join-Path $OutputDir "app") + ";" + (Join-Path $OutputDir "site-packages")

            $prefetchPy = @'
import os

model_id = os.environ.get("LIGHTONOCR_MODEL_ID") or "lightonai/LightOnOCR-2-1B"
print(f"Prefetching LightOnOCR model: {model_id}")

import transformers

processor_cls = getattr(transformers, "LightOnOcrProcessor", None)
model_cls = getattr(transformers, "LightOnOcrForConditionalGeneration", None)
if processor_cls is None or model_cls is None:
    raise RuntimeError(
        "transformers does not expose LightOnOcr* classes. "
        "LightOnOCR-2 requires transformers installed from source."
    )

processor_cls.from_pretrained(model_id)
model_cls.from_pretrained(model_id)

print("Prefetch complete.")
'@

            & $pythonExe -c $prefetchPy
            Assert-LastExitCode "model prefetch"
        }
        finally {
            if ($null -ne $origHfHome) { $env:HF_HOME = $origHfHome } else { Remove-Item env:HF_HOME -ErrorAction SilentlyContinue }
            if ($null -ne $origModelId) { $env:LIGHTONOCR_MODEL_ID = $origModelId } else { Remove-Item env:LIGHTONOCR_MODEL_ID -ErrorAction SilentlyContinue }
            if ($null -ne $origPythonPath) { $env:PYTHONPATH = $origPythonPath } else { Remove-Item env:PYTHONPATH -ErrorAction SilentlyContinue }
            if ($null -ne $origNoUserSite) { $env:PYTHONNOUSERSITE = $origNoUserSite } else { Remove-Item env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue }
            if ($null -ne $origPythonUtf8) { $env:PYTHONUTF8 = $origPythonUtf8 } else { Remove-Item env:PYTHONUTF8 -ErrorAction SilentlyContinue }
        }
    }
    else {
        Write-Host "Skipping model prefetch (-SkipModelPrefetch)." -ForegroundColor Yellow
    }

    $runPs1 = @"
[CmdletBinding()]
param(
    [string]`$Host = "127.0.0.1",
    [int]`$Port = 8000
)

Set-StrictMode -Version Latest
`$ErrorActionPreference = "Stop"

`$root = Split-Path -Parent `$MyInvocation.MyCommand.Path
`$pythonExe = Join-Path `$root "python/python.exe"
if (-not (Test-Path `$pythonExe)) {
    throw "Missing `$pythonExe. Run scripts/build-standalone.ps1 first."
}

if (-not `$env:HF_HOME -or [string]::IsNullOrWhiteSpace(`$env:HF_HOME)) {
    `$hfHome = Join-Path `$root "data/hf"
    New-Item -ItemType Directory -Force -Path `$hfHome | Out-Null
    `$env:HF_HOME = `$hfHome
}

`$env:PYTHONNOUSERSITE = "1"
`$env:PYTHONUTF8 = "1"
`$env:PYTHONPATH = (Join-Path `$root "app") + ";" + (Join-Path `$root "site-packages")

& `$pythonExe -m uvicorn ragprep.web.app:app --host `$Host --port `$Port
"@
    Set-Content -Path (Join-Path $OutputDir "run.ps1") -Value $runPs1 -Encoding UTF8

    $runCmd = @"
@echo off
setlocal
set ROOT=%~dp0
if "%HF_HOME%"=="" (
  set HF_HOME=%ROOT%data\hf
  if not exist "%HF_HOME%" mkdir "%HF_HOME%"
)
set PYTHONNOUSERSITE=1
set PYTHONUTF8=1
set PYTHONPATH=%ROOT%app;%ROOT%site-packages
"%ROOT%python\python.exe" -m uvicorn ragprep.web.app:app --host 127.0.0.1 --port 8000
"@
    Set-Content -Path (Join-Path $OutputDir "run.cmd") -Value $runCmd -Encoding ASCII

    Write-Host "Done." -ForegroundColor Green
    Write-Host "Output: $OutputDir"
    Write-Host "Run:    $OutputDir/run.ps1"
}
finally {
    Pop-Location
}

[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$PythonVersion = "3.11.14",
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [string]$PbsRelease = "latest",
    [string]$PipTempRoot = "",
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

function Get-RequiredTrimmedString {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $false)][string]$Value
    )

    if ($null -eq $Value) {
        throw "$Name must be non-empty (got: <null>)"
    }

    $trimmed = $Value.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
        throw "$Name must be non-empty (got: '$Value')"
    }

    return $trimmed
}

$currentStep = "initializing"
$startedAt = Get-Date

# filesystem helpers (retry + robust move)
. (Join-Path $PSScriptRoot "standalone-io.ps1")

Push-Location $repoRoot
try {
    $currentStep = "normalize parameters"
    $PythonVersion = Get-RequiredTrimmedString -Name "PythonVersion" -Value $PythonVersion
    $TargetTriple = Get-RequiredTrimmedString -Name "TargetTriple" -Value $TargetTriple
    $PbsRelease = Get-RequiredTrimmedString -Name "PbsRelease" -Value $PbsRelease

    if ([string]::IsNullOrWhiteSpace($OutputDir)) {
        $OutputDir = "dist/standalone"
    }
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
    $cacheDir = Join-Path $OutputDir "_cache"
    $pipCacheDir = Join-Path $cacheDir "pip"

    $pipTempRootResolved = $PipTempRoot
    if ([string]::IsNullOrWhiteSpace($pipTempRootResolved)) {
        $pipTempRootResolved = Join-Path $env:LOCALAPPDATA "t"
    }
    $pipTempRootResolved = [System.IO.Path]::GetFullPath($pipTempRootResolved)

    Write-Host "RAGPrep standalone build" -ForegroundColor Cyan
    Write-Host "  repo:       $repoRoot" -ForegroundColor DarkGray
    Write-Host "  output:     $OutputDir" -ForegroundColor DarkGray
    Write-Host "  python:     $PythonVersion ($TargetTriple)" -ForegroundColor DarkGray
    Write-Host "  pbs:        $PbsRelease" -ForegroundColor DarkGray
    Write-Host "  clean:      $($Clean.IsPresent)" -ForegroundColor DarkGray
    Write-Host "  pip_temp:   $pipTempRootResolved" -ForegroundColor DarkGray
    Write-Host "  pip_cache:  $pipCacheDir" -ForegroundColor DarkGray

    if ($Clean -and (Test-Path $OutputDir)) {
        try {
            Invoke-RetryIO -Step "Clean output dir" -Action {
                Remove-Item -LiteralPath $OutputDir -Recurse -Force
            } | Out-Null
        }
        catch [System.IO.IOException], [System.UnauthorizedAccessException] {
            $suffix = (Get-Date).ToString("yyyyMMdd-HHmmss") + "-" + [guid]::NewGuid().ToString("N")
            $trashDir = "$OutputDir.old-$suffix"
            Write-Warning "Clean output dir failed. Renaming existing output dir to: $trashDir"
            Invoke-RetryIO -Step "Rename output dir" -Action {
                Move-Item -LiteralPath $OutputDir -Destination $trashDir -Force
            } | Out-Null
        }
    }

    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

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

    $currentStep = "fetch python-build-standalone release metadata"
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
        $currentStep = "download python-build-standalone archive"
        Write-Host "Downloading $($asset.name)..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $archivePath
    }
    else {
        Write-Host "Using cached $($asset.name)" -ForegroundColor Cyan
    }

    if (Test-Path $extractDir) {
        Invoke-RetryIO -Step "Remove extract dir" -Action { Remove-Item -LiteralPath $extractDir -Recurse -Force } | Out-Null
    }
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

    $currentStep = "extract python runtime"
    Write-Host "Extracting Python runtime..." -ForegroundColor Cyan
    & tar -xf $archivePath -C $extractDir
    Assert-LastExitCode "tar extract"

    $currentStep = "relocate python runtime"
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

    try {
        Move-DirectoryRobust -SourceDir $pythonSourceDir -DestinationDir $pythonDir -VerifyRelativePath "python.exe" | Out-Null
    }
    finally {
        if (Test-Path $extractDir) {
            try {
                Invoke-RetryIO -Step "Cleanup extract dir" -Retries 3 -Action { Remove-Item -LiteralPath $extractDir -Recurse -Force } | Out-Null
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

    $currentStep = "uv export requirements"
    Write-Host "Exporting locked requirements (uv.lock)..." -ForegroundColor Cyan
    uv export --format requirements.txt --no-dev --frozen --no-hashes --no-emit-project -o $requirementsPath | Out-Null
    Assert-LastExitCode "uv export"

    $currentStep = "bootstrap pip"
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
        Invoke-RetryIO -Step "Remove site-packages dir" -Retries 3 -Action {
            Remove-Item -LiteralPath $sitePackagesDir -Recurse -Force
        } | Out-Null
    }
    New-Item -ItemType Directory -Force -Path $sitePackagesDir | Out-Null

    $origTemp = $env:TEMP
    $origTmp = $env:TMP
    $origGitConfig = $env:GIT_CONFIG_PARAMETERS
    $origPipVersionCheck = $env:PIP_DISABLE_PIP_VERSION_CHECK
    $origPipCacheDir = $env:PIP_CACHE_DIR
    $pipTempRoot = $pipTempRootResolved
    $pipTempName = "p" + [guid]::NewGuid().ToString("N").Substring(0, 8)
    $pipTemp = Join-Path $pipTempRoot $pipTempName
    New-Item -ItemType Directory -Force -Path $pipTempRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $pipTemp | Out-Null

    $env:TEMP = $pipTemp
    $env:TMP = $pipTemp
    $env:GIT_CONFIG_PARAMETERS = "'core.longpaths=true'"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $env:PIP_CACHE_DIR = $pipCacheDir
    New-Item -ItemType Directory -Force -Path $pipCacheDir | Out-Null

    $currentStep = "pip install dependencies"
    $pipArgs = @(
        "install",
        "--no-deps",
        "--target", $sitePackagesDir,
        "--no-input",
        "--progress-bar", "off",
        "--retries", "10",
        "--timeout", "60",
        "-r", $requirementsPath
    )
    try {
        & $pythonExe -m pip @pipArgs
        Assert-LastExitCode "pip install"
    }
    finally {
        $env:TEMP = $origTemp
        $env:TMP = $origTmp
        if ($null -ne $origGitConfig) { $env:GIT_CONFIG_PARAMETERS = $origGitConfig } else { Remove-Item env:GIT_CONFIG_PARAMETERS -ErrorAction SilentlyContinue }
        if ($null -ne $origPipVersionCheck) { $env:PIP_DISABLE_PIP_VERSION_CHECK = $origPipVersionCheck } else { Remove-Item env:PIP_DISABLE_PIP_VERSION_CHECK -ErrorAction SilentlyContinue }
        if ($null -ne $origPipCacheDir) { $env:PIP_CACHE_DIR = $origPipCacheDir } else { Remove-Item env:PIP_CACHE_DIR -ErrorAction SilentlyContinue }
        if (Test-Path $pipTemp) {
            try {
                Invoke-RetryIO -Step "Cleanup pip temp dir" -Retries 3 -Action { Remove-Item -LiteralPath $pipTemp -Recurse -Force } | Out-Null
            }
            catch {
                Write-Warning "Failed to clean up pip temp dir (best-effort): $pipTemp"
            }
        }
    }

    $currentStep = "copy app sources"
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

    $currentStep = "write run scripts"
    $runPs1 = @"
[CmdletBinding()]
param(
    [Alias("Host")]
    [string]`$BindHost = "127.0.0.1",
    [int]`$Port = 8000
)

Set-StrictMode -Version Latest
`$ErrorActionPreference = "Stop"

`$root = Split-Path -Parent `$MyInvocation.MyCommand.Path
`$pythonExe = Join-Path `$root "python/python.exe"
if (-not (Test-Path `$pythonExe)) {
    throw "Missing `$pythonExe. Run scripts/build-standalone.ps1 first."
}

if (-not `$env:RAGPREP_GLM_OCR_BASE_URL -or [string]::IsNullOrWhiteSpace(`$env:RAGPREP_GLM_OCR_BASE_URL)) {
    `$env:RAGPREP_GLM_OCR_BASE_URL = "http://127.0.0.1:8080"
}
if (-not `$env:RAGPREP_PDF_BACKEND -or [string]::IsNullOrWhiteSpace(`$env:RAGPREP_PDF_BACKEND)) {
    `$env:RAGPREP_PDF_BACKEND = "glm-ocr"
}
if (`$env:RAGPREP_PDF_BACKEND -ne "glm-ocr") {
    throw "RAGPREP_PDF_BACKEND must be 'glm-ocr' (got: `$env:RAGPREP_PDF_BACKEND)."
}

`$probeUrl = `$env:RAGPREP_GLM_OCR_BASE_URL.TrimEnd("/") + "/v1/models"
try {
    `$resp = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri `$probeUrl
    if (`$resp.StatusCode -ne 200) {
        throw "Unexpected status: `$(`$resp.StatusCode)"
    }
}
catch {
    throw "GLM-OCR server is not reachable: `$probeUrl. Start your server (vLLM/SGLang) and retry."
}

`$env:PYTHONNOUSERSITE = "1"
`$env:PYTHONUTF8 = "1"
`$env:PYTHONPATH = (Join-Path `$root "app") + ";" + (Join-Path `$root "site-packages")

& `$pythonExe -m ragprep.desktop --host `$BindHost --port `$Port
"@
    Set-Content -Path (Join-Path $OutputDir "run.ps1") -Value $runPs1 -Encoding UTF8

    $runCmd = @"
@echo off
setlocal
set ROOT=%~dp0

if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage
if "%~1"=="/?" goto :usage

set "BIND_HOST=127.0.0.1"
if not "%RAGPREP_BIND_HOST%"=="" set "BIND_HOST=%RAGPREP_BIND_HOST%"
if not "%~1"=="" set "BIND_HOST=%~1"

set "PORT=8000"
if not "%RAGPREP_PORT%"=="" set "PORT=%RAGPREP_PORT%"
if not "%~2"=="" set "PORT=%~2"

if "%RAGPREP_GLM_OCR_BASE_URL%"=="" (
  set RAGPREP_GLM_OCR_BASE_URL=http://127.0.0.1:8080
)
if "%RAGPREP_PDF_BACKEND%"=="" (
  set RAGPREP_PDF_BACKEND=glm-ocr
)
if /I not "%RAGPREP_PDF_BACKEND%"=="glm-ocr" (
  echo [ERROR] RAGPREP_PDF_BACKEND must be glm-ocr (got: %RAGPREP_PDF_BACKEND%)
  exit /b 1
)

powershell -NoProfile -Command "try { $u=$env:RAGPREP_GLM_OCR_BASE_URL; $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri ($u.TrimEnd('/') + '/v1/models'); if ($r.StatusCode -eq 200) { exit 0 } exit 1 } catch { exit 1 }"
if not "%ERRORLEVEL%"=="0" (
  echo [ERROR] GLM-OCR server is not reachable: %RAGPREP_GLM_OCR_BASE_URL%/v1/models
  echo Start your server (vLLM/SGLang) and retry.
  exit /b 1
)

set PYTHONNOUSERSITE=1
set PYTHONUTF8=1
set PYTHONPATH=%ROOT%app;%ROOT%site-packages
"%ROOT%python\python.exe" -m ragprep.desktop --host %BIND_HOST% --port %PORT%
exit /b %ERRORLEVEL%

:usage
echo Usage: %~nx0 [bind_host] [port]
echo   Defaults: 127.0.0.1 8000
echo   Or set env vars: RAGPREP_BIND_HOST / RAGPREP_PORT
exit /b 0
"@
    Set-Content -Path (Join-Path $OutputDir "run.cmd") -Value $runCmd -Encoding ASCII

    $currentStep = "verify standalone output"
    $verifyScript = Join-Path $repoRoot "scripts/verify-standalone.ps1"
    if (-not (Test-Path -LiteralPath $verifyScript -PathType Leaf)) {
        throw "Missing verify script: $verifyScript"
    }
    & $verifyScript -OutputDir $OutputDir
    Assert-LastExitCode "verify standalone"

    Write-Host "Done." -ForegroundColor Green
    Write-Host "Output: $OutputDir"
    Write-Host "Run:    $OutputDir/run.ps1"
    Write-Host "        $OutputDir/run.cmd"
}
catch {
    $duration = (Get-Date) - $startedAt
    Write-Warning "Standalone build failed after $([int]$duration.TotalSeconds)s."
    Write-Warning "  step:   $currentStep"
    Write-Warning "  repo:   $repoRoot"
    Write-Warning "  output: $OutputDir"
    Write-Warning "  python: $PythonVersion ($TargetTriple)"
    Write-Warning "  pbs:    $PbsRelease"
    Write-Warning "  pip_temp:  $pipTempRootResolved"
    Write-Warning "  pip_cache: $pipCacheDir"
    throw
}
finally {
    Pop-Location
}

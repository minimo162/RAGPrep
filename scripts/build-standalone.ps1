[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$PythonVersion = "3.11.14",
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [string]$PbsRelease = "latest",
    [string]$PipTempRoot = "",
    [string]$LlamaCppPythonExtraIndexUrl = "https://abetlen.github.io/llama-cpp-python/whl/cpu",
    [string]$GgufRepoId = "noctrex/LightOnOCR-2-1B-GGUF",
    [string]$GgufModelFile = "LightOnOCR-2-1B-Q6_K.gguf",
    [string]$GgufMmprojFile = "mmproj-BF16.gguf",
    [switch]$SkipGgufPrefetch,
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
    Write-Host "  gguf_fetch: $(-not $SkipGgufPrefetch.IsPresent)" -ForegroundColor DarkGray
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

    Write-Host "Source directory (top-level):" -ForegroundColor DarkGray
    Get-ChildItem -Path $pythonSourceDir -Force |
        Sort-Object Name |
        Select-Object Mode, Length, LastWriteTime, Name |
        Format-Table -AutoSize

    try {
        $moveInfo = Move-DirectoryRobust -SourceDir $pythonSourceDir -DestinationDir $pythonDir -VerifyRelativePath "python.exe"
        if ($moveInfo.Method -ne "Move-Item" -or $moveInfo.Attempts -gt 1) {
            $suffix = if ($moveInfo.Method -eq "robocopy") { " (exit $($moveInfo.RobocopyExitCode))" } else { "" }
            Write-Host "  relocation: $($moveInfo.Method) after $($moveInfo.Attempts) attempt(s)$suffix" -ForegroundColor DarkGray
        }
    }
    catch {
        $msg = $_.Exception.Message
        $isAccessDenied = ($_.Exception -is [System.UnauthorizedAccessException]) -or ($_.Exception -is [System.IO.IOException])

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
        "--timeout", "60"
    )
    $needsLlamaCppWheelIndex = Select-String -LiteralPath $requirementsPath -Pattern "^llama-cpp-python" -Quiet
    if ($needsLlamaCppWheelIndex) {
        $llamaExtraIndexUrl = $LlamaCppPythonExtraIndexUrl.Trim()
        if ([string]::IsNullOrWhiteSpace($llamaExtraIndexUrl)) {
            throw "LlamaCppPythonExtraIndexUrl must be non-empty because requirements include llama-cpp-python."
        }

        $pipArgs += @(
            "--only-binary", "llama-cpp-python",
            "--extra-index-url", $llamaExtraIndexUrl
        )
    }
    $pipArgs += @("-r", $requirementsPath)
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

    $currentStep = "copy app source"
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

    $ggufModelFileTrimmed = $GgufModelFile.Trim()
    if ([string]::IsNullOrWhiteSpace($ggufModelFileTrimmed)) {
        throw "GgufModelFile must be non-empty (got: $GgufModelFile)"
    }

    $ggufMmprojFileTrimmed = $GgufMmprojFile.Trim()
    if ([string]::IsNullOrWhiteSpace($ggufMmprojFileTrimmed)) {
        throw "GgufMmprojFile must be non-empty (got: $GgufMmprojFile)"
    }

    $ggufOutDir = Join-Path $dataDir "models\\lightonocr-gguf"
    $ggufModelPath = Join-Path $ggufOutDir $ggufModelFileTrimmed
    $ggufMmprojPath = Join-Path $ggufOutDir $ggufMmprojFileTrimmed

    if (-not $SkipGgufPrefetch) {
        $currentStep = "prefetch gguf artifacts"
        $ggufRepoIdTrimmed = $GgufRepoId.Trim()
        if ([string]::IsNullOrWhiteSpace($ggufRepoIdTrimmed)) {
            throw "GgufRepoId must be non-empty (got: $GgufRepoId)"
        }

        New-Item -ItemType Directory -Force -Path $ggufOutDir | Out-Null

        Write-Host "Prefetching LightOnOCR GGUF artifacts (this can take a while)..." -ForegroundColor Cyan
        Write-Host "  repo:   $ggufRepoIdTrimmed" -ForegroundColor DarkGray
        Write-Host "  model:  $ggufModelFileTrimmed" -ForegroundColor DarkGray
        Write-Host "  mmproj: $ggufMmprojFileTrimmed" -ForegroundColor DarkGray
        Write-Host "  out:    $ggufOutDir" -ForegroundColor DarkGray
        Write-Host "  HF_HOME: $hfHomeDir" -ForegroundColor DarkGray

        $origHfHome2 = $env:HF_HOME
        $origPythonPath2 = $env:PYTHONPATH
        $origNoUserSite2 = $env:PYTHONNOUSERSITE
        $origPythonUtf82 = $env:PYTHONUTF8
        try {
            $env:HF_HOME = $hfHomeDir
            $env:PYTHONNOUSERSITE = "1"
            $env:PYTHONUTF8 = "1"
            $env:PYTHONPATH = (Join-Path $OutputDir "app") + ";" + (Join-Path $OutputDir "site-packages")

$prefetchGgufPy = @'
import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--mmproj-file", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    repo_id = args.repo_id
    model_file = args.model_file
    mmproj_file = args.mmproj_file
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prefetching GGUF files from: {repo_id}")
    print(f"  model:  {model_file}")
    print(f"  mmproj: {mmproj_file}")
    print(f"  out:    {out_dir}")
    print(f"  HF_HOME: {os.environ.get('HF_HOME')}")

    def _download_via_hf_hub(filename: str) -> Path | None:
        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            print("huggingface_hub not available, falling back to direct download.")
            return None

        cached_path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
        dest = out_dir / filename
        try:
            if dest.is_file() and dest.stat().st_size == cached_path.stat().st_size:
                return dest
        except OSError:
            pass
        shutil.copy2(cached_path, dest)
        return dest

    def _download_direct(filename: str) -> Path:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=1"
        dest = out_dir / filename
        tmp = dest.with_suffix(dest.suffix + ".part")
        request = Request(url, headers={"User-Agent": "ragprep-standalone-build"})
        try:
            with urlopen(request) as response, open(tmp, "wb") as handle:  # noqa: S310
                shutil.copyfileobj(response, handle)
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Direct download failed: {url}") from exc
        size = tmp.stat().st_size if tmp.exists() else 0
        if size <= 0:
            raise RuntimeError(f"Direct download produced empty file: {url}")
        tmp.replace(dest)
        return dest

    def stage(filename: str) -> Path:
        dest = out_dir / filename
        try:
            if dest.is_file() and dest.stat().st_size > 0:
                return dest
        except OSError:
            pass

        staged = _download_via_hf_hub(filename)
        if staged is not None:
            return staged
        return _download_direct(filename)

    model_dest = stage(model_file)
    mmproj_dest = stage(mmproj_file)

    print("Prefetch complete.")
    print(f"  model_path:  {model_dest}")
    print(f"  mmproj_path: {mmproj_dest}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("GGUF prefetch failed.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
'@

            $prefetchGgufScriptPath = Join-Path $cacheDir ("prefetch-gguf-" + [guid]::NewGuid().ToString("N") + ".py")
            try {
                Set-Content -Path $prefetchGgufScriptPath -Value $prefetchGgufPy -Encoding UTF8

                & $pythonExe $prefetchGgufScriptPath `
                    --repo-id $ggufRepoIdTrimmed `
                    --model-file $ggufModelFileTrimmed `
                    --mmproj-file $ggufMmprojFileTrimmed `
                    --out-dir $ggufOutDir
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "GGUF prefetch failed. To skip, re-run with -SkipGgufPrefetch."
                }
                Assert-LastExitCode "gguf prefetch"
            }
            finally {
                Remove-Item -LiteralPath $prefetchGgufScriptPath -Force -ErrorAction SilentlyContinue
            }
        }
        finally {
            if ($null -ne $origHfHome2) { $env:HF_HOME = $origHfHome2 } else { Remove-Item env:HF_HOME -ErrorAction SilentlyContinue }
            if ($null -ne $origPythonPath2) { $env:PYTHONPATH = $origPythonPath2 } else { Remove-Item env:PYTHONPATH -ErrorAction SilentlyContinue }
            if ($null -ne $origNoUserSite2) { $env:PYTHONNOUSERSITE = $origNoUserSite2 } else { Remove-Item env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue }
            if ($null -ne $origPythonUtf82) { $env:PYTHONUTF8 = $origPythonUtf82 } else { Remove-Item env:PYTHONUTF8 -ErrorAction SilentlyContinue }
        }

        foreach ($ggufPath in @($ggufModelPath, $ggufMmprojPath)) {
            if (-not (Test-Path -LiteralPath $ggufPath -PathType Leaf)) {
                throw "GGUF artifact missing after prefetch: $ggufPath"
            }
            $artifactSize = (Get-Item -LiteralPath $ggufPath).Length
            if ($artifactSize -le 0) {
                throw "GGUF artifact is empty after prefetch: $ggufPath"
            }
        }

        Write-Host "Standalone run scripts default to these GGUF paths:" -ForegroundColor Cyan
        Write-Host "  `$env:LIGHTONOCR_GGUF_MODEL_PATH=$ggufModelPath" -ForegroundColor DarkGray
        Write-Host "  `$env:LIGHTONOCR_GGUF_MMPROJ_PATH=$ggufMmprojPath" -ForegroundColor DarkGray
    }
    else {
        Write-Host "Skipping GGUF prefetch (-SkipGgufPrefetch)." -ForegroundColor Yellow
    }

    # Bundle llama.cpp (vision CLI) for standalone
    # NOTE: We pin URL + SHA256 to avoid supply-chain drift.
    $currentStep = "bundle llama.cpp"
    $llamaCppTag = "b7815"
    $llamaCppBinDir = Join-Path $OutputDir "bin/llama.cpp"
    New-Item -ItemType Directory -Force -Path $llamaCppBinDir | Out-Null
    $llamaCppBaseUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$llamaCppTag"
    $llamaCppVariants = @(
        @{
            Name = "avx2"
            Asset = "llama-$llamaCppTag-bin-win-cpu-x64.zip"
            Sha256 = "7d0fea9f0879cff4a3b6ad16051d28d394566abe7870a20e7f8c14abf9973b57"
        },
        @{
            Name = "vulkan"
            Asset = "llama-$llamaCppTag-bin-win-vulkan-x64.zip"
            Sha256 = "1012aa05900ae8a5685a0c7dbb98a5a9d8b6a9e70cd8119990bc61bdc4e9e475"
        }
    )

    $bundledVariants = @()

    foreach ($variant in $llamaCppVariants) {
        $variantName = $variant.Name
        $variantAsset = $variant.Asset
        $variantSha256 = $variant.Sha256
        $variantUrl = "$llamaCppBaseUrl/$variantAsset"
        $variantArchivePath = Join-Path $cacheDir $variantAsset

        if (-not (Test-Path -LiteralPath $variantArchivePath -PathType Leaf)) {
            Write-Host "Downloading llama.cpp $llamaCppTag ($variantName / $variantAsset)..." -ForegroundColor Cyan
            Invoke-WebRequest -Uri $variantUrl -OutFile $variantArchivePath
        }
        else {
            Write-Host "Using cached llama.cpp bundle ($variantName / $variantAsset)" -ForegroundColor Cyan
        }

        $variantHash = (Get-FileHash -Algorithm SHA256 -Path $variantArchivePath).Hash.ToLowerInvariant()
        if ($variantHash -ne $variantSha256) {
            throw "llama.cpp bundle checksum mismatch ($variantName). expected=$variantSha256 got=$variantHash file=$variantArchivePath"
        }

        $variantExtractDir = Join-Path $extractDir ("llama.cpp-" + $llamaCppTag + "-" + $variantName)
        if (Test-Path -LiteralPath $variantExtractDir -PathType Container) {
            Remove-Item -LiteralPath $variantExtractDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        New-Item -ItemType Directory -Force -Path $variantExtractDir | Out-Null

        Write-Host "Extracting llama.cpp bundle ($variantName)..." -ForegroundColor Cyan
        Expand-Archive -LiteralPath $variantArchivePath -DestinationPath $variantExtractDir -Force

        $mtmdCliExe = Join-Path $variantExtractDir "llama-mtmd-cli.exe"
        if (-not (Test-Path -LiteralPath $mtmdCliExe -PathType Leaf)) {
            $available = (Get-ChildItem -Path $variantExtractDir -File -Filter "*.exe" | Select-Object -ExpandProperty Name) -join ", "
            throw "Could not locate llama-mtmd-cli.exe in the llama.cpp bundle ($variantName). Available: $available"
        }

        $llavaCliCandidates = @(
            (Join-Path $variantExtractDir "llava-cli.exe"),
            (Join-Path $variantExtractDir "llama-llava-cli.exe")
        )
        $llavaCliExe = $llavaCliCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

        $variantBinDir = Join-Path $llamaCppBinDir $variantName
        New-Item -ItemType Directory -Force -Path $variantBinDir | Out-Null

        # Bundle the preferred multimodal CLI (llama-mtmd-cli.exe).
        Copy-Item -Force -LiteralPath $mtmdCliExe -Destination (Join-Path $variantBinDir "llama-mtmd-cli.exe")

        # Bundle llava-cli.exe for backward compatibility (if present in the release).
        # Normalize to llava-cli.exe for runtime discovery.
        if ($llavaCliExe) {
            Copy-Item -Force -LiteralPath $llavaCliExe -Destination (Join-Path $variantBinDir "llava-cli.exe")
        }
        Get-ChildItem -Path $variantExtractDir -File -Filter "*.dll" | ForEach-Object {
            Copy-Item -Force -LiteralPath $_.FullName -Destination $variantBinDir
        }

        $bundledVariants += [pscustomobject]@{
            Name = $variantName
            BinDir = $variantBinDir
            MtmdCli = Join-Path $variantBinDir "llama-mtmd-cli.exe"
            LlavaCli = if ($llavaCliExe) { Join-Path $variantBinDir "llava-cli.exe" } else { "" }
        }
    }

    # Keep a root-level AVX2 copy for backward compatibility with existing run scripts.
    $avx2BinDir = Join-Path $llamaCppBinDir "avx2"
    if (Test-Path -LiteralPath $avx2BinDir -PathType Container) {
        $rootMtmdCli = Join-Path $llamaCppBinDir "llama-mtmd-cli.exe"
        Copy-Item -Force -LiteralPath (Join-Path $avx2BinDir "llama-mtmd-cli.exe") -Destination $rootMtmdCli

        $avx2LlavaCli = Join-Path $avx2BinDir "llava-cli.exe"
        if (Test-Path -LiteralPath $avx2LlavaCli -PathType Leaf) {
            Copy-Item -Force -LiteralPath $avx2LlavaCli -Destination (Join-Path $llamaCppBinDir "llava-cli.exe")
        }
        Get-ChildItem -Path $avx2BinDir -File -Filter "*.dll" | ForEach-Object {
            Copy-Item -Force -LiteralPath $_.FullName -Destination $llamaCppBinDir
        }
    }

    Write-Host "Bundled llama.cpp binaries:" -ForegroundColor Cyan
    foreach ($entry in $bundledVariants) {
        Write-Host "  $($entry.Name): $(Join-Path $entry.BinDir "llama-mtmd-cli.exe")" -ForegroundColor DarkGray
        if ($entry.LlavaCli) {
            Write-Host "    llava-cli: $($entry.LlavaCli)" -ForegroundColor DarkGray
        }
    }
    if (Test-Path -LiteralPath (Join-Path $llamaCppBinDir "llama-mtmd-cli.exe") -PathType Leaf) {
        Write-Host "  root (compat): $(Join-Path $llamaCppBinDir "llama-mtmd-cli.exe")" -ForegroundColor DarkGray
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

if (-not `$env:HF_HOME -or [string]::IsNullOrWhiteSpace(`$env:HF_HOME)) {
    `$hfHome = Join-Path `$root "data/hf"
    New-Item -ItemType Directory -Force -Path `$hfHome | Out-Null
    `$env:HF_HOME = `$hfHome
}

if (-not `$env:RAGPREP_PDF_BACKEND -or [string]::IsNullOrWhiteSpace(`$env:RAGPREP_PDF_BACKEND)) {
    `$env:RAGPREP_PDF_BACKEND = "glm-ocr"
}

if (`$env:RAGPREP_PDF_BACKEND -eq "lightonocr") {
    if (-not `$env:LIGHTONOCR_GGUF_MODEL_PATH -or [string]::IsNullOrWhiteSpace(`$env:LIGHTONOCR_GGUF_MODEL_PATH)) {
        `$env:LIGHTONOCR_GGUF_MODEL_PATH = Join-Path `$root "data/models/lightonocr-gguf/$ggufModelFileTrimmed"
    }
    if (-not `$env:LIGHTONOCR_GGUF_MMPROJ_PATH -or [string]::IsNullOrWhiteSpace(`$env:LIGHTONOCR_GGUF_MMPROJ_PATH)) {
        `$env:LIGHTONOCR_GGUF_MMPROJ_PATH = Join-Path `$root "data/models/lightonocr-gguf/$ggufMmprojFileTrimmed"
    }

    `$ggufExpectedDir = Join-Path `$root "data/models/lightonocr-gguf"
    `$requiredGguf = @(
        @{ Name = "LIGHTONOCR_GGUF_MODEL_PATH"; Path = `$env:LIGHTONOCR_GGUF_MODEL_PATH },
        @{ Name = "LIGHTONOCR_GGUF_MMPROJ_PATH"; Path = `$env:LIGHTONOCR_GGUF_MMPROJ_PATH }
    )
    foreach (`$artifact in `$requiredGguf) {
        `$artifactPath = `$artifact.Path
        if (-not (Test-Path -LiteralPath `$artifactPath -PathType Leaf)) {
            `$name = `$artifact.Name
            `$message = @(
                "[ERROR] Missing GGUF artifact: `$artifactPath",
                "Set `$name to a valid .gguf file.",
                "Expected under: `$ggufExpectedDir",
                "Rebuild standalone: scripts/build-standalone.ps1 -Clean"
            ) -join [Environment]::NewLine
            throw `$message
        }
    }
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

if "%HF_HOME%"=="" (
  set HF_HOME=%ROOT%data\hf
  if not exist "%ROOT%data\hf" mkdir "%ROOT%data\hf"
)
if "%RAGPREP_PDF_BACKEND%"=="" (
  set RAGPREP_PDF_BACKEND=glm-ocr
)
if /I "%RAGPREP_PDF_BACKEND%"=="lightonocr" (
  if "%LIGHTONOCR_GGUF_MODEL_PATH%"=="" (
    set LIGHTONOCR_GGUF_MODEL_PATH=%ROOT%data\models\lightonocr-gguf\$ggufModelFileTrimmed
  )
  if "%LIGHTONOCR_GGUF_MMPROJ_PATH%"=="" (
    set LIGHTONOCR_GGUF_MMPROJ_PATH=%ROOT%data\models\lightonocr-gguf\$ggufMmprojFileTrimmed
  )
  set "GGUF_EXPECTED_DIR=%ROOT%data\models\lightonocr-gguf"
  if not exist "%LIGHTONOCR_GGUF_MODEL_PATH%" (
    echo [ERROR] Missing GGUF model: %LIGHTONOCR_GGUF_MODEL_PATH%
    echo Set LIGHTONOCR_GGUF_MODEL_PATH to a valid .gguf file.
    echo Expected under: %GGUF_EXPECTED_DIR%
    echo Rebuild standalone: scripts\build-standalone.ps1 -Clean
    exit /b 1
  )
  if not exist "%LIGHTONOCR_GGUF_MMPROJ_PATH%" (
    echo [ERROR] Missing GGUF mmproj: %LIGHTONOCR_GGUF_MMPROJ_PATH%
    echo Set LIGHTONOCR_GGUF_MMPROJ_PATH to a valid .gguf file.
    echo Expected under: %GGUF_EXPECTED_DIR%
    echo Rebuild standalone: scripts\build-standalone.ps1 -Clean
    exit /b 1
  )
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
    & $verifyScript `
        -OutputDir $OutputDir `
        -GgufModelFile $ggufModelFileTrimmed `
        -GgufMmprojFile $ggufMmprojFileTrimmed
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

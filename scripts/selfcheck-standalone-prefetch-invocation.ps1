[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [switch]$SkipBuildScriptCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Self-check for the standalone build model prefetch invocation. No network access required.
#
# Goal: prevent regressions back to `python.exe -c <code>` (which is prone to quote/escaping issues on Windows).
#
# Run:
#   powershell -ExecutionPolicy Bypass -File scripts/selfcheck-standalone-prefetch-invocation.ps1

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    }
    else {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd -and -not [string]::IsNullOrWhiteSpace($cmd.Source)) {
            $PythonExe = $cmd.Source
        }
        else {
            throw "Python not found. Run 'uv sync --dev' to create .venv, or install Python and ensure it's on PATH."
        }
    }
}

$PythonExe = [System.IO.Path]::GetFullPath($PythonExe)
Assert-True (Test-Path -LiteralPath $PythonExe) "PythonExe not found: $PythonExe"

if (-not $SkipBuildScriptCheck) {
    $buildScript = Join-Path $repoRoot "scripts\\build-standalone.ps1"
    Assert-True (Test-Path -LiteralPath $buildScript) "build script not found: $buildScript"

    $raw = Get-Content -LiteralPath $buildScript -Raw -Encoding UTF8
    if ($raw -match '(?im)^\s*&\s*\$pythonExe\s+-c\s+\$prefetchPy\b') {
        throw "Regression detected: build-standalone.ps1 appears to run model prefetch via 'python.exe -c'. Expected temp-file execution."
    }
}

$tempRoot = Join-Path $env:TEMP ("ragprep prefetch invocation selfcheck " + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

try {
    $pyPath = Join-Path $tempRoot "prefetch_invocation_test.py"
    $py = @'
print('SINGLE_QUOTE_OK')
print("DOUBLE_QUOTE_OK")
print("EMBEDDED_QUOTE_OK: \"")
print(r"RAW_BACKSLASH_OK: \\")
'@
    Set-Content -Path $pyPath -Value $py -Encoding UTF8

    $output = & $PythonExe $pyPath 2>&1
    $exitCode = $LASTEXITCODE
    Assert-True ($exitCode -eq 0) ("Python failed with exit code ${exitCode}:`n" + ($output | Out-String))

    $actual = @($output) | ForEach-Object { $_.ToString().TrimEnd() }
    $expected = @(
        "SINGLE_QUOTE_OK",
        "DOUBLE_QUOTE_OK",
        'EMBEDDED_QUOTE_OK: "',
        'RAW_BACKSLASH_OK: \\'
    )

    for ($i = 0; $i -lt $expected.Count; $i++) {
        Assert-True ($i -lt $actual.Count) ("Missing expected output line: " + $expected[$i])
        Assert-True ($actual[$i] -eq $expected[$i]) ("Unexpected output line $i. Expected '$($expected[$i])' got '$($actual[$i])'")
    }

    Write-Host "OK: model prefetch invocation self-check passed." -ForegroundColor Green
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

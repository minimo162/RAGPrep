[CmdletBinding()]
param(
    [int]$LockMs = 1200
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Self-check for the standalone build move helper. No network access required.
#
# Run:
#   powershell -ExecutionPolicy Bypass -File scripts/selfcheck-standalone-move.ps1

. (Join-Path $PSScriptRoot "standalone-io.ps1")

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

$tempRoot = Join-Path $env:TEMP ("ragprep-standalone-move-selfcheck-" + [guid]::NewGuid().ToString("N"))
$sourceDir = Join-Path $tempRoot "src"
$destDir = Join-Path $tempRoot "dest"
$sourcePython = Join-Path $sourceDir "python.exe"
$destPython = Join-Path $destDir "python.exe"
$expectedContent = "fake-python-exe"

New-Item -ItemType Directory -Force -Path $sourceDir | Out-Null
[System.IO.File]::WriteAllText($sourcePython, $expectedContent)

New-Item -ItemType Directory -Force -Path $destDir | Out-Null
$lockFile = Join-Path $destDir "lock.txt"
[System.IO.File]::WriteAllText($lockFile, "lock")
$readyFile = Join-Path $tempRoot "lock-held.ready"
$lockHolderScript = Join-Path $tempRoot "lock-holder.ps1"
$lockHolder = @"
param(
    [Parameter(Mandatory = `$true)][string]`$LockFile,
    [Parameter(Mandatory = `$true)][int]`$LockMs,
    [Parameter(Mandatory = `$true)][string]`$ReadyFile
)

Set-StrictMode -Version Latest
`$ErrorActionPreference = "Stop"
`$ProgressPreference = "SilentlyContinue"

`$fs = [System.IO.File]::Open(
    `$LockFile,
    [System.IO.FileMode]::Open,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None
)
try {
    [System.IO.File]::WriteAllText(`$ReadyFile, "ready")
    Start-Sleep -Milliseconds `$LockMs
}
finally {
    `$fs.Dispose()
}
"@
[System.IO.File]::WriteAllText($lockHolderScript, $lockHolder, [System.Text.Encoding]::UTF8)

try {
    $locker = Start-Process -FilePath powershell -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $lockHolderScript,
        "-LockFile",
        $lockFile,
        "-LockMs",
        $LockMs,
        "-ReadyFile",
        $readyFile
    ) -PassThru -WindowStyle Hidden

    $readySw = [System.Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Path -LiteralPath $readyFile)) {
        if ($locker.HasExited) {
            throw "Lock-holder process exited before signaling readiness."
        }
        if ($readySw.ElapsedMilliseconds -gt 2000) {
            throw "Timed out waiting for lock-holder readiness."
        }
        Start-Sleep -Milliseconds 50
    }

    $moveSw = [System.Diagnostics.Stopwatch]::StartNew()
    $moveInfo = Move-DirectoryRobust -SourceDir $sourceDir -DestinationDir $destDir -VerifyRelativePath "python.exe"
    $moveSw.Stop()
    Assert-True ($moveSw.ElapsedMilliseconds -ge 150) "Move completed too quickly; expected at least one retry."
    Assert-True (Test-Path -LiteralPath $destPython) "Expected python.exe at destination: $destPython"
    $actualContent = [System.IO.File]::ReadAllText($destPython)
    Assert-True ($actualContent -eq $expectedContent) "Destination python.exe content mismatch."
    Assert-True (-not (Test-Path -LiteralPath $sourcePython)) "Source python.exe still exists after move."

    $suffix = if ($moveInfo.Method -eq "robocopy") { " (exit $($moveInfo.RobocopyExitCode))" } else { "" }
    Write-Host "OK: Move-DirectoryRobust succeeded via $($moveInfo.Method) after $($moveInfo.Attempts) attempt(s)$suffix" -ForegroundColor Green
}
finally {
    if ($locker -and -not $locker.HasExited) {
        try { $locker | Stop-Process -Force } catch { }
    }
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

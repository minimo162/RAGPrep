# Standalone build filesystem helpers (no side effects; dot-source from other scripts).

function Invoke-RetryIO {
    param(
        [Parameter(Mandatory = $true)][string]$Step,
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [int]$Retries = 8,
        [int]$InitialDelayMs = 200
    )

    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            & $Action
            return $attempt
        }
        catch [System.IO.IOException], [System.UnauthorizedAccessException] {
            if ($attempt -ge $Retries) {
                throw
            }

            $delayMs = [Math]::Min(5000, [int]($InitialDelayMs * [Math]::Pow(2, $attempt - 1)))
            Write-Warning "$Step failed (attempt $attempt/$Retries): $($_.Exception.Message)"
            Start-Sleep -Milliseconds $delayMs
        }
    }

    throw "$Step failed after $Retries attempts."
}

function Move-DirectoryRobust {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$SourceDir,
        [Parameter(Mandatory = $true)][string]$DestinationDir,
        [Parameter(Mandatory = $true)][string]$VerifyRelativePath,
        [int]$MoveRetries = 8
    )

    $sourceFull = [System.IO.Path]::GetFullPath($SourceDir)
    $destFull = [System.IO.Path]::GetFullPath($DestinationDir)
    $verifyPath = Join-Path $destFull $VerifyRelativePath

    if (-not (Test-Path $sourceFull)) {
        throw "Source dir not found: $sourceFull"
    }

    try {
        if (Test-Path $destFull) {
            Invoke-RetryIO -Step "Remove existing destination dir" -Retries $MoveRetries -Action {
                Remove-Item -LiteralPath $destFull -Recurse -Force
            } | Out-Null
        }

        $attempts = Invoke-RetryIO -Step "Move-Item (rename) extracted Python dir" -Retries $MoveRetries -Action {
            Move-Item -LiteralPath $sourceFull -Destination $destFull -Force
        }

        if (-not (Test-Path $verifyPath)) {
            throw "Move-Item completed but expected path missing: $verifyPath"
        }

        return [pscustomobject]@{
            Method           = "Move-Item"
            Attempts         = $attempts
            VerifiedPath     = $verifyPath
            RobocopyExitCode = $null
        }
    }
    catch [System.IO.IOException], [System.UnauthorizedAccessException] {
        $originalError = $_.Exception.Message
        Write-Warning "Move-Item failed after $MoveRetries attempt(s). Falling back to robocopy (/MOVE)."

        $robocopy = Get-Command robocopy -ErrorAction SilentlyContinue
        if (-not $robocopy) {
            throw "robocopy not found; cannot fallback move. Original error: $originalError"
        }

        $args = @(
            $sourceFull,
            $destFull,
            "/MOVE",
            "/E",
            "/R:3",
            "/W:1",
            "/NFL",
            "/NDL",
            "/NJH",
            "/NJS",
            "/NP",
            "/XJ"
        )

        & $robocopy.Source @args | Out-Null
        $rc = $LASTEXITCODE
        if ($rc -ge 8) {
            throw "robocopy failed with exit code $rc"
        }
        if (-not (Test-Path $verifyPath)) {
            throw "robocopy succeeded (exit code $rc) but expected path missing: $verifyPath"
        }

        return [pscustomobject]@{
            Method           = "robocopy"
            Attempts         = $MoveRetries
            VerifiedPath     = $verifyPath
            RobocopyExitCode = $rc
        }
    }
}


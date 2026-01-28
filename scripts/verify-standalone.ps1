[CmdletBinding()]
param(
    [string]$OutputDir = "dist/standalone",
    [string]$GgufModelFile = "LightOnOCR-2-1B-Q6_K.gguf",
    [string]$GgufMmprojFile = "mmproj-BF16.gguf",
    [string]$ServerUrl = "http://127.0.0.1:8080",
    [switch]$AutoPort
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

function New-LogDirectory {
    param([string]$BaseDir)
    $logDir = Join-Path $BaseDir "logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    return $logDir
}

function Read-LogTail {
    param([string]$Path, [int]$Lines = 80)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    try {
        $content = Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction SilentlyContinue
        if ($content) {
            return ($content -join [Environment]::NewLine)
        }
    }
    catch {
    }
    return $null
}

function Get-PortStatus {
    param([int]$Port)
    try {
        $cmd = Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue
        if ($cmd) {
            $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
            if ($connections) {
                return "in use"
            }
        }
    }
    catch {
    }
    return "not listening"
}

function Get-FreeTcpPort {
    param([string]$Host = "127.0.0.1")
    $ip = [System.Net.IPAddress]::Loopback
    try {
        $parsed = [System.Net.IPAddress]::Parse($Host)
        if ($parsed) {
            $ip = $parsed
        }
    }
    catch {
    }
    $listener = [System.Net.Sockets.TcpListener]::new($ip, 0)
    $listener.Start()
    $port = $listener.LocalEndpoint.Port
    $listener.Stop()
    return $port
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

$serverUrl = $ServerUrl
$serverExe = Resolve-LlamaServerExe `
    -RootExe $llamaServerExe `
    -Avx2Exe $llamaServerAvx2Exe `
    -VulkanExe $llamaServerVulkanExe

$serverProcess = $null
$startedServer = $false
$logDir = $null
$stdoutLog = $null
$stderrLog = $null
$metaLog = $null
$effectiveServerUrl = $serverUrl
$effectiveServerUri = $null
try {
    if (-not (Test-LlamaServer -BaseUrl $serverUrl)) {
        $serverUri = [Uri]$serverUrl
        $portStatus = Get-PortStatus -Port $serverUri.Port
        if ($AutoPort -or $portStatus -eq "in use") {
            $freePort = Get-FreeTcpPort -Host $serverUri.Host
            $effectiveServerUrl = "http://$($serverUri.Host):$freePort"
        }
        $effectiveServerUri = [Uri]$effectiveServerUrl
        $serverArgs = @(
            "-m", $ggufModelPath,
            "--mmproj", $ggufMmprojPath,
            "--host", $effectiveServerUri.Host,
            "--port", $effectiveServerUri.Port
        )
        $logDir = New-LogDirectory -BaseDir $resolvedOutputDir
        $stdoutLog = Join-Path $logDir "llama-server.stdout.log"
        $stderrLog = Join-Path $logDir "llama-server.stderr.log"
        $metaLog = Join-Path $logDir "llama-server.meta.log"
        Remove-Item -LiteralPath $stdoutLog, $stderrLog, $metaLog -ErrorAction SilentlyContinue
        @(
            "start: $(Get-Date -Format o)"
            "serverExe: $serverExe"
            "serverUrl: $serverUrl"
            "effectiveServerUrl: $effectiveServerUrl"
            "autoPort: $AutoPort"
            "args: $($serverArgs -join ' ')"
            "ggufModel: $ggufModelPath"
            "ggufMmproj: $ggufMmprojPath"
        ) | Set-Content -LiteralPath $metaLog -Encoding UTF8

        $serverProcess = Start-Process `
            -FilePath $serverExe `
            -ArgumentList $serverArgs `
            -PassThru `
            -WindowStyle Minimized `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog
        $startedServer = $true

        $ready = $false
        $exitedEarly = $false
        $exitCode = $null
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Milliseconds 500
            if ($serverProcess -and $serverProcess.HasExited) {
                $exitedEarly = $true
                $exitCode = $serverProcess.ExitCode
                break
            }
            if (Test-LlamaServer -BaseUrl $effectiveServerUrl) {
                $ready = $true
                break
            }
        }
        if (-not $ready) {
            $stderrTail = Read-LogTail -Path $stderrLog -Lines 80
            $stdoutTail = if (-not $stderrTail) { Read-LogTail -Path $stdoutLog -Lines 80 } else { $null }
            $detailLines = @(
                "llama-server failed to start: $effectiveServerUrl",
                "requested url: $serverUrl",
                "server exe: $serverExe",
                "exit: " + ($(if ($exitedEarly) { "exited (ExitCode=$exitCode)" } else { "not exited within wait window" })),
                "port: $($effectiveServerUri.Port) ($portStatus)",
                "stdout log: $stdoutLog",
                "stderr log: $stderrLog",
                "meta log: $metaLog"
            )
            if ($stderrTail) {
                $detailLines += ""
                $detailLines += "stderr tail:"
                $detailLines += $stderrTail
            }
            elseif ($stdoutTail) {
                $detailLines += ""
                $detailLines += "stdout tail:"
                $detailLines += $stdoutTail
            }
            $detailLines += ""
            $detailLines += "Hint: check GPU/Vulkan availability, model load time, and port conflicts."
            throw ($detailLines -join [Environment]::NewLine)
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

param(
    [switch]$StopOllama
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
. "$PSScriptRoot\install-utils.ps1"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-ApiPort {
    $port = [Environment]::GetEnvironmentVariable("API_PORT")
    if ($port) { return [int]$port }
    if (-not (Test-Path -LiteralPath ".\.env")) { return 8000 }
    foreach ($line in Get-Content -LiteralPath ".\.env") {
        if ($line.Trim() -match "^API_PORT\s*=\s*(.+)$") {
            return [int]$Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return 8000
}

function Test-Health {
    param([int]$Port)
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Stop-PidFile {
    param([string]$Path, [string]$Label, [string]$CommandLineLike)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $rawPid = "$(Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue)".Trim()
    if (Stop-RecordedProcess -PidFile $Path -CommandLineLike $CommandLineLike) {
        Write-Step "Stopped $Label (PID: $rawPid)"
    }
    else {
        Write-Step "Removed a stale $Label PID file (PID $rawPid no longer runs it)"
    }
}

function Get-PythonAncestorIds {
    # Some Windows Python setups (notably a venv created from a Conda base
    # interpreter) transparently relaunch a script under a *different*
    # interpreter as a child process - the venv's own python.exe stays alive
    # as an orphaned parent while its child actually binds the port. Walking
    # up while every ancestor is still a python-family process catches both,
    # instead of leaving the parent running after the child is killed.
    param([int]$ProcessId)
    $ids = [System.Collections.Generic.List[int]]::new()
    $current = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    while ($current) {
        $ids.Add([int]$current.ProcessId)
        if (-not $current.ParentProcessId) { break }
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($current.ParentProcessId)" -ErrorAction SilentlyContinue
        if (-not $parent -or $parent.Name -notmatch "^(python|python3|pythonw|py)(\.exe)?$") { break }
        $current = $parent
    }
    return $ids
}

function Stop-ApiListener {
    param([int]$Port)
    if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) { return }
    if (-not (Get-Command Get-CimInstance -ErrorAction SilentlyContinue)) { return }

    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction SilentlyContinue
        # Match by process family + script name, not a full path substring: a
        # relaunched child (see Get-PythonAncestorIds) can carry a *relative*
        # "api_server.py" argument, so the project's absolute path never
        # appears verbatim in its own command line.
        if (-not $proc -or $proc.Name -notmatch "^(python|python3|pythonw|py)(\.exe)?$" -or $proc.CommandLine -notlike "*api_server.py*") {
            continue
        }
        $killIds = Get-PythonAncestorIds -ProcessId $proc.ProcessId
        foreach ($killId in $killIds) {
            Stop-Process -Id $killId -Force -ErrorAction SilentlyContinue
        }
        Write-Step "Stopped API server listener on port $Port (PID $($killIds -join ', '))"
    }
}

$apiPort = Get-ApiPort

Stop-PidFile -Path ".\logs\api_server.pid" -Label "LocalDeploy API" -CommandLineLike "*api_server.py*"
Stop-ApiListener -Port $apiPort

if (Test-Health -Port $apiPort) {
    Write-Warning "LocalDeploy API still responds on port $apiPort. Stop it manually if this checkout owns it."
}
else {
    Write-Host "LocalDeploy API stopped successfully" -ForegroundColor Green
}

Stop-PidFile -Path ".\logs\llama_server.pid" -Label "llama.cpp server" -CommandLineLike "*llama-server*"
Get-Process "llama-server" -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    Write-Step "Stopped llama.cpp server (PID: $($_.Id))"
}

if ($StopOllama) {
    Get-Process "ollama" -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        Write-Step "Stopped Ollama (PID: $($_.Id))"
    }
}
else {
    Write-Host "Ollama was left running. To stop it too: .\scripts\stop.ps1 -StopOllama" -ForegroundColor Cyan
}

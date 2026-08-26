# Starts LocalDeploy: creates .env/config.json/.venv when missing, starts
# Ollama if installed, launches the API in the background, and opens the UI.
#   -Foreground   run the API in this terminal with live logs (no browser)
#   -NoBrowser    start in the background without opening the UI
#   -Restart      stop any running LocalDeploy API first, then start fresh
param(
    [switch]$Foreground,
    [switch]$NoBrowser,
    [switch]$SkipInstall,
    [switch]$SkipLlamaCpp,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
. "$PSScriptRoot\install-utils.ps1"
Initialize-Install -RepositoryRoot $ProjectRoot -ProductName "LocalDeploy"
trap { Write-InstallFailure $_; Exit-InstallLock; exit 1 }
Enter-InstallLock
Assert-InstallFreeSpace -Path $ProjectRoot -RequiredGB 3

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Read-DotEnv {
    $values = @{}
    if (-not (Test-Path -LiteralPath ".\.env")) { return $values }
    foreach ($line in Get-Content -LiteralPath ".\.env") {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) { continue }
        $name, $value = $trimmed.Split("=", 2)
        $values[$name.Trim()] = $value.Trim().Trim('"').Trim("'")
    }
    return $values
}

function Env-Value {
    param([hashtable]$EnvFile, [string]$Name, [string]$Default = $null)
    $processValue = [Environment]::GetEnvironmentVariable($Name)
    if ($processValue) { return $processValue }
    if ($EnvFile.ContainsKey($Name)) { return $EnvFile[$Name] }
    return $Default
}

function Get-ApiBaseUrl {
    param([hashtable]$EnvFile)
    $hostName = Env-Value -EnvFile $EnvFile -Name "API_HOST" -Default "127.0.0.1"
    $port = Env-Value -EnvFile $EnvFile -Name "API_PORT" -Default "8000"
    if ($hostName -eq "0.0.0.0" -or $hostName -eq "::") { $hostName = "127.0.0.1" }
    elseif ($hostName -eq "::1") { $hostName = "[::1]" }
    return "http://${hostName}:${port}"
}

function Test-Http {
    param([string]$Url, [int]$Timeout = 5)
    try {
        Invoke-RestMethod -Uri $Url -TimeoutSec $Timeout | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Find-Ollama {
    $cmd = Get-Command "ollama" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidate = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    return $null
}

function Find-Python {
    foreach ($cmd in @("python", "python3", "py")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) {
            try {
                # Run in a child process so the Windows Store stub's stderr
                # does not trigger $ErrorActionPreference = "Stop"
                $ver = & $cmd --version 2>&1
                if ("$ver" -match "Python \d") {
                    return $cmd
                }
            }
            catch {
                # Store stub or broken install - skip this candidate
            }
        }
    }
    return $null
}

function Install-PythonGuide {
    Write-Host ""
    Write-Host "Python is required but was not found on this system." -ForegroundColor Yellow
    Write-Host ""

    $hasWinget = Get-Command winget -ErrorAction SilentlyContinue
    if ($hasWinget) {
        Write-Host "Choose an installation method:" -ForegroundColor Cyan
        Write-Host "  [1] Install automatically via winget (recommended)"
        Write-Host "  [2] Open python.org download page"
        Write-Host "  [Q] Quit and install manually"
        Write-Host ""
        $choice = Read-Host "Your choice"
    }
    else {
        Write-Host "winget not available. Choose how to install Python:" -ForegroundColor Cyan
        Write-Host "  [1] Open python.org download page"
        Write-Host "  [Q] Quit"
        Write-Host ""
        $choice = Read-Host "Your choice"
        # remap so option 1 opens the browser
        if ($choice.Trim() -eq "1") { $choice = "2" }
    }

    switch ($choice.Trim().ToUpper()) {
        "1" {
            Write-Step "Installing Python 3.12 via winget..."
            # Pipe to Out-Host so winget output goes to the console only and does
            # not get captured as part of this function's return value.
            Invoke-InstallRetry "Python installation" {
                $output = winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements 2>&1
                $code = $LASTEXITCODE
                $output | Out-Host
                if ($code -ne 0) { throw "winget Python install failed with exit $($code): $($output -join [Environment]::NewLine)" }
            }
            # Refresh PATH so the new Python binary is visible in this session
            $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                        [System.Environment]::GetEnvironmentVariable("PATH", "User")
            $pyCmd = Find-Python
            if (-not $pyCmd) {
                Write-Host ""
                Write-Host "Python was installed but is not yet visible in PATH." -ForegroundColor Yellow
                Write-Host "Please close this terminal and re-run start.ps1." -ForegroundColor Yellow
                exit 0
            }
            return $pyCmd
        }
        "2" {
            Start-Process "https://www.python.org/downloads/"
            Write-Host ""
            Write-Host "Opening python.org in your browser." -ForegroundColor Cyan
            Write-Host "After installing Python, re-run this script." -ForegroundColor Yellow
            exit 0
        }
        default {
            Write-Host ""
            Write-Host "Exiting. Install Python from https://www.python.org/downloads/ then re-run this script." -ForegroundColor Yellow
            exit 0
        }
    }
}

function Install-OllamaGuide {
    Write-Host ""
    Write-Host "Ollama is required to pull and serve models, but was not found on this system." -ForegroundColor Yellow
    Write-Host ""

    $hasWinget = Get-Command winget -ErrorAction SilentlyContinue
    if ($hasWinget) {
        Write-Host "Choose an installation method:" -ForegroundColor Cyan
        Write-Host "  [1] Install automatically via winget (recommended)"
        Write-Host "  [2] Open ollama.com download page"
        Write-Host "  [Q] Skip for now (you won't be able to pull or serve models)"
        Write-Host ""
        $choice = Read-Host "Your choice"
    }
    else {
        Write-Host "winget not available. Choose how to install Ollama:" -ForegroundColor Cyan
        Write-Host "  [1] Open ollama.com download page"
        Write-Host "  [Q] Skip for now"
        Write-Host ""
        $choice = Read-Host "Your choice"
        # remap so option 1 opens the browser
        if ($choice.Trim() -eq "1") { $choice = "2" }
    }

    switch ($choice.Trim().ToUpper()) {
        "1" {
            Write-Step "Installing Ollama via winget..."
            Invoke-InstallRetry "Ollama installation" {
                $output = winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements 2>&1
                $code = $LASTEXITCODE
                $output | Out-Host
                if ($code -ne 0) { throw "winget Ollama install failed with exit $($code): $($output -join [Environment]::NewLine)" }
            }
            # Refresh PATH so the new ollama binary is visible in this session
            $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                        [System.Environment]::GetEnvironmentVariable("PATH", "User")
            $found = Find-Ollama
            if (-not $found) {
                Write-Host ""
                Write-Host "Ollama was installed but is not yet visible in PATH." -ForegroundColor Yellow
                Write-Host "Close this terminal and re-run start.ps1 to pick it up." -ForegroundColor Yellow
                return $null
            }
            return $found
        }
        "2" {
            Start-Process "https://ollama.com/download"
            Write-Host ""
            Write-Host "Opening ollama.com in your browser." -ForegroundColor Cyan
            Write-Host "After installing Ollama, re-run this script." -ForegroundColor Yellow
            return $null
        }
        default {
            Write-Host ""
            Write-Host "Skipping Ollama install. Pulling and serving models will fail until it's installed." -ForegroundColor Yellow
            return $null
        }
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

function Clear-ZombieOnPort {
    param([string]$Url)
    if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) { return }
    $port = ([System.Uri]$Url).Port
    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) { return }
    Write-Warning "Port $port is held by PID $($listener.OwningProcess) but not answering HTTP. Stopping it before startup."
    if (Get-Command Get-CimInstance -ErrorAction SilentlyContinue) {
        foreach ($killId in (Get-PythonAncestorIds -ProcessId $listener.OwningProcess)) {
            Stop-Process -Id $killId -Force -ErrorAction SilentlyContinue
        }
    }
    else {
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 600
}

if (-not (Test-Path -LiteralPath ".\.env")) {
    Copy-Item -LiteralPath ".\.env.example" -Destination ".\.env"
    Write-Step "Created .env"
}

if (-not (Test-Path -LiteralPath ".\config.json")) {
    # Seed a MINIMAL config.json rather than copying the example's dozen sample
    # profiles. config.json is a live mirror of what the user actually pulls:
    # pulling a model auto-creates its profile, so a fresh install should start
    # empty instead of listing models the user hasn't downloaded.
    $minimalConfig = @'
{
  "version": 1,
  "default_profile": null,
  "global_defaults": {
    "safe_mode": true,
    "stream": false,
    "require_gpu_only": false
  },
  "profiles": {}
}
'@
    Set-Content -LiteralPath ".\config.json" -Value $minimalConfig -Encoding utf8
    Write-Step "Created config.json (empty - profiles are created as you pull models)"
}

$envFile = Read-DotEnv
$apiBaseUrl = Get-ApiBaseUrl -EnvFile $envFile
$healthUrl = "$apiBaseUrl/health"
$uiUrl = "$apiBaseUrl/ui"
$docsUrl = "$apiBaseUrl/docs"

function Install-Requirements {
    param([string]$PythonExe)
    Write-Step "Installing Python dependencies (this can take a minute on first run)"
    Invoke-InstallRetry "pip upgrade" {
        $output = & $PythonExe -m pip install --upgrade pip --quiet 2>&1
        $code = $LASTEXITCODE
        $output | Out-Host
        if ($code -ne 0) { throw "pip upgrade failed with exit $($code): $($output -join [Environment]::NewLine)" }
    }
    Invoke-InstallRetry "dependency installation" {
        $output = & $PythonExe -m pip install --quiet -r requirements.txt 2>&1
        $code = $LASTEXITCODE
        $output | Out-Host
        if ($code -ne 0) { throw "pip install failed with exit $($code): $($output -join [Environment]::NewLine)" }
    }
    (Get-FileHash -LiteralPath ".\requirements.txt" -Algorithm SHA256).Hash |
        Set-Content -LiteralPath ".\.venv\requirements.sha256"
}

function Test-RequirementsCurrent {
    # True when the venv was installed from the requirements.txt we have now.
    # Catches the "git pull added a dependency" case that used to end in a raw
    # ImportError, because deps only installed when .venv was missing.
    $marker = ".\.venv\requirements.sha256"
    if (-not (Test-Path -LiteralPath $marker)) { return $false }
    $current = (Get-FileHash -LiteralPath ".\requirements.txt" -Algorithm SHA256).Hash
    return (Get-Content -LiteralPath $marker -ErrorAction SilentlyContinue) -eq $current
}

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    if ($SkipInstall) {
        $python = "python"
    }
    else {
        $pyCmd = Find-Python
        if (-not $pyCmd) {
            $pyCmd = Install-PythonGuide
        }
        Write-Step "Creating Python virtual environment"
        & $pyCmd -m venv .venv
        if (-not (Test-Path -LiteralPath $python)) {
            throw "Virtual environment creation failed. Verify your Python installation and try again."
        }
        Install-Requirements -PythonExe $python
    }
}
elseif (-not $SkipInstall -and -not (Test-RequirementsCurrent)) {
    Install-Requirements -PythonExe $python
}

$ollama = Find-Ollama
if (-not $ollama -and -not $SkipInstall) {
    $ollama = Install-OllamaGuide
}
if (-not $ollama) {
    Write-Warning "Ollama was not found. Install it from https://ollama.com/download (or: winget install Ollama.Ollama), then re-run this script."
}
elseif (-not (Test-Http "http://localhost:11434/api/tags" -Timeout 5)) {
    Write-Step "Starting Ollama"
    # Windows PowerShell 5 has no Start-Process -Environment parameter. Set the
    # process value briefly so the child receives the .env setting, then restore
    # the launcher's original environment.
    $previousOllamaNoCloud = [Environment]::GetEnvironmentVariable("OLLAMA_NO_CLOUD")
    $env:OLLAMA_NO_CLOUD = Env-Value -EnvFile $envFile -Name "OLLAMA_NO_CLOUD" -Default "true"
    try {
        Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    }
    finally {
        if ($null -eq $previousOllamaNoCloud) {
            Remove-Item -LiteralPath Env:\OLLAMA_NO_CLOUD -ErrorAction SilentlyContinue
        }
        else {
            $env:OLLAMA_NO_CLOUD = $previousOllamaNoCloud
        }
    }
    # Poll instead of a fixed sleep: first launch can take a while.
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 750
        if (Test-Http "http://localhost:11434/api/tags" -Timeout 2) { break }
    }
    if (-not (Test-Http "http://localhost:11434/api/tags" -Timeout 2)) {
        Write-Warning "Ollama has not answered yet. The UI will show its status; pulls will work once it's up."
    }
}

if (-not $SkipLlamaCpp) {
    & "$PSScriptRoot\start_llamacpp.ps1" -Optional
}
Complete-Install

function Stop-RunningApi {
    # Prefer the PID file the background launcher writes; fall back to whoever
    # holds the port.
    $pidFile = ".\logs\api_server.pid"
    if (Test-Path -LiteralPath $pidFile) {
        $apiPid = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue
        if ($apiPid) { Stop-Process -Id $apiPid -Force -ErrorAction SilentlyContinue }
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }
    Clear-ZombieOnPort $healthUrl
    Start-Sleep -Milliseconds 400
}

$alreadyRunning = Test-Http $healthUrl -Timeout 5
if ($alreadyRunning -and $Restart) {
    Write-Step "Restarting LocalDeploy API (-Restart)"
    Stop-RunningApi
    $alreadyRunning = $false
}
elseif ($alreadyRunning -and
        ((Env-Value -EnvFile $envFile -Name "ENABLE_WEB_UI" -Default "true") -ne "false") -and
        -not (Test-Http $uiUrl -Timeout 5)) {
    # /health answers but /ui errors: a server from an older checkout is still
    # running (e.g. after a git pull moved files). Heal it instead of opening a
    # broken page and telling the user everything is fine.
    Write-Warning "A LocalDeploy API is running but its UI is broken (stale process from an older version). Restarting it."
    Stop-RunningApi
    $alreadyRunning = $false
}

if ($alreadyRunning) {
    Write-Step "LocalDeploy API is already running (use -Restart to force a fresh start)"
}
elseif ($Foreground) {
    Write-Step "Starting LocalDeploy API in the foreground"
    Write-Host "Stop with Ctrl+C. Open another terminal for chat commands."
    & $python api_server.py
    exit $LASTEXITCODE
}
else {
    Write-Step "Starting LocalDeploy API in the background"
    Clear-ZombieOnPort $healthUrl
    New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null
    $out = Join-Path (Resolve-Path ".\logs") "api_server.out.log"
    $err = Join-Path (Resolve-Path ".\logs") "api_server.err.log"
    $pythonAbs = if (Test-Path -LiteralPath $python) { (Resolve-Path -LiteralPath $python).Path } else { $python }
    $process = Start-Process `
        -FilePath $pythonAbs `
        -ArgumentList "api_server.py" `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput $out `
        -RedirectStandardError $err
    $process.Id | Set-Content -LiteralPath ".\logs\api_server.pid"

    Write-Host "Waiting for server to start" -NoNewline
    $serverReady = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        Write-Host "." -NoNewline
        if (Test-Http $healthUrl -Timeout 30) {
            $serverReady = $true
            break
        }
    }
    Write-Host ""

    if (-not $serverReady) {
        throw "LocalDeploy API did not start at $apiBaseUrl. Check logs\api_server.err.log."
    }
}

Write-Host ""
Write-Host "LocalDeploy is ready:" -ForegroundColor Green
Write-Host "  UI:      $uiUrl"
Write-Host "  API:     $apiBaseUrl"
Write-Host "  Docs:    $docsUrl"
Write-Host "  Stop:    .\scripts\stop.ps1"
Write-Host "  Chat:    .\scripts\chat.ps1 -Prompt `"How are you?`""
Write-Host "  Logs:    Get-Content .\logs\api_server.err.log -Wait"

if (-not $NoBrowser) {
    # Opening the browser is a convenience, not part of a successful start. On a
    # machine with no default browser association this throws, and because
    # $ErrorActionPreference is "Stop" that would make start.ps1 exit non-zero -
    # so start.bat would print "LocalDeploy did not start" even though it is
    # running fine. Never let the browser step fail the launch.
    try {
        Start-Process $uiUrl -ErrorAction Stop
    }
    catch {
        Write-Host "Could not open a browser automatically. Open $uiUrl yourself." -ForegroundColor Yellow
    }
}

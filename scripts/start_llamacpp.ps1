param(
    [switch]$SkipIfDisabled,
    [switch]$Optional
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Stop-LlamaStartup {
    param([string]$Message)
    if ($SkipIfDisabled -or $Optional) {
        Write-Warning "$Message Skipping optional llama.cpp startup."
        return $true
    }
    throw $Message
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

function Env-Bool {
    param([hashtable]$EnvFile, [string]$Name, [bool]$Default = $false)
    $value = Env-Value -EnvFile $EnvFile -Name $Name -Default $null
    if ($null -eq $value -or $value -eq "") { return $Default }
    return @("1", "true", "yes", "on") -contains $value.ToString().ToLowerInvariant()
}

function Assert-GpuOnlyProfile {
    param(
        [hashtable]$EnvFile,
        [object]$Profile
    )
    $gpuOnly = Env-Bool -EnvFile $EnvFile -Name "REQUIRE_GPU_ONLY" -Default $false
    if (-not $gpuOnly) {
        return
    }
    $gpuLayers = if ($null -ne $Profile.gpu_layers) { $Profile.gpu_layers.ToString().ToLowerInvariant() } else { "" }
    if ($Profile.backend -ne "llamacpp" -or $gpuLayers -ne "all") {
        throw "GPU-only mode requires an enabled llama.cpp profile with gpu_layers=all. Selected profile '$($Profile.name)' has backend=$($Profile.backend), gpu_layers=$($Profile.gpu_layers)."
    }
}

function Test-LlamaCppReady {
    param([string]$ModelsUrl)
    try {
        $result = Invoke-RestMethod -Uri $ModelsUrl -TimeoutSec 5
        return $result.data -and $result.data.Count -gt 0
    }
    catch {
        return $false
    }
}

function Find-LlamaServer {
    param([hashtable]$EnvFile)
    $configured = Env-Value -EnvFile $EnvFile -Name "LLAMACPP_SERVER_PATH"
    if ($configured -and (Test-Path -LiteralPath $configured)) {
        return (Resolve-Path -LiteralPath $configured).Path
    }
    $cmd = Get-Command "llama-server" -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    $wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path -LiteralPath $wingetRoot) {
        $candidate = Get-ChildItem -LiteralPath $wingetRoot -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
    }
    return $null
}

function Get-Config {
    param([hashtable]$EnvFile)
    $configPath = Env-Value -EnvFile $EnvFile -Name "CONFIG_PATH" -Default "config.json"
    if (-not [System.IO.Path]::IsPathRooted($configPath)) {
        $configPath = Join-Path $ProjectRoot $configPath
    }
    if (-not (Test-Path -LiteralPath $configPath)) {
        throw "Config file not found: $configPath"
    }
    return Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
}

function Resolve-LlamaProfile {
    param(
        [object]$Config,
        [hashtable]$EnvFile
    )
    $defaultProfile = Env-Value -EnvFile $EnvFile -Name "DEFAULT_MODEL_PROFILE" -Default $Config.default_profile
    if ($defaultProfile -and $Config.profiles.PSObject.Properties.Name -contains $defaultProfile) {
        $profile = $Config.profiles.$defaultProfile
        if ($profile.backend -eq "llamacpp" -and $profile.enabled) {
            return $profile
        }
    }
    foreach ($property in $Config.profiles.PSObject.Properties) {
        $profile = $property.Value
        if ($profile.backend -eq "llamacpp" -and $profile.enabled) {
            return $profile
        }
    }
    return $null
}

function Add-ArgIfPresent {
    param(
        [System.Collections.Generic.List[string]]$ArgumentList,
        [string]$Name,
        [object]$Value
    )
    if ($null -ne $Value -and $Value.ToString() -ne "") {
        $ArgumentList.Add($Name)
        $ArgumentList.Add($Value.ToString())
    }
}

$envFile = Read-DotEnv
if (-not (Env-Bool -EnvFile $envFile -Name "ENABLE_LLAMA_CPP" -Default $false)) {
    if (Stop-LlamaStartup "llama.cpp is disabled. Set ENABLE_LLAMA_CPP=true in .env.") { return }
}

$baseUrl = Env-Value -EnvFile $envFile -Name "LLAMACPP_BASE_URL" -Default "http://localhost:8080"
$modelsUrl = "$($baseUrl.TrimEnd('/'))/v1/models"
$config = Get-Config -EnvFile $envFile
$profile = Resolve-LlamaProfile -Config $config -EnvFile $envFile
if (-not $profile) {
    if (Stop-LlamaStartup "No enabled llama.cpp profile found in config.json.") { return }
}
try {
    Assert-GpuOnlyProfile -EnvFile $envFile -Profile $profile
}
catch {
    if (Stop-LlamaStartup $_.Exception.Message) { return }
}

if (Test-LlamaCppReady $modelsUrl) {
    Write-Step "llama.cpp server is already running at $baseUrl"
    return
}

$serverPath = Find-LlamaServer -EnvFile $envFile
if (-not $serverPath) {
    if (Stop-LlamaStartup "llama-server.exe was not found. Install llama.cpp or set LLAMACPP_SERVER_PATH in .env.") { return }
}

$modelPath = $profile.model_id
if (-not $modelPath -or -not (Test-Path -LiteralPath $modelPath)) {
    if (Stop-LlamaStartup "llama.cpp model file not found: $modelPath") { return }
}

$uri = [Uri]$baseUrl
$hostName = if ($uri.Host) { $uri.Host } else { "127.0.0.1" }
$port = if ($uri.Port -gt 0) { $uri.Port } else { 8080 }

New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null
$out = Join-Path (Resolve-Path ".\logs") "llama_server.out.log"
$err = Join-Path (Resolve-Path ".\logs") "llama_server.err.log"
$pidFile = Join-Path (Resolve-Path ".\logs") "llama_server.pid"

$args = [System.Collections.Generic.List[string]]::new()
Add-ArgIfPresent -ArgumentList $args -Name "-m" -Value $modelPath
Add-ArgIfPresent -ArgumentList $args -Name "--host" -Value $hostName
Add-ArgIfPresent -ArgumentList $args -Name "--port" -Value $port
Add-ArgIfPresent -ArgumentList $args -Name "-c" -Value $profile.context_limit
Add-ArgIfPresent -ArgumentList $args -Name "-ngl" -Value $profile.gpu_layers
if ($profile.flash_attention -eq $true) {
    $args.Add("-fa")
    $args.Add("on")
}
Add-ArgIfPresent -ArgumentList $args -Name "-ctk" -Value $profile.kv_cache_type_k
Add-ArgIfPresent -ArgumentList $args -Name "-ctv" -Value $profile.kv_cache_type_v
Add-ArgIfPresent -ArgumentList $args -Name "-b" -Value $profile.batch_size
Add-ArgIfPresent -ArgumentList $args -Name "-ub" -Value $profile.ubatch_size
Add-ArgIfPresent -ArgumentList $args -Name "-t" -Value $profile.threads
if ($profile.mmap -eq $false) {
    $args.Add("--no-mmap")
}
if ($profile.mlock -eq $true) {
    $args.Add("--mlock")
}
# Flags that have no field of their own, one array item per argument: "--reasoning", "off" for a thinking model,
# "--n-cpu-moe", "30" to keep a mixture-of-experts model's experts in RAM, "-fitt", "1024" to leave VRAM free.
foreach ($extra in @($profile.server_args)) {
    if ($null -ne $extra -and "$extra" -ne "") { $args.Add([string]$extra) }
}

Write-Step "Starting llama.cpp server using $modelPath"
$process = Start-Process `
    -FilePath $serverPath `
    -ArgumentList $args `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err `
    -PassThru
$process.Id | Set-Content -LiteralPath $pidFile

$startupTimeout = [int](Env-Value -EnvFile $envFile -Name "LLAMACPP_STARTUP_TIMEOUT_SECONDS" -Default "180")
for ($i = 0; $i -lt $startupTimeout; $i++) {
    Start-Sleep -Seconds 1
    if (Test-LlamaCppReady $modelsUrl) {
        Write-Step "llama.cpp server is ready at $baseUrl"
        return
    }
    if ($process.HasExited) {
        $stderr = ""
        if (Test-Path -LiteralPath $err) {
            $stderr = (Get-Content -LiteralPath $err -Tail 40) -join "`n"
        }
        if (Stop-LlamaStartup "llama.cpp server exited during startup. $stderr") { return }
    }
    $process.Refresh()
}

if (Stop-LlamaStartup "llama.cpp server did not become ready at $baseUrl within $startupTimeout seconds.") { return }

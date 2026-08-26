param(
  [ValidateSet("run", "doctor", "repair", "docker", "stop", "logs")]
  [string]$Action = "run",
  [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\scripts\install-utils.ps1
Initialize-Install -RepositoryRoot $PSScriptRoot -ProductName "LocalDeploy"
trap { Write-InstallFailure $_; Exit-InstallLock; exit 1 }
$port = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$url = "http://127.0.0.1:$port"

function Test-Ready {
  try { Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}
function Wait-Ready {
  for ($i = 0; $i -lt 120; $i++) { if (Test-Ready) { return $true }; Start-Sleep -Milliseconds 500 }
  return $false
}

function Test-DockerReady {
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { return $false }
  docker info *> $null
  return $LASTEXITCODE -eq 0
}
function Test-DockerService {
  if (-not (Test-DockerReady)) { return $false }
  $ids = docker compose ps --quiet 2>$null
  return -not [string]::IsNullOrWhiteSpace(($ids -join ""))
}

if ($Action -eq "stop") {
  if (Test-Path .\logs\api_server.pid) { & .\scripts\stop.ps1; exit $LASTEXITCODE }
  if (Test-DockerService) { docker compose down; exit $LASTEXITCODE }
  & .\scripts\stop.ps1
  exit $LASTEXITCODE
}
if ($Action -eq "logs") {
  if (Test-DockerService) { docker compose logs --follow; exit $LASTEXITCODE }
  $nativeLogs = @(Get-ChildItem -LiteralPath .\logs -Filter *.log -File -ErrorAction SilentlyContinue)
  if ($nativeLogs.Count -eq 0) {
    Write-Host "No LocalDeploy logs exist yet. Start it with .\run.bat."
    exit 0
  }
  Get-Content -LiteralPath $nativeLogs.FullName -Tail 100 -Wait
  exit 0
}
if ($Action -eq "docker") {
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "Docker is not installed." }
  if (-not (Test-DockerReady)) { throw "Docker is installed but its engine is not running." }
  Enter-InstallLock
  Assert-InstallFreeSpace -Path $PSScriptRoot -RequiredGB 3
  docker compose up --detach --build
  if ($LASTEXITCODE -ne 0) { throw "Docker Compose build or startup failed." }
  if (-not (Wait-Ready)) { docker compose logs; throw "LocalDeploy did not become ready at $url." }
  Complete-Install
  Write-Host "LocalDeploy is ready at $url/ui" -ForegroundColor Green
  if (-not $NoBrowser) { Start-Process "$url/ui" }
  exit 0
}

if ($Action -eq "doctor") {
  $python = ".\.venv\Scripts\python.exe"
  $installed = Test-Path -LiteralPath $python
  Write-Host "Environment: $(if ($installed) { 'ready' } else { 'missing, run .\run.bat once' })"
  if ($installed) { & $python -c "import api_server, localdeploy; print('Imports: ready')" }
  Write-Host "API: $(if (Test-Ready) { $url } else { 'not running' })"
  exit $(if ($installed) { 0 } else { 1 })
}
if ($Action -eq "repair") {
  Remove-Item -LiteralPath ".\.venv\requirements.sha256" -Force -ErrorAction SilentlyContinue
}
& .\scripts\start.ps1 -NoBrowser:$NoBrowser
exit $LASTEXITCODE

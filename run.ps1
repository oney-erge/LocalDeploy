param(
  [ValidateSet("run", "doctor", "repair", "docker", "stop", "logs")]
  [string]$Action = "run",
  [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$port = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$url = "http://127.0.0.1:$port"

function Test-Ready {
  try { Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}
function Wait-Ready {
  for ($i = 0; $i -lt 120; $i++) { if (Test-Ready) { return $true }; Start-Sleep -Milliseconds 500 }
  return $false
}

if ($Action -in @("docker", "logs") -or ($Action -eq "stop" -and (Test-Path .\docker-compose.yml))) {
  if ($Action -eq "stop" -and (Test-Path .\logs\api_server.pid)) { & .\scripts\stop.ps1; exit $LASTEXITCODE }
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    if ($Action -eq "stop") { & .\scripts\stop.ps1; exit $LASTEXITCODE }
    throw "Docker is not installed."
  }
  docker info *> $null
  if ($LASTEXITCODE -ne 0) {
    if ($Action -eq "stop") { & .\scripts\stop.ps1; exit $LASTEXITCODE }
    throw "Docker is installed but its engine is not running."
  }
  if ($Action -eq "stop") { docker compose down; exit $LASTEXITCODE }
  if ($Action -eq "logs") { docker compose logs --follow; exit $LASTEXITCODE }
  docker compose up --detach --build
  if (-not (Wait-Ready)) { docker compose logs; throw "LocalDeploy did not become ready at $url." }
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

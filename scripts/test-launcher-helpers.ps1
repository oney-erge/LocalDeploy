# Checks the launcher helpers in scripts/install-utils.ps1 under the PowerShell that run.bat starts
# (Windows PowerShell 5.1). Run it with:  powershell -NoProfile -File scripts\test-launcher-helpers.ps1
# The launchers were once only parsed, never executed, under 5.1, and a plain "Using CPython ..." line on
# stderr aborted every first run.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "install-utils.ps1")

function Assert-That([bool]$Condition, [string]$Message) {
  if (-not $Condition) { throw "FAILED: $Message" }
  Write-Host "ok: $Message"
}

# 1. stderr output from a native command is captured, not fatal, and the exit code survives.
$result = Invoke-InstallNative "cmd.exe" @("/c", "echo progress on stderr 1>&2 & exit 3")
Assert-That ($result.ExitCode -eq 3) "native exit code is returned (got $($result.ExitCode))"
Assert-That (($result.Output -join " ") -match "progress on stderr") "stderr text is captured as a plain line"
Assert-That ($ErrorActionPreference -eq "Stop") "the caller's ErrorActionPreference is restored"

$clean = Invoke-InstallNative "cmd.exe" @("/c", "echo fine & exit 0")
Assert-That ($clean.ExitCode -eq 0) "a successful command returns exit code 0"

# 2. A quiet probe drops stderr without failing and leaves the exit code in $LASTEXITCODE.
$quiet = Invoke-NativeQuiet "cmd.exe" @("/c", "echo engine not running 1>&2 & exit 1")
Assert-That ($LASTEXITCODE -eq 1) "a quiet probe keeps the exit code (got $LASTEXITCODE)"
Assert-That ($quiet.Count -eq 0) "a quiet probe drops stderr"
$listed = Invoke-NativeQuiet "cmd.exe" @("/c", "echo id-123 & exit 0")
Assert-That (($listed -join " ") -match "id-123") "a quiet probe returns stdout"
Assert-That ($ErrorActionPreference -eq "Stop") "the caller's ErrorActionPreference is restored after a probe"

# 3. An interrupted setup's .venv (no Python inside) is removed; a working one is never touched.
$scratch = Join-Path ([IO.Path]::GetTempPath()) ("launcher-helper-test-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $scratch | Out-Null
try {
  Initialize-Install -RepositoryRoot $scratch -ProductName "helper test"

  New-Item -ItemType Directory -Path (Join-Path $scratch ".venv\Lib") -Force | Out-Null
  Remove-IncompleteVenv $scratch
  Assert-That (-not (Test-Path -LiteralPath (Join-Path $scratch ".venv"))) "a .venv without Python is removed"

  # A working interpreter: copy cmd.exe and probe it with arguments it accepts.
  New-Item -ItemType Directory -Path (Join-Path $scratch ".venv\Scripts") -Force | Out-Null
  Copy-Item -LiteralPath (Join-Path $env:SystemRoot "System32\cmd.exe") -Destination (Join-Path $scratch ".venv\Scripts\python.exe")
  $script:VenvProbeArguments = @("/c", "exit 0")
  Remove-IncompleteVenv $scratch
  Assert-That (Test-Path -LiteralPath (Join-Path $scratch ".venv\Scripts\python.exe")) "a .venv whose Python runs is left alone"
  Assert-That (@(Get-ChildItem -LiteralPath (Join-Path $scratch ".setup") -Directory -Filter "venv-broken-*" -ErrorAction SilentlyContinue).Count -eq 0) "a working .venv is not moved anywhere"

  # A Python that exists but cannot start (here a text file; in real life a removed base interpreter).
  Remove-Item -LiteralPath (Join-Path $scratch ".venv") -Recurse -Force
  New-Item -ItemType Directory -Path (Join-Path $scratch ".venv\Scripts") -Force | Out-Null
  Set-Content -LiteralPath (Join-Path $scratch ".venv\Scripts\python.exe") -Value "not an executable"
  Remove-IncompleteVenv $scratch
  Assert-That (-not (Test-Path -LiteralPath (Join-Path $scratch ".venv"))) "a .venv whose Python cannot run is moved out of the way"
  $parked = @(Get-ChildItem -LiteralPath (Join-Path $scratch ".setup") -Directory -Filter "venv-broken-*")
  Assert-That ($parked.Count -eq 1) "the broken .venv is kept under .setup, not deleted"
  Assert-That (Test-Path -LiteralPath (Join-Path $parked[0].FullName "Scripts\python.exe")) "the parked venv still has its files"

  # A Python that starts but exits non-zero (the 'No Python at ...' case: exit 103) is broken too.
  New-Item -ItemType Directory -Path (Join-Path $scratch ".venv\Scripts") -Force | Out-Null
  Copy-Item -LiteralPath (Join-Path $env:SystemRoot "System32\cmd.exe") -Destination (Join-Path $scratch ".venv\Scripts\python.exe")
  $script:VenvProbeArguments = @("/c", "exit 103")
  Remove-IncompleteVenv $scratch
  Assert-That (-not (Test-Path -LiteralPath (Join-Path $scratch ".venv"))) "a Python that exits non-zero is treated as broken"
  $parked = @(Get-ChildItem -LiteralPath (Join-Path $scratch ".setup") -Directory -Filter "venv-broken-*")
  Assert-That ($parked.Count -eq 2) "each broken .venv is kept under its own name"

  Remove-IncompleteVenv (Join-Path $scratch "does-not-exist")
  Assert-That $true "a missing folder is a no-op"
} finally {
  Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}

# A PID file names a process only while that PID still runs the recorded program. One left behind
# after the API was killed elsewhere made stop.ps1 force-stop whatever held the number next.
$pidFile = Join-Path ([IO.Path]::GetTempPath()) ("launcher-helper-" + [Guid]::NewGuid().ToString("N") + ".pid")
$other = Start-Process -FilePath "cmd.exe" -ArgumentList "/c ping -n 30 127.0.0.1 > nul" -WindowStyle Hidden -PassThru
try {
  Set-Content -LiteralPath $pidFile -Value $other.Id
  Assert-That (-not (Stop-RecordedProcess -PidFile $pidFile -CommandLineLike "*api_server.py*")) "a PID now running something else is not stopped"
  Assert-That (-not $other.HasExited) "the unrelated process is still running"
  Assert-That (-not (Test-Path -LiteralPath $pidFile)) "the stale PID file is removed"

  Set-Content -LiteralPath $pidFile -Value $other.Id
  Assert-That (Stop-RecordedProcess -PidFile $pidFile -CommandLineLike "*ping -n 30*") "the recorded program is stopped"
  Assert-That (-not (Stop-RecordedProcess -PidFile $pidFile -CommandLineLike "*")) "a missing PID file stops nothing"
} finally {
  Stop-Process -Id $other.Id -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}
Write-Host "All launcher helper checks passed."
# The probes above leave $LASTEXITCODE at 103 on purpose, and CI's PowerShell step exits with $LASTEXITCODE.
exit 0

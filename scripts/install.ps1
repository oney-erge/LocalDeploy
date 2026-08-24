# LocalDeploy one-liner web installer for Windows.
#
#   irm https://raw.githubusercontent.com/oney-erge/LocalDeploy/main/scripts/install.ps1 | iex
#
# Needs nothing preinstalled: uses git when available, otherwise downloads the
# repo as a ZIP. Then hands off to scripts\start.ps1, which walks through
# Python and Ollama installation (winget) and starts the app.
$ErrorActionPreference = "Stop"

$dest = if ($env:LOCALDEPLOY_DIR) { $env:LOCALDEPLOY_DIR } else { Join-Path $HOME "LocalDeploy" }
$repo = "https://github.com/oney-erge/LocalDeploy"

Write-Host "==> Installing LocalDeploy to $dest" -ForegroundColor Cyan

if (Test-Path -LiteralPath (Join-Path $dest "scripts\start.ps1")) {
    Write-Host "==> Existing install found at $dest" -ForegroundColor Cyan
    if (Test-Path -LiteralPath (Join-Path $dest ".git")) {
        Push-Location $dest
        $statusOutput = git status --porcelain
        if ($statusOutput) {
            Write-Warning "Local changes found in $dest - skipping update so nothing is overwritten. Run 'git pull' there yourself once those are committed or stashed."
        }
        else {
            Write-Host "==> Updating with git pull ..." -ForegroundColor Cyan
            git pull --ff-only
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "git pull failed (offline, or the local branch has diverged) - continuing with the existing checkout."
            }
        }
        Pop-Location
    }
    else {
        Write-Host "==> This copy was not installed via git (e.g. a ZIP download), so it will not be auto-updated. Starting it as-is." -ForegroundColor Cyan
    }
}
elseif (Get-Command git -ErrorAction SilentlyContinue) {
    git clone --depth 1 $repo $dest
}
else {
    Write-Host "==> git not found - downloading ZIP instead" -ForegroundColor Cyan
    $zip = Join-Path $env:TEMP "localdeploy-main.zip"
    Invoke-WebRequest -Uri "$repo/archive/refs/heads/main.zip" -OutFile $zip
    $tmp = Join-Path $env:TEMP "localdeploy-extract"
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    Expand-Archive -Path $zip -DestinationPath $tmp
    # The ZIP wraps everything in a LocalDeploy-main/ folder.
    Move-Item -Path (Join-Path $tmp "LocalDeploy-main") -Destination $dest
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

Set-Location $dest
& .\scripts\start.ps1

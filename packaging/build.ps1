# Build the Windows LocalDeploy tray executable (Release R7, Phase A), and
# optionally package it as a per-user .msi installer.
#
# Usage:
#   .\packaging\build.ps1            # dist\LocalDeploy\LocalDeploy.exe only
#   .\packaging\build.ps1 -Msi       # also produces dist\LocalDeploy-Setup.msi
#
# Uses the project's own .venv (created via scripts\start.ps1 if you haven't
# already), not a bare `python`/`pip install --user` - a system Python whose
# user-site Scripts directory isn't on PATH will otherwise install pyinstaller
# somewhere the very next line can't find it (hit exactly this while testing
# this script; --user + PATH is machine-dependent, .venv isn't).
#
# -Msi needs the WiX v3 toolset (candle/light/heat). No admin rights and no
# EULA click-through required: WiX v3.14 predates the "Open Source
# Maintenance Fee" that WiX v6/v7 gate every command behind, and it's fetched
# here as a plain NuGet package into build\wix-tools (not installed
# system-wide), so this works the same in CI as on a laptop.
param(
    [switch]$Msi
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $repoRoot "scripts\install-utils.ps1")

Push-Location $repoRoot
try {
    $python = ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host "Creating virtual environment (.venv)..."
        python -m venv .venv
        if (-not (Test-Path -LiteralPath $python)) {
            throw "Could not create .venv. Run scripts\start.ps1 once first, or check your Python install."
        }
    }
    $null = Invoke-NativeQuiet $python @("-c", "import PyInstaller")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing packaging dependencies (pyinstaller, pystray, pillow)..."
        & $python -m pip install -e ".[packaging]"
    }
    & $python -m PyInstaller packaging/localdeploy.spec --distpath dist --workpath build --noconfirm
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }
    Write-Host "`nBuilt: $repoRoot\dist\LocalDeploy\LocalDeploy.exe"

    if (-not $Msi) {
        Write-Host "Not done: code signing, the .msi installer (-Msi), or auto-start-at-login."
        return
    }

    # --- .msi packaging -----------------------------------------------------
    $wixDir = Join-Path $repoRoot "build\wix-tools"
    $wixTools = Join-Path $wixDir "extracted\tools"
    if (-not (Test-Path (Join-Path $wixTools "candle.exe"))) {
        Write-Host "`nFetching WiX v3.14 toolset (one-time, ~40 MB, no admin rights needed)..."
        New-Item -ItemType Directory -Force -Path $wixDir | Out-Null
        $nupkg = Join-Path $wixDir "wix314.nupkg"
        Invoke-WebRequest -Uri "https://www.nuget.org/api/v2/package/WiX/3.14.1" -OutFile $nupkg
        Expand-Archive -Path $nupkg -DestinationPath (Join-Path $wixDir "extracted") -Force
    }
    $candle = Join-Path $wixTools "candle.exe"
    $light = Join-Path $wixTools "light.exe"
    $heat = Join-Path $wixTools "heat.exe"
    $wixUIExt = Join-Path $wixTools "WixUIExtension.dll"
    $wixUtilExt = Join-Path $wixTools "WixUtilExtension.dll"

    $version = & $python -c "import localdeploy; print(localdeploy.__version__)"
    Write-Host "`nHarvesting PyInstaller output (dist\LocalDeploy) into WiX components..."
    New-Item -ItemType Directory -Force -Path "build\wixobj" | Out-Null
    & $heat dir "dist\LocalDeploy" `
        -cg LocalDeployFiles -ag -scom -sreg -sfrag -srd `
        -dr INSTALLFOLDER -var var.SourceDir `
        -out "build\files.wxs" -nologo
    if ($LASTEXITCODE -ne 0) { throw "heat.exe failed (exit $LASTEXITCODE)" }

    # Per-user MSI components must use an HKCU registry KeyPath, not a file
    # (Windows Installer's ICE38 rule) - see fix_perfile_keypaths.py's own
    # docstring for the full reasoning. heat.exe's raw output always makes the
    # File the KeyPath, so every harvest needs this pass before compiling.
    & $python "packaging\windows\fix_perfile_keypaths.py" "build\files.wxs" "build\files.fixed.wxs"
    if ($LASTEXITCODE -ne 0) { throw "fix_perfile_keypaths.py failed (exit $LASTEXITCODE)" }

    Write-Host "Compiling..."
    & $candle -nologo `
        -dSourceDir="dist\LocalDeploy" -dProductVersion="$version" `
        -ext $wixUIExt -ext $wixUtilExt `
        -out "build\wixobj\" `
        "packaging\windows\product.wxs" "build\files.fixed.wxs"
    if ($LASTEXITCODE -ne 0) { throw "candle.exe failed (exit $LASTEXITCODE)" }

    Write-Host "Linking..."
    # ICE64 is suppressed deliberately, not silenced blindly: it flags every
    # per-user directory as "not in the RemoveFile table" because it can't see
    # that util:RemoveFolderEx (product.wxs) provides equivalent cleanup at
    # uninstall time. Verified manually - install, confirm files/shortcuts/ARP
    # entry, uninstall, confirm the whole tree and registry keys are gone.
    & $light -nologo -sice:ICE64 `
        -ext $wixUIExt -ext $wixUtilExt -cultures:en-us `
        -b "packaging\windows" `
        -out "dist\LocalDeploy-Setup.msi" `
        "build\wixobj\product.wixobj" "build\wixobj\files.fixed.wixobj"
    if ($LASTEXITCODE -ne 0) { throw "light.exe failed (exit $LASTEXITCODE)" }

    Write-Host "`nBuilt: $repoRoot\dist\LocalDeploy-Setup.msi"
    Write-Host "Per-user install (no admin/UAC prompt), Start Menu + Desktop shortcuts, real Settings > Apps entry."
    Write-Host "Not done: code signing (unsigned .exe/.msi trigger a Windows SmartScreen warning on first run)."
} finally {
    Pop-Location
}

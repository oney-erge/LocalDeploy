# PyInstaller spec for the LocalDeploy desktop tray launcher on macOS.
# Mirrors packaging/localdeploy.spec (Windows); the differences are macOS-
# specific: a proper .app bundle (BUNDLE()) instead of a plain folder, an
# .icns icon, and an Info.plist with LSUIElement=1 so the app behaves as a
# menu-bar-only utility (no Dock icon, no Cmd+Tab entry) - matching the
# Windows build's tray-only, no-taskbar-button behavior.
#
# NOT verified on real macOS hardware as of writing (built and reasoned about
# from a Windows machine, using the officially documented PyInstaller BUNDLE()
# API) - see .github/workflows/build-macos.yml, which builds and smoke-tests
# this on an actual macos-latest GitHub Actions runner. Treat this file as
# reviewed-but-unverified until that workflow (or a real Mac) has run it once.
#
# Build (on macOS):
#   pip install -e ".[packaging]"
#   pyinstaller packaging/macos.spec --distpath dist --workpath build
#
# Output: dist/LocalDeploy.app
from __future__ import annotations

from pathlib import Path

block_cipher = None
REPO_ROOT = Path(SPECPATH).resolve().parent  # noqa: F821

a = Analysis(
    [str(REPO_ROOT / "packaging" / "tray_app.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=[
        (str(REPO_ROOT / "localdeploy" / "web"), "localdeploy/web"),
    ],
    hiddenimports=[
        "api_server",
        "benchmark",
        "chat_cli",
        "compare_models",
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
        "yaml",
        "multipart",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalDeploy",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(REPO_ROOT / "packaging" / "macos" / "localdeploy.icns"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="LocalDeploy",
)

app = BUNDLE(
    coll,
    name="LocalDeploy.app",
    icon=str(REPO_ROOT / "packaging" / "macos" / "localdeploy.icns"),
    bundle_identifier="com.localdeploy.app",
    info_plist={
        "CFBundleName": "LocalDeploy",
        "CFBundleDisplayName": "LocalDeploy",
        "CFBundleShortVersionString": "0.0.0",  # overwritten by the build workflow with localdeploy.__version__
        "NSHighResolutionCapable": True,
        # Menu-bar-only utility: no Dock icon, no app-switcher entry -
        # matches the Windows build's tray-only, no-taskbar-button behavior.
        "LSUIElement": True,
        "NSHumanReadableCopyright": "MIT License - oney-erge",
    },
)

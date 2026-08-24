# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the TEMPEST Analysis Suite.

Build with:   pyinstaller tempest_suite.spec
Output:       dist/TempestSuite/TempestSuite.exe
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# customtkinter ships theme JSON / assets that must be bundled as data files.
datas = collect_data_files("customtkinter")

# Pull in optional SDR backends only if they happen to be installed; the app
# degrades gracefully when they are absent.
hiddenimports = []
for mod in ("rtlsdr", "SoapySDR", "uhd", "soundfile"):
    try:
        __import__(mod)
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass

a = Analysis(
    ["tempest_suite22.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="TempestSuite",
    console=False,          # windowed GUI app
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=True, name="TempestSuite",
)

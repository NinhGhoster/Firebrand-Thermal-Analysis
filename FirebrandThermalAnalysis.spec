# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import sys

datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('fnv')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Determine correct icon format depending on operating system
if sys.platform == 'darwin':
    icon_path = 'docs/logo.icns'
else:
    icon_path = 'docs/logo.ico'

a = Analysis(
    ['FirebrandThermalAnalysis.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FirebrandThermalAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FirebrandThermalAnalysis',
)
app = BUNDLE(
    coll,
    name='FirebrandThermalAnalysis.app',
    icon=icon_path,
    bundle_identifier='com.ninhghoster.firebrandthermalanalysis',
    info_plist={
        'CFBundleName': 'FirebrandThermalAnalysis',
        'CFBundleDisplayName': 'FirebrandThermalAnalysis',
        'CFBundleExecutable': 'FirebrandThermalAnalysis',
    },
)

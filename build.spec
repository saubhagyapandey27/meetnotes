# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Exclude heavy modules not needed by the application to minimize bundle size
excluded_modules = [
    'unittest', 'email', 'html', 'http', 'urllib', 'xmlrpc', 
    'pydoc', 'doctest', 'logging', 'multiprocessing', 'concurrent',
    'scipy', 'matplotlib', 'pandas', 'IPython', 'notebook',
    'PIL', 'tkinter.test'
]

# Collect packaging-critical data files for models and packages
datas = []
try:
    datas += collect_data_files('model2vec')
except Exception:
    pass

try:
    datas += collect_data_files('moonshine_voice')
except Exception:
    pass

# Dynamic icon detection
icon_path = 'assets/icon.ico' if os.path.exists('assets/icon.ico') else None

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'soundcard', 'sounddevice', 'soundfile', 'model2vec', 'moonshine_voice',
        'sqlite3', 'requests', 'numpy'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MeetNotes',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,              # Compress executable binaries
    console=False,          # Disable terminal window popup (fully windowed GUI)
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
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MeetNotes',
)

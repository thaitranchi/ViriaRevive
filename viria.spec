# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

is_debug = os.environ.get('VIRIA_DEBUG') == '1'

datas = [('gui', 'gui')]
# Ultralytics/YOLO requires specific data files to be bundled for the engine to load
datas += collect_data_files('ultralytics')

hiddenimports = [
    'select',
    'selectors',
    'webview.platforms.winforms',
    'ultralytics',
    'faster_whisper',
    'torch',
    'googleapiclient.discovery',
    'google_auth_oauthlib.flow',
    'viral_score',
]
# pgvector is imported inside functions in database/vector.py, so PyInstaller's
# static analysis cannot see it. Every submodule matters: Vector lives in
# pgvector.sqlalchemy.vector, not pgvector.sqlalchemy itself.
hiddenimports += collect_submodules('pgvector')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ViriaRevive',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=is_debug,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[], name='ViriaRevive',
)
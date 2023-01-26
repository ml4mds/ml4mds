# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        (
            "C:\\Users\\binzh\\AppData\\Roaming\\Python\\Python38\\site-packages\\altair\\vegalite\\v4\\schema\\vega-lite-schema.json",
            ".\\altair\\vegalite\\v4\\schema"
        ),
        (
            "C:\\Users\\binzh\\AppData\\Roaming\\Python\\Python38\\site-packages\\streamlit",
            ".\\streamlit"
        ),
        (
            "C:\\Users\\binzh\\AppData\\Roaming\\Python\\Python38\\site-packages\\sklearn",
            ".\\sklearn"
        ),
        (
            "C:\\Users\\binzh\\AppData\\Roaming\\Python\\Python38\\site-packages\\joblib",
            ".\\joblib"
        )
    ],
    hiddenimports=[],
    hookspath=['.\\hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='main',
    debug=None,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

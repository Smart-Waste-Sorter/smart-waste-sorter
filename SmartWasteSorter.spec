# -*- mode: python ; coding: utf-8 -*-

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

project_root = Path(SPECPATH)
classifier_model = project_root / "model" / "best.pt"
person_model = project_root / "yolo11n.pt"

missing_models = [
    str(path) for path in (classifier_model, person_model) if not path.is_file()
]
if missing_models:
    raise FileNotFoundError(
        "Required offline model file(s) are missing: " + ", ".join(missing_models)
    )

datas = [
    (str(classifier_model), "model"),
    (str(person_model), "."),
]
torchvision_spec = importlib.util.find_spec("torchvision")
if torchvision_spec is None or not torchvision_spec.submodule_search_locations:
    raise ImportError("TorchVision is required to build the Smart Waste Sorter.")

torchvision_directory = Path(next(iter(torchvision_spec.submodule_search_locations)))
torchvision_extensions = sorted(torchvision_directory.glob("_C*.pyd"))
if not torchvision_extensions:
    raise FileNotFoundError(
        f"TorchVision native extension not found in {torchvision_directory}"
    )

binaries = [(str(path), "torchvision") for path in torchvision_extensions]
hiddenimports = [
    "comtypes.client",
    "pyttsx3.drivers.sapi5",
    "pythoncom",
    "pywintypes",
    "win32com.client",
]
hiddenimports += collect_submodules("ultralytics.nn")

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
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
    a.binaries,
    a.datas,
    [],
    name="SmartWasteSorter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

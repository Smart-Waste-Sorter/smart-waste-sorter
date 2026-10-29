from __future__ import annotations

import os
from pathlib import Path
import sys


def resource_path(*parts: str) -> Path:
    """Locate read-only files in the source tree or PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS"))
    else:
        base = Path(__file__).resolve().parent.parent
    return base.joinpath(*parts)


def data_path(*parts: str) -> Path:
    """Locate writable app data without writing into a frozen bundle."""
    if getattr(sys, "frozen", False):
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        ) / "SmartWasteSorter"
    else:
        base = resource_path("data")

    base.mkdir(parents=True, exist_ok=True)
    return base.joinpath(*parts)

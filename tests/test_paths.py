from pathlib import Path

from src import paths


def test_source_resources_and_data_use_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "__file__", str(tmp_path / "src" / "paths.py"))

    assert paths.resource_path("model", "best.pt") == (
        tmp_path / "model" / "best.pt"
    )
    assert paths.data_path("settings.json") == (
        tmp_path / "data" / "settings.json"
    )
    assert (tmp_path / "data").is_dir()


def test_frozen_resources_use_bundle_and_data_use_local_appdata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    assert paths.resource_path("model", "best.pt") == (
        tmp_path / "bundle" / "model" / "best.pt"
    )
    assert paths.data_path("smart_waste.db") == (
        tmp_path / "local" / "SmartWasteSorter" / "smart_waste.db"
    )
    assert (tmp_path / "local" / "SmartWasteSorter").is_dir()

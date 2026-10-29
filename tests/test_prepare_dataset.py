from pathlib import Path

from PIL import Image

from scripts import prepare_dataset


def test_missing_raw_category_does_not_delete_existing_dataset(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw_images"
    (raw_dir / "Organic").mkdir(parents=True)
    (raw_dir / "Organic" / "sample.jpg").touch()
    output_dir = tmp_path / "dataset"
    existing_image = output_dir / "train" / "Plastic" / "keep.jpg"
    existing_image.parent.mkdir(parents=True)
    existing_image.touch()

    monkeypatch.setattr(prepare_dataset, "RAW_DIR", raw_dir)
    monkeypatch.setattr(prepare_dataset, "OUT_DIR", output_dir)

    prepare_dataset.main()

    assert existing_image.exists()
    assert "No existing split files were changed" in capsys.readouterr().out


def test_non_waste_folder_alias_is_supported():
    assert prepare_dataset.category_of("Non-Waste") == "Other"


def test_five_required_waste_categories_are_sufficient():
    assert set(prepare_dataset.REQUIRED_CATEGORIES) == {
        "Plastic",
        "Paper",
        "Can_Metal",
        "Glass",
        "Organic",
    }


def test_prepare_dataset_builds_splits_without_optional_other(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw_images"
    output_dir = tmp_path / "dataset"
    for category in prepare_dataset.REQUIRED_CATEGORIES:
        category_dir = raw_dir / category
        category_dir.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (8, 8), color=(index * 40, 20, 30)).save(
                category_dir / f"sample_{index}.jpg"
            )

    monkeypatch.setattr(prepare_dataset, "RAW_DIR", raw_dir)
    monkeypatch.setattr(prepare_dataset, "OUT_DIR", output_dir)
    monkeypatch.setattr(prepare_dataset, "MANIFEST", tmp_path / "manifest.csv")

    prepare_dataset.main()

    assert not (output_dir / "train" / "Other").exists()
    for split in prepare_dataset.SPLITS:
        for category in prepare_dataset.REQUIRED_CATEGORIES:
            assert list((output_dir / split / category).glob("*.jpg"))

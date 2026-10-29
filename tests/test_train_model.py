from pathlib import Path

import pytest

from scripts.train_model import CATEGORIES, OTHER_CATEGORY, SPLITS, validate_dataset


def make_dataset(root: Path) -> None:
    for split in SPLITS:
        for category in [*CATEGORIES, OTHER_CATEGORY]:
            class_dir = root / split / category
            class_dir.mkdir(parents=True)
            (class_dir / "sample.jpg").touch()


def test_validate_dataset_counts_images_by_split_and_category(tmp_path):
    make_dataset(tmp_path)

    counts = validate_dataset(tmp_path)

    assert counts["train"]["Plastic"] == 1
    assert counts["val"]["Organic"] == 1
    assert counts["test"]["Glass"] == 1
    assert counts["train"]["Other"] == 1


def test_validate_dataset_rejects_missing_category_images(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "train" / "Organic" / "sample.jpg").unlink()

    with pytest.raises(ValueError, match="No images found"):
        validate_dataset(tmp_path)


def test_validate_dataset_accepts_five_waste_categories_without_other(tmp_path):
    for split in SPLITS:
        for category in CATEGORIES:
            class_dir = tmp_path / split / category
            class_dir.mkdir(parents=True)
            (class_dir / "sample.jpg").touch()

    counts = validate_dataset(tmp_path)

    assert "Other" not in counts["train"]
    assert set(counts["train"]) == set(CATEGORIES)


def test_validate_dataset_rejects_other_present_in_only_some_splits(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "train" / OTHER_CATEGORY / "sample.jpg").unlink()
    (tmp_path / "train" / OTHER_CATEGORY).rmdir()

    with pytest.raises(ValueError, match="present in every dataset split"):
        validate_dataset(tmp_path)

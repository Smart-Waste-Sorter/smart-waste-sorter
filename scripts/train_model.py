"""Train the waste classifier and install its weights for the desktop app.

Run from the project root:
    python scripts/train_model.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CATEGORIES, MODEL_PATH, OTHER_CATEGORY

DATASET_DIR = ROOT / "dataset"
SPLITS = ("train", "val", "test")
IMAGE_TYPES = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp"}
RECOMMENDED_COUNTS = {"train": 50, "val": 10, "test": 10}
TRAINING_CATEGORIES = [*CATEGORIES, OTHER_CATEGORY]


def validate_dataset(dataset_dir: Path) -> dict[str, dict[str, int]]:
    """Check the Ultralytics classification folder layout and count images."""
    counts: dict[str, dict[str, int]] = {}
    has_other_category: bool | None = None
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            raise ValueError(f"Dataset split folder is missing: {split_dir}")

        class_dirs = {path.name for path in split_dir.iterdir() if path.is_dir()}
        missing = set(CATEGORIES) - class_dirs
        extra = class_dirs - set(TRAINING_CATEGORIES)
        if missing:
            raise ValueError(
                f"{split_dir} is missing category folders: {', '.join(sorted(missing))}"
            )
        if extra:
            raise ValueError(
                f"{split_dir} has unsupported category folders: {', '.join(sorted(extra))}"
            )

        split_has_other = OTHER_CATEGORY in class_dirs
        if has_other_category is None:
            has_other_category = split_has_other
        elif has_other_category != split_has_other:
            raise ValueError(
                f"{OTHER_CATEGORY} must be present in every dataset split or none."
            )

        categories = [*CATEGORIES]
        if split_has_other:
            categories.append(OTHER_CATEGORY)
        counts[split] = {}
        for category in categories:
            class_dir = split_dir / category
            image_count = sum(
                1
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_TYPES
            )
            if image_count == 0:
                raise ValueError(f"No images found in {class_dir}")
            counts[split][category] = image_count
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=32, help="Training batch size (default: 32)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error("--epochs and --batch must be positive integers")

    try:
        counts = validate_dataset(DATASET_DIR)
    except ValueError as exc:
        raise SystemExit(f"Dataset error: {exc}") from exc

    print("Images per category:")
    print(f"{'Category':<12}{'train':>8}{'val':>8}{'test':>8}")
    training_categories = list(counts["train"])
    for category in training_categories:
        print(
            f"{category:<12}"
            f"{counts['train'][category]:>8}"
            f"{counts['val'][category]:>8}"
            f"{counts['test'][category]:>8}"
        )

    below_recommendation = [
        f"{category} ({split}: {counts[split][category]}, recommended "
        f"{minimum}+)"
        for category in training_categories
        for split, minimum in RECOMMENDED_COUNTS.items()
        if counts[split][category] < minimum
    ]
    if below_recommendation:
        print(
            "\nWarning: some categories have too few images for reliable "
            "recognition:\n  " + "\n  ".join(below_recommendation)
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Activate the project environment "
            "and run: pip install -r requirements.txt"
        ) from exc

    run_name = f"waste_sorter_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = ROOT / "runs" / "classify" / run_name
    person_detector = YOLO("yolo11n.pt")
    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=str(DATASET_DIR),
        epochs=args.epochs,
        batch=args.batch,
        project=str(run_dir.parent),
        name=run_name,
    )

    trained_weights = run_dir / "weights" / "best.pt"
    if not trained_weights.is_file():
        raise RuntimeError(f"Training finished without producing weights: {trained_weights}")

    detector_weights = run_dir / "person_detector.pt"
    person_detector.save(str(detector_weights))

    app_weights = ROOT / MODEL_PATH
    app_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained_weights, app_weights)
    shutil.copy2(detector_weights, app_weights.with_name("person_detector.pt"))
    print(f"\nModel saved to {app_weights}")
    print(f"Person detector saved to {app_weights.with_name('person_detector.pt')}")
    print("Restart the app to load the trained model.")


if __name__ == "__main__":
    main()

"""Prepare the dataset: convert, remove duplicates, and split into train/val/test.

Put your pictures in raw_images/<Category>/ then run:
    python scripts/prepare_dataset.py

Provide images for every category before running this script. It rebuilds the
matching category folders under dataset/train, dataset/val, and dataset/test.
"""
import csv
import hashlib
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw_images"
OUT_DIR = ROOT / "dataset"
MANIFEST = ROOT / "docs" / "dataset_manifest.csv"

# These names MUST match the folder names the AI will learn from.
CATEGORIES = ["Plastic", "Paper", "Can_Metal", "Glass", "Organic", "Other"]
REQUIRED_CATEGORIES = [category for category in CATEGORIES if category != "Other"]

# Other spellings of a raw folder name that we accept.
ALIASES = {
    "plastic": "Plastic",
    "paper": "Paper",
    "can_metal": "Can_Metal",
    "metal": "Can_Metal",
    "can": "Can_Metal",
    "glass": "Glass",
    "organic": "Organic",
    "other": "Other",
    "non_waste": "Other",
    "nonwaste": "Other",
}

IMAGE_TYPES = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def category_of(folder_name):
    key = folder_name.lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(key)


def group_of(path):
    """Photos named item__001.jpg share the group 'item' and stay together."""
    stem = path.stem
    return stem.split("__")[0] if "__" in stem else stem


def main():
    if not RAW_DIR.exists():
        print(f"Folder not found: {RAW_DIR}")
        return

    raw_folders = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    available_categories = {
        category_of(folder.name)
        for folder in raw_folders
        if any(path.suffix.lower() in IMAGE_TYPES for path in folder.rglob("*"))
    }
    missing_categories = set(REQUIRED_CATEGORIES) - available_categories
    if missing_categories:
        print(
            "Add source images for every waste category before rebuilding the dataset. "
            "No existing split files were changed. Missing: "
            + ", ".join(sorted(missing_categories))
        )
        return
    categories_to_use = [
        category for category in CATEGORIES if category in available_categories
    ]

    # Remove only .jpg files made by an earlier run, so runs never mix.
    for split in SPLITS:
        for cat in CATEGORIES:
            folder = OUT_DIR / split / cat
            if cat not in categories_to_use and not folder.exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            for old in folder.glob("*.jpg"):
                old.unlink()

    rng = random.Random(SEED)
    manifest = []
    summary = {cat: {s: 0 for s in SPLITS} for cat in categories_to_use}

    for raw_folder in raw_folders:
        cat = category_of(raw_folder.name)
        if cat is None:
            print(f"Skipping unknown folder: {raw_folder.name}")
            continue

        files = sorted(p for p in raw_folder.rglob("*") if p.suffix.lower() in IMAGE_TYPES)
        seen_hashes, groups = set(), {}
        duplicates = broken = 0

        for path in files:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                duplicates += 1
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                broken += 1
                print(f"  Broken image skipped: {path.name}")
                continue
            seen_hashes.add(digest)
            groups.setdefault(group_of(path), []).append(path)

        total = sum(len(v) for v in groups.values())
        if total == 0:
            print(f"{cat}: no usable images found")
            continue

        group_names = sorted(groups)
        rng.shuffle(group_names)
        want_val = max(1, round(total * VAL_RATIO))
        want_test = max(1, round(total * TEST_RATIO))
        counts = {s: 0 for s in SPLITS}
        counter = 0

        for name in group_names:
            if counts["val"] < want_val:
                split = "val"
            elif counts["test"] < want_test:
                split = "test"
            else:
                split = "train"
            for path in groups[name]:
                counter += 1
                new_name = f"{cat.lower()}_{counter:04d}.jpg"
                with Image.open(path) as img:
                    img.convert("RGB").save(OUT_DIR / split / cat / new_name, quality=95)
                counts[split] += 1
                summary[cat][split] += 1
                manifest.append([new_name, cat, split, path.name])

        print(f"{cat}: {total} images used, {duplicates} duplicates removed, {broken} broken skipped")

    MANIFEST.parent.mkdir(exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_name", "category", "split", "original_name"])
        writer.writerows(manifest)

    print("\nImages per category and split")
    print(f"{'Category':<12}{'train':>7}{'val':>6}{'test':>6}")
    for cat in categories_to_use:
        s = summary[cat]
        print(f"{cat:<12}{s['train']:>7}{s['val']:>6}{s['test']:>6}")
    print(f"\nA record of every file was saved to {MANIFEST}")


if __name__ == "__main__":
    main()
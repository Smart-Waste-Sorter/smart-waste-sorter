# Smart Waste Sorter Dataset Documentation

## Project Overview

Smart Waste Sorter is a Python desktop application that classifies a waste item
from a camera image and suggests a disposal bin. The project uses Ultralytics
YOLO for image classification. The saved waste model is a **classification**
model, not a waste-object detector.

This document describes the five-class dataset in `dataset/train`, `dataset/val`,
and `dataset/test`. A separate O/R image collection is also present under
`dataset/DATASET/TRAIN`; it is not the input used by the project's five-class
training script.

## Dataset Overview

The current classifier dataset contains **1,898 images** in five categories:

- Training: 1,328 images
- Validation: 285 images
- Testing: 285 images

Counts were made from the image files currently on disk and checked against
`docs/dataset_manifest.csv`. The manifest contains one row for each of the 1,898
images and its category and split.

The downloaded O/R collection is separate. Its `O` folder has 12,565 images and
its `R` folder has 6,824 images, for **19,389 additional images**. Their label
definitions are **Not yet verified** from the project files. Including both
collections, there are 21,287 image files under `dataset/`, but only 1,898 are
organized as the current model's train/validation/test dataset.

## Dataset Source

The project documentation identifies these sources:

| Data | Source recorded in the project | What is present locally |
| --- | --- | --- |
| Plastic, Glass, Can_Metal, Paper | [TrashNet](https://github.com/garythung/trashnet), using `data/dataset-resized.zip` | 1,893 images in `raw_images/` and in the prepared splits |
| Organic | [RealWaste (UCI)](https://archive.ics.uci.edu/dataset/908/realwaste), Food Organics class | 5 images in `raw_images/` and in the prepared splits |
| Separate O/R collection | `download_dataset.py` names the Hugging Face dataset `bryandts/waste_organic_anorganic_classification` | 19,389 images under `dataset/DATASET/TRAIN/O` and `R` |

`docs/dataset_sources.md` records the TrashNet category selection: 482 Plastic,
501 Glass, 410 Metal, and 500 Paper images (299 Paper and 201 Cardboard source
images). It also records RealWaste as the intended Organic source and describes
about 411 Food Organics images. However, the actual local Organic files and
manifest contain only **5** images. The project does not include sufficient
per-image provenance to confirm that these five files came from RealWaste.
Update the source notes and manifest if the source or count changes.

The project’s `download_dataset.py` calls Hugging Face `snapshot_download()` and
prints the downloaded file paths. It does not itself copy, rename, or split the
downloaded files into the five-class training folders. The separate O/R files
are already present locally, but the project does not document their exact
placement or how they were split.

## Waste Categories and Class Labels

The class names were verified from both `src/config.py` and the saved
`model/best.pt` checkpoint:

| Model class ID | Label in the model | App display |
| ---: | --- | --- |
| 0 | `Can_Metal` | Can / Metal |
| 1 | `Glass` | Glass |
| 2 | `Organic` | Organic |
| 3 | `Paper` | Paper |
| 4 | `Plastic` | Plastic |

These are the five actual classes in the current classifier. The project has no
verified `Other`, Cardboard, or separate Glass/Can subcategory in this model.
Cardboard source images are combined into the `Paper` label per the project
source notes.

## Number of Images per Category

These counts are image files, not object instances or annotations:

| Category | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Can_Metal | 286 | 62 | 62 | 410 |
| Glass | 351 | 75 | 75 | 501 |
| Organic | 3 | 1 | 1 | 5 |
| Paper | 350 | 75 | 75 | 500 |
| Plastic | 338 | 72 | 72 | 482 |
| **Total** | **1,328** | **285** | **285** | **1,898** |

The separate O/R collection is not included in this table or the 1,898-image
classifier total.

## Dataset Folder Structure

The five-class dataset follows the folder-based layout used for image
classification:

```text
dataset/
├── train/
│   ├── Can_Metal/  (286 images)
│   ├── Glass/      (351 images)
│   ├── Organic/    (3 images)
│   ├── Paper/      (350 images)
│   └── Plastic/    (338 images)
├── val/
│   ├── Can_Metal/  (62 images)
│   ├── Glass/      (75 images)
│   ├── Organic/    (1 image)
│   ├── Paper/      (75 images)
│   └── Plastic/    (72 images)
├── test/
│   ├── Can_Metal/  (62 images)
│   ├── Glass/      (75 images)
│   ├── Organic/    (1 image)
│   ├── Paper/      (75 images)
│   └── Plastic/    (72 images)
└── DATASET/
    └── TRAIN/
        ├── O/      (12,565 images; separate collection)
        └── R/      (6,824 images; separate collection)
```

The original five-class images are organized under `raw_images/<Category>/`.
`docs/dataset_manifest.csv` records the prepared file name, category, split, and
original file name.

## Image Annotation and Label Format

The five-class dataset uses **folder labels**: the name of a class directory
(for example, `dataset/train/Plastic/`) is the image's class label. The class ID
mapping is stored in the YOLO checkpoint and listed above.

This classification dataset has no accompanying per-image bounding-box text
files, segmentation masks, or `data.yaml`. No object coordinates or other
object-level annotations were found. This is expected for the current
classification model: it predicts one class for an input image rather than
locating waste objects in an image.

## Training, Validation, and Testing Splits

The `train`, `val`, and `test` folders are present and their counts are shown
above. The preparation script documents a 70/15/15 split target, seed `42`, and
grouping rule: files whose names share the portion before `__` stay together.
Actual per-class counts are authoritative; rounding and group sizes can affect
the exact split sizes.

`scripts/train_model.py` passes the `dataset/` folder to YOLO for training and
prints counts for all three splits. The recorded run configuration
(`runs/classify/waste_sorter_20261008_132810/args.yaml`) shows training with
validation enabled and `split: val`. The training script does not run a test-set
evaluation. Therefore, `test/` is available as a holdout split, but its
performance has **Not yet been verified** by the current script.

The project has no `data.yaml`; class labels come from directory names and the
checkpoint. Do not include `dataset/DATASET/TRAIN/O` or `R` as current
five-class splits.

## Dataset Preparation and Preprocessing

Based on `scripts/prepare_dataset.py`, the documented process is:

1. Put source images in `raw_images/<Category>/`.
2. Check that required category folders contain images.
3. Check image bytes for exact duplicates within each raw category and verify
   that files can be opened by Pillow. Exact duplicates and unreadable images
   are skipped.
4. Shuffle image-name groups using seed `42` and assign images to train,
   validation, and test.
5. Convert accepted images to RGB and save them as JPEG files in the matching
   class and split folders.
6. Write a row for each prepared file to `docs/dataset_manifest.csv`.

The raw collection currently contains 1,898 images: 1,893 `.jpg` images from
the four documented TrashNet categories, plus five Organic images (three
`.jfif`, one `.jpg`, and one `.png`). The prepared splits contain JPEG files.
The recorded YOLO training configuration uses 224-pixel input images. Other
runtime resizing and augmentation behavior is handled by the installed
Ultralytics training pipeline; the project does not define a separate
preprocessing configuration file.

## How the Dataset Is Used by YOLO

`scripts/train_model.py` loads the pretrained `yolo11n-cls.pt` classification
checkpoint and trains it using `dataset/`. YOLO uses the subfolder names under
`train/` as class labels and the `val/` folder for validation. The script's
defaults are 30 epochs and a batch size of 32. The project contains
`model/best.pt`, and loading that checkpoint verified that it is a classifier
with the five class labels listed above.

The project also has `yolo11n.pt` for a separate pretrained person detector
used by the application. It is not the five-class waste model and does not
change the waste dataset labels. The saved training run includes configuration,
results, and plots, but this document does not claim an evaluation result.

## Dataset Quality and Limitations

### Verified from the files and scripts

- The dataset has a severe class imbalance: Organic has 5 images, while each
  other class has 410–501. Organic has only 3 training images and 1 image in
  each evaluation split.
- Organic's local count (5) is far below the about-411 count recorded in
  `docs/dataset_sources.md`.
- The preparation script checks exact byte duplicates within each raw category
  and verifies image readability when the script is run. The manifest and
  current folder counts agree, but this does not prove that all current files
  were produced by the latest script run.
- The five-class data is a classification dataset without bounding-box
  annotations.
- The project does not record an image-by-image review of visual quality,
  lighting, background variety, object visibility, or label correctness.
- The test set exists, but the current training script does not evaluate the
  model on it.

### Potential issues that still need checking

- Similar or near-duplicate images may exist even when their file bytes differ.
  Duplicate detection is limited to exact matches within each raw category;
  it does not compare across categories or use visual similarity.
- Similar views of the same physical item could appear in different splits if
  their filenames do not use the documented `__` grouping convention.
- The small and uneven sample count may cause the model to favor the larger
  classes and perform poorly on Organic waste.
- Image lighting, camera distance, backgrounds, clutter, and the appearance of
  dirty, crushed, or partly hidden items may not match live camera conditions.
  These conditions have not been measured from the files.
- Source licenses and citations should be checked against the source records
  before sharing the dataset or project outside its permitted use.

## Recommendations for Improvement

1. Add many more varied Organic images. The current 3/1/1 train/validation/test
   split is too small for reliable training or evaluation.
2. Review every class label and image visually, remove corrupt or ambiguous
   images, and document the review. Check near-duplicates and ensure images of
   the same physical item or capture session remain in only one split.
3. Collect examples that match the intended camera use, including varied
   lighting, backgrounds, distances, orientations, occlusion, and clean and
   dirty items.
4. Keep the existing test set separate from training and model selection. Add a
   documented test-evaluation step after training, and report per-class
   precision/recall or a confusion matrix along with the overall result.
5. Keep counts in `docs/dataset_manifest.csv` and `docs/dataset_sources.md`
   synchronized with the files. Resolve the documented ~411 RealWaste Organic
   count versus the 5 local Organic files, and record source, license, and
   attribution for each added collection.
6. If using the separate O/R download, first verify what `O` and `R` mean and
   whether its images are suitable for this model. Do not mix it into the
   five-class splits without deliberate relabeling and validation.
7. Consider a `data.yaml` only if the project moves to a detection dataset.
   For the present classification setup, maintain the documented directory
   names and checkpoint label mapping instead.

## Conclusion

The active waste classifier has five verified labels and 1,898 images arranged
in train, validation, and test folders. Counts and splits are available in the
dataset manifest and summarized here. The main gap is the Organic class, which
has only five local images despite a larger count in the source notes. The
separate 19,389-image O/R collection is not used by the current five-class
training flow. Improve and review the Organic data, confirm provenance, and
evaluate the model once on the held-out test split before relying on its
predictions.

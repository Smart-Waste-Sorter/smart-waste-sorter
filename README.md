# Smart Waste Sorter

This project is a Python 3.11+ application for classifying waste items from a webcam and suggesting the correct bin.

## Project structure

- `main.py` – application entry point
- `src/` – classifier, camera, UI, statistics, stabilizer, and sorting logic
- `model/` – trained YOLO model files
- `dataset/` – train/val/test data, one folder per category
- `data/` – SQLite database
- `tests/` – unit tests
- `docs/` – project docs and evaluation outputs

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Prepare a complete dataset. The app sorts `Plastic`, `Paper`, `Can_Metal`,
   `Glass`, and `Organic`. An `Other` rejection class is optional; if used,
   provide it in train, validation, and test with non-waste examples. The
   existing split folders already contain many images for the first four
   categories, but only 3/1/1 Organic images in train/validation/test. Add
   more varied Organic examples directly
   to `dataset\train\Organic`, `dataset\val\Organic`, and
   `dataset\test\Organic`. Aim for at least 50 training and 10
   validation and test images in every class; include faces and people framed
   similarly to how the camera will be used. Keep photos of the same physical
   item/person in only one split.

   To build all splits from scratch instead, put labeled photos in
   `raw_images\<Category>\` and run:

   ```bash
   python scripts\prepare_dataset.py
   ```

   This rebuilds the category `.jpg` files under `dataset\train`,
   `dataset\val`, and `dataset\test`. It removes existing `.jpg` files in
   those split folders, so make sure `raw_images` contains all five waste
   classes (`Plastic`, `Paper`, `Can_Metal`, `Glass`, and `Organic`) first.
   Add `Other` there only if you want to train that rejection class too. The
   script stops without deleting files when a required waste class is missing.
   It does not use the separate downloaded
   `dataset\DATASET\TRAIN` two-class O/R dataset.

3. Train the classifier:

   ```bash
   python scripts\train_model.py
   ```

   The first run downloads pretrained YOLO classification and person-detection
   weights and requires an internet connection. The script trains on
   `dataset\`, then copies the best classifier to `model\best.pt` and the
   person detector to `model\person_detector.pt`. The detector prevents a
   person in the scan box from being sorted. It reports sparse classes; aim
   for at least 50 training images and 10 validation and test images per
   class. You can change the training length with `--epochs 50`.

4. Start or restart the app with the project interpreter:

   ```bash
   .venv\Scripts\python.exe main.py
   ```

   On startup, the terminal prints the interpreter path. It should end in
   `smart_waste\.venv\Scripts\python.exe`. Close any previously running app
   window before restarting; an already-open process keeps its original
   interpreter.

   A live camera preview only confirms that the camera works. Scanning is
   enabled after the waste classifier loads. If
   `model\person_detector.pt` is missing, the app loads the official pretrained
   `yolo11n.pt` detector automatically from the project folder. Internet is
   needed for the first download. Person detection now checks the full camera
   frame so a close-up face is not lost by the smaller waste-classification
   crop; a person blocks classification only when the detected box overlaps the
   green scan box. The dashboard should show `READY` instead of `MODEL NEEDED`.
   Keep your face out of the green scan box and place one waste item inside it.

5. Voice announcements are enabled by default. The app uses offline
   `pyttsx3` speech to say the confirmed category, recommended bin, and
   category-specific recycling guidance. Disable **Voice announcements** in
   Settings to run silently. Install `pyttsx3` with:

   ```bash
   .venv\Scripts\python.exe -m pip install pyttsx3
   ```

   It is also included in `requirements.txt`; install all dependencies with
   `.venv\Scripts\python.exe -m pip install -r requirements.txt`. To test the
   computer voice without starting the camera, YOLO, or Arduino, run:

   ```bash
   .venv\Scripts\python.exe scripts\check_voice.py
   ```

   A successful command means the speech engine completed its request; listen
   for the sentence to confirm that the computer's speakers actually produced
   sound. In Settings, check that **Voice announcements** is on, adjust
   volume and speech speed, and press **Test sound**.
   These voice preferences are saved in `data\settings.json`.

   **Troubleshooting:** If Settings says `pyttsx3 is missing` or
   `Voice announcements unavailable`, start the app with the project
   interpreter and install dependencies into that same environment:

   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   .venv\Scripts\python.exe -c "import sys; print('Python:', sys.executable); import pyttsx3; e = pyttsx3.init(); e.say('test'); e.runAndWait()"
   .venv\Scripts\python.exe main.py
   ```

   The diagnostic prints the interpreter used for the speech test. It should
   end in `smart_waste\.venv\Scripts\python.exe`. The VS Code launch
   configuration uses this interpreter too. If the speech test succeeds but
   the app still reports a missing module, close the app and use the command
   above to restart it from the project folder. Voice controls remain disabled
   if the speech engine cannot initialize, and the reason is shown in Settings.

   To verify the Settings **Test sound** button, open Settings, check that
   **Voice announcements** is on, set Volume above 0%, then press **Test sound**. You should
   hear the sample Paper announcement. For end-to-end verification, return to
   the Dashboard, hold a paper item steadily inside the green box until
   `LOCKED · ITEM CONFIRMED` appears, and listen for the Paper recycling-bin
   announcement. Move the item out of view before testing another class.

## Voice announcements

Voice announcements are on by default. The app announces only after three
matching inference results confirm and lock an item. It speaks the confirmed
category and confidence first, then the recommended bin and careful
recycling/composting guidance. Both messages are queued as one ordered batch
on the speech worker, so they do not overlap or block the camera. After
confirmation, new inference pauses for 0.8 seconds while the
camera preview and UI continue updating. The classifier is a
single-label classifier, not an item-presence detector: removing an item is
recognized only if its next prediction changes (for example, to `UNKNOWN`).
After that change resets the lock, another item can be confirmed and announced.
A configurable two-second cooldown prevents rapid repeat messages, and a
background worker keeps speech from blocking the camera or UI. UNKNOWN
and below-threshold predictions get only the manual-check sentence, with no
spoken bin recommendation. A class missing from the mapping is announced as
`Item detected: <name>. No bin recommendation available.`
The `Other`/non-waste label is explicitly treated as unrecognized and never
receives a bin recommendation. For mapped categories, the announcement names
the item, says whether it is
recyclable (or compostable), and recommends its bin. Plastic and paper are
not promised to be accepted everywhere: the voice advises checking local
acceptance and, for paper, keeping it clean and dry. Metal cans are described
as generally recyclable and should be emptied first. Organic waste is
identified as compostable when suitable facilities are available. Glass is
also qualified by local acceptance.

The current model's actual labels are `Can_Metal`, `Glass`, `Organic`,
`Paper`, and `Plastic`; all five have entries in `WASTE_INFO` in
[`src/config.py`](./src/config.py). There are no missing model classes.
Cardboard, `Other`, general trash, battery, and e-waste entries are examples
for possible future model classes; the current model does not announce these
as detected categories. In particular, the current `Other` model class is a
rejection class that includes empty scenes and non-waste objects. The
classifier deliberately blocks it from bin recommendations. Do not route
`Other` to the general waste bin unless the model has been retrained using
only actual waste for that class and its label behavior has been validated.

To customize a class recommendation, edit that class's mapping entry in
`src/config.py`. Each entry contains a `display_name`, recycling status
(`recyclable` is `True`, `False`, or `None` when local/material acceptance
is unknown), `bin_name`, `bin_color`, and optional `note`. The configured
`VOICE_DETECTION_TEMPLATE`, `VOICE_CONFIDENCE_TEMPLATE`, and
`VOICE_GUIDANCE_TEMPLATE` control the spoken detection and sorting advice;
related constants control unknown, unmapped, error, and test messages.
`DEFAULT_VOICE_COOLDOWN_SECONDS` controls the minimum interval between
announcements. The Settings page saves enabled state, volume, speech rate,
selected voice, and the optional confidence preference in
`data\settings.json`.

There is no Arduino sketch, serial communication, or servo sorting protocol
in this project, so it cannot verify that a physical sort has completed. The
voice announcer provides `announce_sorting_complete()` for a future hardware
callback, but it is intentionally not called by this app. Add the Arduino
firmware and its documented completion response before connecting that exact
response to the method. Detection announcements are not a substitute for
hardware-confirmed completion.

## Notes

- The app looks for the trained classifier and person detector at
  `model\best.pt` and `model\person_detector.pt`.
- The current `dataset\train`, `dataset\val`, and `dataset\test` folders contain
  very few `Organic` images. Collect more examples before relying on the
  predictions; the separate downloaded O/R dataset does not provide the five
  app categories.
- The project follows the requirements for a single-item waste sorting system with category recognition and bin rules.

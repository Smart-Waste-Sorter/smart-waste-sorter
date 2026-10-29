# Build the Smart Waste Sorter Windows app

This project can be packaged as one Windows executable:

```text
dist\SmartWasteSorter.exe
```

The EXE contains the Python runtime, application dependencies, the waste
classifier (`model\best.pt`), and the person detector (`yolo11n.pt`). The app
does not download either model when it runs. Training data, saved settings,
and history are not bundled.

## What you need

- A 64-bit Windows 10 or Windows 11 computer to build the app.
- The project files, including both model files listed above.
- The project `.venv` and its dependencies. The checked project environment
  uses Python 3.11.9; use 64-bit Python 3.11 if you need to create it again.
- A webcam to test live camera detection.
- Installed Windows speech voices for spoken guidance. The Python voice
  package does not include Windows system voices.

The app also includes the existing category classification, confidence
threshold and stable-result logic, sorting advice, spoken guidance, statistics,
and saved history.

## Build the one-file EXE

Open a terminal in the project folder. Install the project packages if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` does not exist, create it using 64-bit Python 3.11 and install the
packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Build the console-enabled debug EXE:

```powershell
.\build_exe.bat
```

The batch file checks the two model files and required Python imports, then
uses the one-file spec. It does not delete source files, models, datasets, or
user data. The build can take several minutes and needs extra temporary disk
space because PyTorch is included.

The exact PyInstaller command used by the batch file is:

```powershell
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --distpath ".\dist" --workpath ".\build\SmartWasteSorter-onefile" ".\SmartWasteSorter.spec"
```

When it succeeds, the standalone file is:

```text
dist\SmartWasteSorter.exe
```

This build intentionally keeps a console window visible so startup, model,
camera, and voice errors can be diagnosed. Do not switch to a windowless build
until the debug EXE has been tested on the intended hardware.

## Test the app

Run the EXE from PowerShell so you can read its diagnostic messages:

```powershell
.\dist\SmartWasteSorter.exe
```

Check these features:

1. **Window and models:** Confirm the app opens and the Model info page reports
   the classifier loaded. The console should not report missing model files.
2. **Camera:** Allow camera access in Windows privacy settings. Confirm the
   live preview appears; try the retry control if the camera is busy.
3. **Detection:** Hold one supported waste item in the scan area. Confirm the
   category and confidence appear after the stable-result checks.
4. **Sorting advice:** Check the displayed bin/recycling or compost guidance
   for the recognized category.
5. **Voice:** In Settings, enable voice, raise the volume, and use the test
   sound. Then classify an item and listen for detection and sorting guidance.
   Windows speech voices and an audio output device are required.
6. **History and statistics:** Classify a few items, inspect those pages, close
   and reopen the app, and verify history remains.
7. **Shutdown:** Close the app normally. Check that the camera indicator turns
   off and the process exits.

This workspace cannot prove that a webcam or Windows speech voice is available
on another computer. Test those hardware-dependent features on the target
computer. The classifier models are included locally, so model inference does
not require an internet connection.

## Where settings and history are saved

When packaged, user settings and the SQLite history are saved under:

```text
%LOCALAPPDATA%\SmartWasteSorter\
```

They are not written into the EXE or its temporary extraction directory.

## Copy and run on another computer

Copy only `dist\SmartWasteSorter.exe` to a compatible 64-bit Windows computer
and double-click it. Python, VS Code, and the project's Python packages do not
need to be installed there. Windows camera permission, a compatible camera
driver, speaker/audio output, and an installed Windows speech voice are still
needed for those features.

`--onefile` does not guarantee compatibility with every Windows release or
computer. Build on Windows, then test on a clean compatible Windows computer.
The first launch may take longer while the EXE extracts its contents, and
security software may scan or quarantine a large executable.

## Troubleshooting

- **Build reports a missing model:** Check that `model\best.pt` and
  `yolo11n.pt` exist at those exact project paths.
- **Build reports a missing import:** Run the pip install command above with
  `.venv\Scripts\python.exe`, then retry the batch file.
- **Camera is unavailable:** Read the `[CAMERA]` diagnostic in the console,
  close other camera apps, and check Windows camera privacy settings.
- **Voice is unavailable or silent:** Read the `[VOICE]` diagnostic, enable
  voice in Settings, check the selected speaker and volume, and confirm a
  Windows SAPI speech voice is installed. A system speech voice cannot be
  bundled into this EXE.
- **The app fails to start:** Run the EXE from PowerShell and keep the full
  console output. This debug build is intentionally console-enabled.

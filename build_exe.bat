@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "APP=%ROOT%dist\SmartWasteSorter.exe"
set "WORK=%ROOT%build\SmartWasteSorter-onefile"

if not exist "%PYTHON%" (
    echo ERROR: The project Python environment was not found:
    echo   %PYTHON%
    echo Create it with: python -m venv .venv
    echo Then install packages with: .venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

pushd "%ROOT%"
echo Checking required model files...
if not exist "%ROOT%model\best.pt" (
    echo ERROR: The waste classifier model is missing: %ROOT%model\best.pt
    popd
    exit /b 1
)
if not exist "%ROOT%yolo11n.pt" (
    echo ERROR: The person detector model is missing: %ROOT%yolo11n.pt
    popd
    exit /b 1
)

echo Checking Python, OpenCV, PyTorch, YOLO, voice, and packaging dependencies...
"%PYTHON%" -c "import cv2, PIL, torch, torchvision, ultralytics, pyttsx3, comtypes.client, win32com.client, tkinter, PyInstaller"
if errorlevel 1 (
    echo ERROR: One or more build dependencies are missing or could not be imported.
    echo Install the project dependencies with:
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    popd
    exit /b 1
)

if not exist "%ROOT%dist" mkdir "%ROOT%dist"
if errorlevel 1 (
    echo ERROR: Could not create the dist output folder.
    popd
    exit /b 1
)

echo Building the one-file, console-enabled debug application.
echo This can take several minutes and needs substantial free disk space.
"%PYTHON%" -m PyInstaller --noconfirm --clean --distpath "%ROOT%dist" --workpath "%WORK%" "%ROOT%SmartWasteSorter.spec"
if errorlevel 1 (
    echo ERROR: PyInstaller failed. Review the diagnostic output above.
    popd
    exit /b 1
)

if not exist "%APP%" (
    echo ERROR: PyInstaller completed but the expected executable was not found:
    echo   %APP%
    popd
    exit /b 1
)

echo.
echo Build complete:
echo   %APP%
echo This console-enabled build keeps startup and model errors visible.
echo Copy only this EXE to a compatible 64-bit Windows computer.
popd
exit /b 0

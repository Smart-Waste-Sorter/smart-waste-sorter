import sys
from types import ModuleType

from src.camera import CameraManager


class _Capture:
    def __init__(self, *, opened=True, frame_available=True):
        self.opened = opened
        self.frame_available = frame_available
        self.release_count = 0

    def isOpened(self):
        return self.opened

    def read(self):
        return self.frame_available, "frame" if self.frame_available else None

    def release(self):
        self.release_count += 1


def test_camera_opens_and_releases_capture(monkeypatch):
    capture = _Capture()
    fake_cv2 = ModuleType("cv2")
    fake_cv2.CAP_DSHOW = 1
    fake_cv2.CAP_MSMF = 2
    fake_cv2.CAP_ANY = 0
    fake_cv2.VideoCapture = lambda _index, _backend: capture
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    camera = CameraManager()

    assert camera.open()
    assert camera.available
    assert camera.read_frame() == "frame"
    assert capture.release_count == 0

    camera.close()

    assert not camera.available
    assert capture.release_count == 1


def test_camera_reports_failure_and_releases_failed_captures(monkeypatch, capsys):
    captures = []
    fake_cv2 = ModuleType("cv2")
    fake_cv2.CAP_DSHOW = 1
    fake_cv2.CAP_MSMF = 2
    fake_cv2.CAP_ANY = 0

    def open_capture(_index, _backend):
        capture = _Capture(opened=False)
        captures.append(capture)
        return capture

    fake_cv2.VideoCapture = open_capture
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    camera = CameraManager()

    assert not camera.open()
    assert not camera.available
    assert "could not open the camera" in camera.last_error
    assert all(capture.release_count == 1 for capture in captures)
    assert "[CAMERA] ERROR:" in capsys.readouterr().out


def test_camera_reports_read_failure(monkeypatch, capsys):
    capture = _Capture(frame_available=False)
    camera = CameraManager()
    camera.capture = capture

    assert camera.read_frame() is None
    assert camera.last_error == "The camera stopped returning frames."
    assert "[CAMERA] ERROR:" in capsys.readouterr().out

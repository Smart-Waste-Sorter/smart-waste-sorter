from __future__ import annotations

import sys


class CameraManager:
    """Wrap camera operations and provide a safe interface for the app."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture = None
        self.last_error: str | None = None

    @staticmethod
    def _backends(cv2) -> list:
        """Ways to open the camera, best first. DirectShow is fastest on Windows."""
        if sys.platform.startswith("win"):
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    def open(self):
        """Attempt to open the webcam. Return True only if it gives a real picture."""
        self.last_error = None
        try:
            import cv2
        except Exception as exc:
            self.last_error = f"OpenCV could not be imported: {exc}"
            print(f"[CAMERA] ERROR: {self.last_error}", flush=True)
            return False

        errors = []
        for backend in self._backends(cv2):
            capture = None
            try:
                capture = cv2.VideoCapture(self.camera_index, backend)
                if capture.isOpened():
                    ok, _frame = capture.read()
                    if ok:
                        self.capture = capture
                        print(
                            f"[CAMERA] Opened camera {self.camera_index} "
                            f"with backend {backend}.",
                            flush=True,
                        )
                        return True
                    errors.append(f"backend {backend} opened but returned no frame")
                else:
                    errors.append(f"backend {backend} could not open the camera")
            except Exception as exc:
                errors.append(f"backend {backend}: {exc}")
            finally:
                if capture is not None and capture is not self.capture:
                    try:
                        capture.release()
                    except Exception as exc:
                        errors.append(f"backend {backend} release failed: {exc}")

        self.capture = None
        self.last_error = "; ".join(errors) or "No camera backend could be tried."
        print(
            f"[CAMERA] ERROR: Camera {self.camera_index} is unavailable: "
            f"{self.last_error}",
            flush=True,
        )
        return False

    def read_frame(self):
        """Read one frame from the camera if available."""
        if self.capture is None:
            return None

        try:
            ret, frame = self.capture.read()
            if not ret:
                self.last_error = "The camera stopped returning frames."
                print(f"[CAMERA] ERROR: {self.last_error}", flush=True)
                return None
            return frame
        except Exception as exc:
            self.last_error = f"Camera frame read failed: {exc}"
            print(f"[CAMERA] ERROR: {self.last_error}", flush=True)
            return None

    def close(self) -> None:
        capture = self.capture
        self.capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception as exc:
                self.last_error = f"Camera release failed: {exc}"
                print(f"[CAMERA] ERROR: {self.last_error}", flush=True)

    @property
    def available(self) -> bool:
        return self.capture is not None
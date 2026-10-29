from __future__ import annotations

import cv2
import numpy as np


def resize_frame(frame, width: int = 224, height: int = 224):
    """Resize a frame to the size required by the image classifier."""
    if frame is None:
        raise ValueError("Frame is empty and cannot be processed.")
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def normalize_frame(frame):
    """Convert a frame to a consistent numeric format."""
    if frame is None:
        raise ValueError("Frame is empty and cannot be normalized.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

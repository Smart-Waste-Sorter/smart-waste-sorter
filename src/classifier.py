from __future__ import annotations

from pathlib import Path

from src.config import MODEL_PATH
from src.paths import resource_path


class WasteClassifier:
    """This placeholder is the AI prediction layer for the waste sorter."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = str(
            model_path if model_path is not None else resource_path(MODEL_PATH)
        )
        self.model = None
        self.person_detector = None
        self.labels = []

    def load_model(self):
        """Load the waste classifier and a pretrained person detector."""
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        detector_path = model_path.with_name("person_detector.pt")
        default_detector = model_path.parent.parent / "yolo11n.pt"
        detector_model = detector_path if detector_path.is_file() else default_detector
        if not detector_model.is_file():
            raise FileNotFoundError(
                f"Person detector model not found: {detector_model}. "
                "Include yolo11n.pt with the application; model files are never "
                "downloaded automatically."
            )

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(model_path))
            self.person_detector = YOLO(str(detector_model))
            return self.model
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load the waste classifier or person detector: {exc}. "
                "Check that both bundled model files are valid."
            ) from exc

    def predict(
        self,
        frame,
        *,
        person_detection_frame=None,
        person_detection_region: tuple[int, int, int, int] | None = None,
    ):
        """Reject people in the scan area before classifying its waste crop."""
        if self.model is None or self.person_detector is None:
            raise RuntimeError("Models have not been loaded yet.")

        detection_frame = (
            frame if person_detection_frame is None else person_detection_frame
        )
        detections = self.person_detector(detection_frame, verbose=False)
        for detection in detections:
            if detection.boxes is None:
                continue
            for box in detection.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                if detection.names.get(class_id) == "person" and confidence >= 0.5:
                    if person_detection_region is not None:
                        box_x1, box_y1, box_x2, box_y2 = box.xyxy[0].tolist()
                        region_x1, region_y1, region_x2, region_y2 = (
                            person_detection_region
                        )
                        if (
                            max(region_x1, box_x1) >= min(region_x2, box_x2)
                            or max(region_y1, box_y1) >= min(region_y2, box_y2)
                        ):
                            continue
                    return {
                        "label": "UNKNOWN",
                        "confidence": 0.0,
                        "blocked_reason": "A person was detected in the scan area.",
                    }

        results = self.model(frame, verbose=False)
        if not results:
            return {"label": "UNKNOWN", "confidence": 0.0}

        prediction = results[0]
        if prediction.probs is None:
            return {"label": "UNKNOWN", "confidence": 0.0}
        class_name = prediction.names.get(prediction.probs.top1, "UNKNOWN")
        confidence = float(prediction.probs.top1conf)
        if class_name.strip().lower() in {"other", "non_waste", "nonwaste"}:
            return {
                "label": "UNKNOWN",
                "confidence": confidence,
                "blocked_reason": "No waste category was recognized.",
            }
        return {"label": class_name, "confidence": confidence}

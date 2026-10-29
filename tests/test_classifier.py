import sys
from types import ModuleType

import pytest

from src.classifier import WasteClassifier


class _TensorValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return self.value


class _Box:
    def __init__(self, class_id, confidence, coordinates=None):
        self.cls = _TensorValue(class_id)
        self.conf = _TensorValue(confidence)
        if coordinates is not None:
            self.xyxy = [_TensorValue(coordinates)]


class _Detection:
    def __init__(self, boxes, names=None):
        self.boxes = boxes
        self.names = names or {0: "person"}


class _PersonDetector:
    def __init__(self, detections):
        self.detections = detections
        self.last_frame = None

    def __call__(self, frame, verbose=False):
        self.last_frame = frame
        return self.detections


class _WasteModel:
    def __init__(self, label):
        self.called = False
        self.label = label

    def __call__(self, frame, verbose=False):
        self.called = True
        return [
            type(
                "Classification",
                (),
                {
                    "names": {0: self.label},
                    "probs": type(
                        "Probabilities",
                        (),
                        {"top1": 0, "top1conf": 0.98},
                    )(),
                },
            )()
        ]


def test_default_model_path_is_set():
    classifier = WasteClassifier()
    assert classifier.model_path.endswith("best.pt")


def test_model_not_loaded_until_requested():
    classifier = WasteClassifier()
    assert classifier.model is None


def test_person_detection_blocks_waste_classifier():
    classifier = WasteClassifier()
    classifier.person_detector = _PersonDetector(
        [_Detection([_Box(class_id=0, confidence=0.98)])]
    )
    classifier.model = _WasteModel("Plastic")

    result = classifier.predict(object())

    assert result["blocked_reason"]
    assert result["label"] == "UNKNOWN"
    assert not classifier.model.called


def test_person_in_scan_area_is_detected_using_full_camera_frame():
    camera_frame = object()
    item_crop = object()
    classifier = WasteClassifier()
    classifier.person_detector = _PersonDetector(
        [_Detection([_Box(0, 0.89, coordinates=[10, 10, 40, 40])])]
    )
    classifier.model = _WasteModel("Plastic")

    result = classifier.predict(
        item_crop,
        person_detection_frame=camera_frame,
        person_detection_region=(20, 20, 50, 50),
    )

    assert classifier.person_detector.last_frame is camera_frame
    assert result["blocked_reason"] == "A person was detected in the scan area."
    assert result["label"] == "UNKNOWN"
    assert not classifier.model.called


def test_person_outside_scan_area_does_not_block_waste_classification():
    classifier = WasteClassifier()
    classifier.person_detector = _PersonDetector(
        [_Detection([_Box(0, 0.89, coordinates=[0, 0, 10, 10])])]
    )
    classifier.model = _WasteModel("Plastic")

    result = classifier.predict(
        object(),
        person_detection_frame=object(),
        person_detection_region=(20, 20, 50, 50),
    )

    assert result["label"] == "Plastic"
    assert classifier.model.called


def test_other_class_is_returned_as_not_a_waste_category():
    classifier = WasteClassifier()
    classifier.person_detector = _PersonDetector([_Detection([])])
    classifier.model = _WasteModel("Other")

    result = classifier.predict(object())

    assert result["blocked_reason"] == "No waste category was recognized."
    assert result["label"] == "UNKNOWN"


def test_load_model_uses_pretrained_detector_when_local_copy_is_missing(
    tmp_path, monkeypatch
):
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    classifier_path = model_directory / "best.pt"
    detector_path = tmp_path / "yolo11n.pt"
    classifier_path.touch()
    detector_path.touch()
    loaded_paths = []

    def fake_yolo(path):
        loaded_paths.append(path)
        return object()

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = fake_yolo
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    classifier = WasteClassifier(model_path=classifier_path)
    classifier.load_model()

    assert loaded_paths == [
        str(classifier_path),
        str(detector_path),
    ]
    assert classifier.model is not None
    assert classifier.person_detector is not None


def test_load_model_does_not_download_a_missing_person_detector(tmp_path, monkeypatch):
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    classifier_path = model_directory / "best.pt"
    classifier_path.touch()
    yolo_called = False

    def fake_yolo(_path):
        nonlocal yolo_called
        yolo_called = True
        return object()

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = fake_yolo
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    classifier = WasteClassifier(model_path=classifier_path)

    with pytest.raises(FileNotFoundError, match="never downloaded automatically"):
        classifier.load_model()

    assert not yolo_called

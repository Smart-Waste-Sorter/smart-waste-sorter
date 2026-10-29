from __future__ import annotations


class ConfidencePolicy:
    """Simple policy wrapper to decide whether a model result is high enough."""

    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def is_confident(self, confidence: float) -> bool:
        return float(confidence) >= float(self.threshold)

    def label_with_threshold(self, label: str, confidence: float):
        if not self.is_confident(confidence):
            return "UNKNOWN", confidence
        return label, confidence

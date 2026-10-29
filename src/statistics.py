from __future__ import annotations

from collections import Counter


class StatisticsTracker:
    """Track totals, per-category counts, and confidence summaries."""

    def __init__(self):
        self.counts = Counter()
        self.confidences = []

    def add_result(self, category: str, confidence: float):
        self.counts[category] += 1
        self.confidences.append(float(confidence))

    def total(self) -> int:
        return sum(self.counts.values())

    def unknown_count(self) -> int:
        return self.counts.get("UNKNOWN", 0)

    def average_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)

    def percentages(self):
        total = self.total()
        if total == 0:
            return {}
        return {category: (count / total) * 100.0 for category, count in self.counts.items()}

    def reset(self):
        self.counts.clear()
        self.confidences.clear()

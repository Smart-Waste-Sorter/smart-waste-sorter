from __future__ import annotations


class Stabilizer:
    """Prevent duplicate counts by requiring several matching frames."""

    def __init__(self, required_matches: int = 3):
        self.required_matches = required_matches
        self.current_label = None
        self.match_count = 0
        self.is_locked = False

    def reset(self):
        self.current_label = None
        self.match_count = 0
        self.is_locked = False

    def update(self, label: str):
        """Return True when the label has been confirmed enough times."""
        if self.is_locked:
            return True

        if self.current_label == label:
            self.match_count += 1
        else:
            self.current_label = label
            self.match_count = 1

        if self.match_count >= self.required_matches:
            self.is_locked = True
            return True

        return False

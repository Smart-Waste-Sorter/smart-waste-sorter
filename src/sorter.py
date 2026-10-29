from __future__ import annotations

from src.config import BIN_RULES


def recommended_bin(category: str, confidence: float = 0.0, threshold: float = 0.60) -> str:
    """Map a category to the correct waste bin or return unknown."""
    if not category or category == "UNKNOWN":
        return BIN_RULES["UNKNOWN"]

    if confidence < threshold:
        return BIN_RULES["UNKNOWN"]

    return BIN_RULES.get(category, BIN_RULES["UNKNOWN"])

from src.sorter import recommended_bin


def test_recommended_bin_for_known_category():
    assert recommended_bin("Plastic", 0.99, 0.60) == (
        "Yellow plastic bin / Check local recycling acceptance"
    )


def test_organic_result_uses_compost_label():
    assert recommended_bin("Organic", 0.99, 0.60) == (
        "Green organic bin / Organic or compostable"
    )


def test_low_confidence_returns_unknown():
    assert recommended_bin("Plastic", 0.20, 0.60) == (
        "Unknown / check the material and disposal guidance"
    )


def test_unknown_label_returns_unknown():
    assert recommended_bin("UNKNOWN", 0.99, 0.60) == (
        "Unknown / check the material and disposal guidance"
    )

from src.statistics import StatisticsTracker


def test_statistics_tracker_counts_and_average_confidence():
    tracker = StatisticsTracker()
    tracker.add_result("Plastic", 0.90)
    tracker.add_result("Paper", 0.70)

    assert tracker.total() == 2
    assert tracker.average_confidence() == 0.8
    assert tracker.percentages()["Plastic"] == 50.0


def test_reset_clears_data():
    tracker = StatisticsTracker()
    tracker.add_result("Organic", 0.80)
    tracker.reset()

    assert tracker.total() == 0
    assert tracker.average_confidence() == 0.0

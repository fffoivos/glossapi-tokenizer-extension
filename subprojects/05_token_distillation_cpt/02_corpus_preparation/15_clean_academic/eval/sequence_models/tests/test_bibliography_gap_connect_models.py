import numpy as np

from sequence_models.bibliography_gap_connect_models import best_safety_threshold


def test_safety_threshold_prefers_recall_subject_to_zero_false_connects() -> None:
    targets = np.asarray([0, 0, 1, 1, 1], dtype=np.uint8)
    probability = np.asarray([0.1, 0.6, 0.55, 0.7, 0.9], dtype=np.float32)
    result = best_safety_threshold(
        targets, probability, max_false_connect_rate=0.0,
    )
    assert result["false_connect_count"] == 0
    assert result["threshold"] > 0.6
    assert np.isclose(result["connect_recall"], 2 / 3)

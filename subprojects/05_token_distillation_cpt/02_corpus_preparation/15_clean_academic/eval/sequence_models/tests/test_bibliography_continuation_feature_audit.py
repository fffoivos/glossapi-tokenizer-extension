import numpy as np

from sequence_models.bibliography_continuation_feature_audit import (
    deterministic_profiles,
    numeric_summary,
)


def test_numeric_summary_reports_robust_quantiles() -> None:
    report = numeric_summary(np.asarray([0.0, 1.0, 2.0, 3.0, np.nan]))
    assert report["count"] == 4
    assert report["mean"] == 1.5
    assert report["median"] == 1.5
    assert report["minimum"] == 0.0
    assert report["maximum"] == 3.0


def test_deterministic_profiles_compare_continuation_to_other_roles() -> None:
    names = ("presence:year_count", "log1p:year_count")
    raw_counts = np.asarray([1.0, 2.0, 0.0, 1.0, 0.0], dtype=np.float32)
    features = np.column_stack((raw_counts > 0, np.log1p(raw_counts))).astype(np.float32)
    role_rows = {
        "CONTINUATION": np.asarray([0, 1]),
        "ENTRY": np.asarray([2]),
        "FILLER": np.asarray([3]),
        "OTHER": np.asarray([4]),
    }
    report = deterministic_profiles(features=features, names=names, role_rows=role_rows)
    assert len(report) == 1
    assert report[0]["feature"] == "year_count"
    assert report[0]["continuation_presence_rate"] == 1.0
    assert np.isclose(report[0]["continuation_mean_count"], 1.5)
    assert report[0]["continuation_minus_filler_presence"] == 0.0
    assert report[0]["continuation_minus_other_presence"] == 1.0

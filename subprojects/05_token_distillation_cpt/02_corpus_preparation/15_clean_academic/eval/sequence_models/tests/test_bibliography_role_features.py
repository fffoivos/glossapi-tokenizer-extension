from __future__ import annotations

import numpy as np

from sequence_models.bibliography_positional_features import FEATURE_NAMES, GAP_SUMMARY_NAMES
from sequence_models.bibliography_role_features import (
    HEADING_PROBABILITY_NAMES,
    LINE_SHAPE_NAMES,
    broad_heading_candidate,
    candidate_window_mask,
    connector_feature_names,
    connector_feature_row,
    line_shape,
    p0d_matrix,
)


def test_shape_features_are_finite_for_blank_and_polytonic_greek() -> None:
    for text in ("", "   ", "ΒΙΒΛΙΟΓΡΑΦΊΑ", "Ἑλληνόγλωσση βιβλιογραφία", "§ — 12"):
        values = line_shape(text)
        assert values.shape == (len(LINE_SHAPE_NAMES),)
        assert np.isfinite(values).all()


def test_broad_heading_candidates_cover_exact_and_generic_headings() -> None:
    assert broad_heading_candidate("ΒΙΒΛΙΟΓΡΑΦΙΑ")
    assert broad_heading_candidate("2. Primary Sources")
    assert broad_heading_candidate("Κεφάλαιο 4", previous_blank=True)
    assert not broad_heading_candidate(
        "This is an ordinary sentence with terminal punctuation.",
        previous_blank=True,
    )


def test_candidate_windows_merge_and_respect_physical_gaps() -> None:
    probability = np.zeros(12, dtype=np.float32)
    probability[[2, 7]] = 0.8
    headings = np.zeros(12, dtype=bool)
    abs_indices = np.asarray([0, 1, 2, 3, 4, 100, 101, 102, 103, 104, 105, 106])
    mask = candidate_window_mask(probability, headings, abs_indices, radius=3)
    assert mask[:5].all()
    assert mask[5:11].all()
    assert not mask[11]


def test_connector_join_features_measure_entry_gain() -> None:
    texts = ["Smith J. (2020). A title,", "Journal 2: 10-20.", "ordinary prose"]
    n = len(texts)
    counts = np.zeros((n, len(FEATURE_NAMES)), dtype=np.uint32)
    gaps = np.zeros((n, len(GAP_SUMMARY_NAMES)), dtype=np.float32)
    entry = np.asarray([0.8, 0.1, 0.01], dtype=np.float32)
    headings = np.zeros((n, len(HEADING_PROBABILITY_NAMES)), dtype=np.float32)
    mask = np.ones(n, dtype=bool)

    def scorer(values: np.ndarray) -> float:
        return float(min(0.99, 0.1 + 0.05 * np.count_nonzero(values)))

    row = connector_feature_row(
        index=1, texts=texts, counts=counts, gap_summaries=gaps,
        abs_indices=np.arange(n, dtype=np.uint32), entry_probability=entry,
        heading_probability=headings, candidate_mask=mask, score_counts=scorer,
    )
    assert row.values.shape == (len(connector_feature_names()),)
    assert np.isfinite(row.values).all()
    assert row.joined_previous_score > 0


def test_candidate_edge_distance_is_local_to_the_contiguous_window() -> None:
    texts = [f"line {index}" for index in range(10)]
    counts = np.zeros((10, len(FEATURE_NAMES)), dtype=np.uint32)
    gaps = np.zeros((10, len(GAP_SUMMARY_NAMES)), dtype=np.float32)
    entry = np.zeros(10, dtype=np.float32)
    headings = np.zeros((10, len(HEADING_PROBABILITY_NAMES)), dtype=np.float32)
    mask = np.asarray([True, True, True, False, False, False, True, True, True, True])
    row = connector_feature_row(
        index=7, texts=texts, counts=counts, gap_summaries=gaps,
        abs_indices=np.arange(10, dtype=np.uint32), entry_probability=entry,
        heading_probability=headings, candidate_mask=mask,
        score_counts=lambda _: 0.0,
    )
    edge_index = connector_feature_names().index("candidate_window_edge_distance")
    assert row.values[edge_index] == 1


def test_p0d_transform_is_presence_plus_log_counts() -> None:
    counts = np.zeros((2, len(FEATURE_NAMES)), dtype=np.uint32)
    counts[1, 0] = 3
    matrix = p0d_matrix(counts)
    assert matrix.shape == (2, 2 * len(FEATURE_NAMES))
    assert matrix[1, 0] == 1
    assert np.isclose(matrix[1, len(FEATURE_NAMES)], np.log1p(3))

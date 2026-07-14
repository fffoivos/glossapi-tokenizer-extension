from __future__ import annotations

import numpy as np

from sequence_models.bibliography_entry_blocks import BlockConfig
from sequence_models.bibliography_entry_component_gate import (
    EXTENT_SATURATION_LINES,
    FEATURE_NAMES,
    _longest_true_run,
    _span_iou,
    candidate_supervision,
    component_feature_vector,
)


CONFIG = BlockConfig(
    anchor_probability=0.7,
    seed_length_limit=380,
    anchors_required=3,
    anchor_window=5,
    maximum_bridge_gap=2,
    inside_probability=0.25,
    header_window=2,
)


def test_component_features_have_one_value_per_documented_job() -> None:
    values = component_feature_vector(
        np.asarray([0.01, 0.8, 0.9, 0.1, 0.6]),
        np.asarray([20, 100, 120, 900, 80]),
        np.asarray([1, 0, 0, 0, 0]),
        np.asarray([False, False, True, False, True]),
        np.arange(5),
        1,
        4,
        CONFIG,
    )
    assert len(values) == len(FEATURE_NAMES) == 6
    assert np.isclose(values[0], 4 / EXTENT_SATURATION_LINES)
    assert np.isclose(values[1], 0.5)
    assert np.isclose(values[2], 0.7)
    assert np.isclose(values[3], 0.25)
    assert values[4] == 1.0
    assert np.isclose(values[5], 0.5)


def test_minimum_extent_saturates_without_rewarding_giant_merges() -> None:
    values = component_feature_vector(
        np.full(50, 0.8),
        np.full(50, 80),
        np.zeros(50),
        np.zeros(50, dtype=bool),
        np.arange(50),
        0,
        49,
        CONFIG,
    )
    assert values[0] == 1.0


def test_component_header_must_be_at_or_immediately_before_start() -> None:
    values = component_feature_vector(
        np.asarray([0.1, 0.1, 0.8, 0.8]),
        np.asarray([20, 20, 80, 80]),
        np.asarray([1, 0, 0, 0]),
        np.zeros(4, dtype=bool),
        np.asarray([0, 9, 10, 11]),
        2,
        3,
        CONFIG,
    )
    assert values[4] == 0.0

    header_at_start = component_feature_vector(
        np.asarray([0.1, 0.1, 0.8, 0.8]),
        np.asarray([20, 20, 80, 80]),
        np.asarray([0, 0, 1, 0]),
        np.zeros(4, dtype=bool),
        np.asarray([0, 9, 10, 11]),
        2,
        3,
        CONFIG,
    )
    assert header_at_start[4] == 1.0


def test_longest_weak_run_and_iou_are_exact() -> None:
    assert _longest_true_run(np.asarray([True, True, False, True])) == 2
    assert np.isclose(_span_iou((2, 5), (4, 7)), 1 / 3)


def test_candidate_supervision_uses_purity_not_whole_block_iou() -> None:
    assert candidate_supervision(np.ones(10, dtype=bool)) == 1
    assert candidate_supervision(
        np.asarray([True] * 8 + [False] * 2)
    ) == 1
    assert candidate_supervision(
        np.asarray([True] * 2 + [False] * 8)
    ) == 0
    assert candidate_supervision(
        np.asarray([True] * 5 + [False] * 5)
    ) == -1

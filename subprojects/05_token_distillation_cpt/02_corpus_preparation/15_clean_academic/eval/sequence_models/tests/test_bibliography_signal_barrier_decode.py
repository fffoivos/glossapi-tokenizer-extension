import numpy as np

from sequence_models.bibliography_deterministic_roles import ROLE_NAMES
from sequence_models.bibliography_entry_blocks import BlockConfig
from sequence_models.bibliography_signal_barrier_decode import (
    _decode_between_barriers,
    build_barrier_mask,
    sustained_low_mask,
)


def test_sustained_low_probability_marks_whole_run_but_not_single_weak_line():
    probability = np.asarray([0.9, 0.01, 0.9, 0.04, 0.03, 0.9])
    assert sustained_low_mask(
        probability, threshold=0.05, minimum_run=2
    ).tolist() == [False, False, False, True, True, False]


def test_selected_roles_and_low_runs_form_union_barrier():
    probability = np.asarray([0.9, 0.9, 0.01, 0.01, 0.9])
    roles = np.zeros((5, len(ROLE_NAMES)), dtype=np.uint8)
    roles[1, ROLE_NAMES.index("generic_markdown_heading")] = 1
    barrier = build_barrier_mask(
        probability,
        roles,
        barrier_arm="headings",
        low_probability=0.05,
        minimum_low_run=2,
    )
    assert barrier.tolist() == [False, True, True, True, False]


def test_barrier_prevents_anchors_on_opposite_sides_from_creating_block():
    config = BlockConfig(
        anchor_probability=0.9,
        seed_length_limit=1,
        anchors_required=2,
        anchor_window=8,
        maximum_bridge_gap=8,
        inside_probability=0.2,
        adjacent_expansion=1,
        header_window=2,
    )
    probability = np.asarray([0.95, 0.1, 0.1, 0.95])
    result = _decode_between_barriers(
        probability,
        np.arange(4),
        np.asarray([False, False, True, False]),
        config,
    )
    assert not result.any()

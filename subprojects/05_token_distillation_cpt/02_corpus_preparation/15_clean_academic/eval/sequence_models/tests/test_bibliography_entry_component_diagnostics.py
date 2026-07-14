import numpy as np

from sequence_models.bibliography_entry_component_diagnostics import (
    _chosen_rows,
    _line_fraction,
    _outside_context_probability,
    _proposal_groups,
    _role_at_or_before_start,
)


def test_proposal_groups_merge_overlapping_spans() -> None:
    assert _proposal_groups(
        np.asarray([1, 3, 20, 22]),
        np.asarray([5, 7, 21, 25]),
    ) == 2


def test_chosen_rows_keep_highest_scoring_nonoverlapping_components() -> None:
    chosen = _chosen_rows(
        np.asarray([0, 0, 0, 1]),
        np.asarray([0, 1, 9, 0]),
        np.asarray([5, 6, 10, 2]),
        np.asarray([0.7, 0.9, 0.6, 0.8]),
        0.5,
    )
    assert chosen.tolist() == [1, 2, 3]


def test_line_fraction_counts_each_line_once_across_columns() -> None:
    counts = np.asarray(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 0],
        ]
    )
    assert _line_fraction(counts, 0, 0, 3, [1, 2]) == 0.5


def test_outside_context_uses_only_nearby_physical_lines() -> None:
    probability = np.asarray([0.9, 0.1, 0.8, 0.8, 0.2, 0.7])
    abs_indices = np.asarray([0, 20, 21, 22, 23, 40])
    assert np.isclose(
        _outside_context_probability(
            probability,
            abs_indices,
            0,
            6,
            2,
            3,
            physical_window=8,
        ),
        0.15,
    )


def test_negative_role_scope_requires_physical_proximity() -> None:
    role = np.asarray([True, True, False, False])
    abs_indices = np.asarray([0, 8, 9, 10])
    assert _role_at_or_before_start(role, abs_indices, 0, 2) == 1.0
    assert _role_at_or_before_start(role, abs_indices, 0, 3) == 1.0
    assert _role_at_or_before_start(
        role, np.asarray([0, 1, 9, 10]), 0, 2
    ) == 0.0

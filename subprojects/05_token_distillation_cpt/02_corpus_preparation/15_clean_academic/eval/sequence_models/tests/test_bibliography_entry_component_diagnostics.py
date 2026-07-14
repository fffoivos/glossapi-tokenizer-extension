import numpy as np

from sequence_models.bibliography_entry_component_diagnostics import (
    _chosen_rows,
    _line_fraction,
    _proposal_groups,
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

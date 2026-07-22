import numpy as np

from sequence_models.bibliography_component_expansion import (
    expand_core_spans,
    select_core_spans,
)


def test_expansion_uses_only_proposal_graphs_containing_a_core() -> None:
    assert expand_core_spans(
        [(10, 20)],
        [(5, 12), (18, 30), (29, 35), (50, 60)],
    ) == [(5, 35)]


def test_overlapping_cores_are_not_selected_twice() -> None:
    chosen = select_core_spans(
        np.asarray([0, 2, 20]),
        np.asarray([10, 8, 25]),
        np.asarray([0.8, 0.9, 0.7]),
        np.asarray([0, 1, 2]),
    )
    assert chosen == [(2, 8), (20, 25)]

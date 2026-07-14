import numpy as np

from sequence_models.bibliography_auxiliary_scope_veto import has_auxiliary_scope


def test_auxiliary_scope_requires_an_exact_nearby_physical_heading() -> None:
    auxiliary = np.asarray([True, False, False, True, False])
    absolute = np.asarray([10, 11, 12, 20, 21])
    assert has_auxiliary_scope(auxiliary, absolute, 2, window=2)
    assert has_auxiliary_scope(auxiliary, absolute, 4, window=2)
    assert not has_auxiliary_scope(
        np.zeros_like(auxiliary), absolute, 2, window=2
    )


def test_auxiliary_scope_does_not_cross_a_physical_gap() -> None:
    auxiliary = np.asarray([True, False])
    absolute = np.asarray([10, 100])
    assert not has_auxiliary_scope(auxiliary, absolute, 1, window=2)

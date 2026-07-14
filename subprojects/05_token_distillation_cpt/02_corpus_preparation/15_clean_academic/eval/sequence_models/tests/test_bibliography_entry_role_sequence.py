import numpy as np

from sequence_models.bibliography_deterministic_roles import ROLE_NAMES
from sequence_models.bibliography_entry_role_sequence import _validate_role_matrix


def test_role_matrix_is_binary_mutually_exclusive_and_aligned(tmp_path) -> None:
    path = tmp_path / "roles.npy"
    np.save(path, np.eye(len(ROLE_NAMES), dtype=np.uint8))
    result = _validate_role_matrix(path, len(ROLE_NAMES))
    assert result.shape == (len(ROLE_NAMES), len(ROLE_NAMES))


def test_role_matrix_rejects_overlapping_roles(tmp_path) -> None:
    path = tmp_path / "roles.npy"
    roles = np.zeros((2, len(ROLE_NAMES)), dtype=np.uint8)
    roles[0, :2] = 1
    np.save(path, roles)
    try:
        _validate_role_matrix(path, 2)
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("overlapping deterministic roles were accepted")

from types import SimpleNamespace

import numpy as np
import pytest

from sequence_models.bibliography_deterministic_roles import ROLE_NAMES
from sequence_models.bibliography_entry_dataset import LABEL_TO_ID, MAX_PHYSICAL_GAP
from sequence_models.bibliography_signal_tcn import (
    FEATURE_NAMES,
    SignalTCN,
    build_signal_features,
    make_signal_chunks,
)


def test_signal_features_are_narrow_and_roles_are_perpendicular():
    probability = np.asarray([0.1, 0.8, 0.4], dtype=np.float32)
    roles = np.zeros((3, len(ROLE_NAMES)), dtype=np.uint8)
    roles[1, 2] = 1
    features = build_signal_features(probability, roles, np.asarray([0, 1, 0]))
    assert features.shape == (3, len(FEATURE_NAMES))
    assert features[1].tolist() == pytest.approx(
        [0.8] + [0.0, 0.0, 1.0] + [0.0] * (len(ROLE_NAMES) - 3) + [1.0]
    )
    roles[1, 3] = 1
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_signal_features(probability, roles, np.asarray([0, 1, 0]))


def test_chunks_cover_known_lines_once_and_break_at_unknown_and_physical_gap():
    unknown = LABEL_TO_ID["UNKNOWN"]
    labels = np.asarray([0, 1, 0, unknown, 0, 1, 1, 0], dtype=np.uint8)
    table = SimpleNamespace(
        documents=({"line_start": 0, "line_end": len(labels)},),
        original_labels=labels,
        abs_indices=np.asarray(
            [0, 1, 2, 3, 4, 5, 5 + MAX_PHYSICAL_GAP + 1, 7 + MAX_PHYSICAL_GAP]
        ),
    )
    chunks = make_signal_chunks(table, [0], central_width=2, context=1)
    assigned = [
        index
        for chunk in chunks
        for index in range(chunk.target_start, chunk.target_end)
    ]
    assert assigned == [0, 1, 2, 4, 5, 6, 7]
    assert len(assigned) == len(set(assigned))
    assert all(
        chunk.input_start <= chunk.target_start < chunk.target_end <= chunk.input_end
        for chunk in chunks
    )


def test_signal_tcn_preserves_batch_and_line_shape():
    torch = pytest.importorskip("torch")
    model = SignalTCN(len(FEATURE_NAMES), hidden_dim=8, dilations=(1, 2), dropout=0.0)
    features = torch.zeros((2, 7, len(FEATURE_NAMES)), dtype=torch.float32)
    mask = torch.asarray(
        [[True] * 7, [True, True, True, True, False, False, False]]
    )
    assert tuple(model(features, mask).shape) == (2, 7)

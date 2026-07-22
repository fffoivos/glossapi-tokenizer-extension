from types import SimpleNamespace

import numpy as np

from sequence_models.bibliography_deterministic_roles import ROLE_NAMES
from sequence_models.bibliography_entry_blocks import BlockConfig
from sequence_models.bibliography_entry_component_gate import CandidateSet
from sequence_models.bibliography_rich_component_gate import (
    FEATURE_NAMES,
    rich_component_features,
)


def test_rich_features_separate_probability_tails_boundaries_and_roles() -> None:
    table = SimpleNamespace(
        targets=np.zeros(5, dtype=np.uint8),
        documents=({"line_start": 0},),
    )
    base = CandidateSet(
        features=np.asarray([[0, 0, 0, 1, 0]], dtype=np.float32),
        document_indices=np.asarray([0]),
        starts=np.asarray([1]),
        ends=np.asarray([4]),
        labels=np.asarray([1]),
    )
    probability = np.asarray([0.0, 0.1, 0.2, 0.8, 0.9])
    roles = np.zeros((5, len(ROLE_NAMES)), dtype=np.uint8)
    roles[1:3, 0] = 1
    config = BlockConfig(0.7, 380, 2, 4, 2)
    result = rich_component_features(table, base, probability, roles, config)
    assert result.shape == (1, len(FEATURE_NAMES))
    assert result[0, FEATURE_NAMES.index("entry_probability_q10")] < 0.2
    assert result[0, FEATURE_NAMES.index("entry_probability_q90")] > 0.8
    assert np.isclose(
        result[0, FEATURE_NAMES.index("minimum_boundary_probability")], 0.15
    )
    assert result[0, FEATURE_NAMES.index("exact_header_at_or_before_start")] == 1
    assert result[0, FEATURE_NAMES.index("role_figure_caption_fraction")] == 0.5

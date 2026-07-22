from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sequence_models.bibliography_entry_blocks import BlockConfig
from sequence_models.bibliography_entry_semimarkov import (
    SPAN_FEATURES,
    _candidate_configs,
    _span_features,
    decode_candidates,
)


CONFIG = BlockConfig(0.8, 330, 2, 3, 2)


def test_candidate_grid_is_filtered_and_deterministic() -> None:
    first = _candidate_configs(CONFIG)
    second = _candidate_configs(CONFIG)
    assert first == second
    assert len(first) == len(set(first))
    assert all(config.anchors_required >= 2 for config in first)
    assert all(config.maximum_bridge_gap <= 4 for config in first)


def test_span_feature_inventory_is_complete() -> None:
    features = _span_features(
        np.asarray([0.9, 0.1, 0.8]),
        np.asarray([100, 900, 100]),
        np.asarray([0, 1, 0]),
        0,
        2,
        CONFIG,
    )
    assert features.shape == (len(SPAN_FEATURES),)
    assert features[0] == 3
    assert features[1] == 2
    assert features[10] == 1
    assert features[11] == 1


def test_span_decoder_selects_nonoverlapping_candidates_and_then_h0() -> None:
    table = SimpleNamespace(
        documents=({"line_start": 0, "line_end": 6},),
        targets=np.zeros(6, dtype=np.int8),
        header_kinds=np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        abs_indices=np.arange(6, dtype=np.uint32),
    )
    prediction = decode_candidates(
        table,
        scores=np.asarray([0.9, 0.8, 0.95]),
        document_indices=np.asarray([0, 0, 0]),
        starts=np.asarray([1, 2, 4]),
        ends=np.asarray([3, 4, 5]),
        probability=np.asarray([0.01, 0.9, 0.8, 0.8, 0.9, 0.9]),
        config=CONFIG,
    )
    # Highest score chooses 4-5, then 1-3; H0 attaches line 0 afterwards.
    assert prediction.tolist() == [True, True, True, True, True, True]

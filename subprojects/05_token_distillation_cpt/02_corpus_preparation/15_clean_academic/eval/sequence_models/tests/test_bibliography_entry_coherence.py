from __future__ import annotations

import numpy as np

from sequence_models.bibliography_entry_blocks import BlockConfig
from sequence_models.bibliography_entry_coherence import (
    AnchoredCoherenceConfig,
    filter_anchored_components,
)


class _Table:
    documents = ({"line_start": 0, "line_end": 12},)
    targets = np.zeros(12, dtype=np.uint8)
    char_lengths = np.asarray([20, 80, 90, 500, 70, 30, 20, 80, 90, 70, 60, 20])
    abs_indices = np.arange(12)
    header_kinds = np.zeros(12, dtype=np.uint8)


BLOCK = BlockConfig(
    anchor_probability=0.7,
    seed_length_limit=380,
    anchors_required=3,
    anchor_window=5,
    maximum_bridge_gap=2,
)


def test_filter_keeps_weak_and_long_lines_inside_anchored_component() -> None:
    raw = np.asarray([False, True, True, True, True, False, False, False, False, False, False, False])
    probability = np.asarray([0.01, 0.9, 0.8, 0.02, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    result = filter_anchored_components(
        _Table(),
        raw,
        probability,
        block_config=BLOCK,
        config=AnchoredCoherenceConfig(0.0, 2, 3),
        attach_headers=False,
    )
    assert result.tolist() == raw.tolist()


def test_filter_rejects_isolated_or_weak_components() -> None:
    raw = np.asarray([False, True, False, False, False, False, False, True, True, True, False, False])
    probability = np.asarray([0.01, 0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.8, 0.2, 0.1, 0.01, 0.01])
    result = filter_anchored_components(
        _Table(),
        raw,
        probability,
        block_config=BLOCK,
        config=AnchoredCoherenceConfig(0.0, 2, 3),
        attach_headers=False,
    )
    assert not result.any()


def test_filter_does_not_count_long_line_as_anchor() -> None:
    raw = np.asarray([False, True, True, True, False, False, False, False, False, False, False, False])
    probability = np.asarray([0.01, 0.95, 0.95, 0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    result = filter_anchored_components(
        _Table(),
        raw,
        probability,
        block_config=BLOCK,
        config=AnchoredCoherenceConfig(0.0, 3, 3),
        attach_headers=False,
    )
    assert not result.any()

from types import SimpleNamespace

import numpy as np

from sequence_models.bibliography_entry_blocks import BlockConfig
from sequence_models.bibliography_signal_block_decode import decode_signal_blocks


def _table(length: int):
    return SimpleNamespace(
        targets=np.zeros(length, dtype=np.int8),
        documents=({"line_start": 0, "line_end": length},),
        abs_indices=np.arange(length, dtype=np.uint32),
        header_kinds=np.zeros(length, dtype=np.uint8),
    )


def _config() -> BlockConfig:
    return BlockConfig(
        anchor_probability=0.9,
        seed_length_limit=1,
        anchors_required=2,
        anchor_window=4,
        maximum_bridge_gap=8,
        inside_probability=0.2,
        adjacent_expansion=1,
        header_window=2,
    )


def test_two_anchors_include_weak_interior_but_one_anchor_cannot_start_block():
    table = _table(8)
    prediction, _ = decode_signal_blocks(
        table,
        np.asarray([0.0, 0.95, 0.01, 0.95, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(8),
        np.zeros(8, dtype=bool),
        _config(),
        qualified_documents={0},
        apply_veto=True,
    )
    assert prediction.tolist() == [False, True, True, True, False, False, False, False]
    isolated, _ = decode_signal_blocks(
        table,
        np.asarray([0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(8),
        np.zeros(8, dtype=bool),
        _config(),
        qualified_documents={0},
        apply_veto=True,
    )
    assert not isolated.any()


def test_exact_scope_veto_removes_established_component_without_creating_one():
    table = _table(8)
    scope = np.zeros(8, dtype=bool)
    scope[1] = True
    prediction, vetoed = decode_signal_blocks(
        table,
        np.asarray([0.0, 0.95, 0.1, 0.95, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(8),
        scope,
        _config(),
        qualified_documents={0},
        apply_veto=True,
    )
    assert vetoed == 1
    assert not prediction.any()

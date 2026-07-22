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


def test_scope_line_cannot_act_as_an_anchor():
    table = _table(8)
    scope = np.zeros(8, dtype=bool)
    scope[1] = True
    prediction, barrier_intervals = decode_signal_blocks(
        table,
        np.asarray([0.0, 0.95, 0.1, 0.95, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(8),
        scope,
        _config(),
        qualified_documents={0},
        apply_veto=True,
    )
    assert barrier_intervals == 1
    assert not prediction.any()


def test_scope_is_a_wall_not_a_poison_pill_for_an_adjacent_block():
    table = _table(9)
    scope = np.zeros(9, dtype=bool)
    scope[4:7] = True
    prediction, barrier_intervals = decode_signal_blocks(
        table,
        np.asarray([0.0, 0.95, 0.1, 0.95, 0.3, 0.3, 0.3, 0.0, 0.0]),
        np.zeros(9),
        scope,
        _config(),
        qualified_documents={0},
        apply_veto=True,
    )
    assert barrier_intervals == 1
    assert prediction.tolist() == [False, True, True, True, False, False, False, False, False]


def test_scope_prevents_anchors_from_joining_across_it():
    table = _table(8)
    scope = np.zeros(8, dtype=bool)
    scope[3:5] = True
    prediction, _ = decode_signal_blocks(
        table,
        np.asarray([0.0, 0.95, 0.1, 0.2, 0.2, 0.1, 0.95, 0.0]),
        np.zeros(8),
        scope,
        _config(),
        qualified_documents={0},
        apply_veto=True,
    )
    assert not prediction.any()


def test_independently_established_blocks_survive_on_both_sides_of_scope():
    table = _table(11)
    scope = np.zeros(11, dtype=bool)
    scope[4:7] = True
    prediction, barrier_intervals = decode_signal_blocks(
        table,
        np.asarray(
            [0.0, 0.95, 0.1, 0.95, 0.3, 0.3, 0.3, 0.95, 0.1, 0.95, 0.0]
        ),
        np.zeros(11),
        scope,
        _config(),
        qualified_documents={0},
        apply_veto=True,
    )
    assert barrier_intervals == 1
    assert prediction.tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]


def test_header_attachment_cannot_jump_across_scope():
    table = _table(7)
    table.header_kinds[1] = 1
    scope = np.zeros(7, dtype=bool)
    scope[2] = True
    prediction, _ = decode_signal_blocks(
        table,
        np.asarray([0.0, 0.0, 0.3, 0.95, 0.1, 0.95, 0.0]),
        np.zeros(7),
        scope,
        _config(),
        qualified_documents={0},
        apply_veto=True,
    )
    assert prediction.tolist() == [False, False, False, True, True, True, False]

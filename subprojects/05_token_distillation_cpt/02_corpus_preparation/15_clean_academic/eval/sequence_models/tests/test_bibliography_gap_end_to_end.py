import numpy as np

from sequence_models.bibliography_gap_end_to_end import (
    attach_main_bib_headers,
    decode_components,
)


def test_connected_components_emit_the_complete_gap() -> None:
    prediction = decode_components(
        [(2, 2), (5, 5)], {(2, 5): True}, line_count=8,
    )
    assert prediction.tolist() == [False, False, True, True, True, True, False, False]


def test_isolated_single_seed_is_not_emitted() -> None:
    prediction = decode_components([(2, 2)], {}, line_count=5)
    assert not np.any(prediction)


def test_two_adjacent_seed_lines_establish_a_block_without_a_gap() -> None:
    prediction = decode_components([(2, 3)], {}, line_count=5)
    assert prediction.tolist() == [False, False, True, True, False]


def test_unconnected_components_remain_separate_supported_blocks() -> None:
    prediction = decode_components(
        [(0, 1), (4, 5)], {(1, 4): False}, line_count=6,
    )
    assert prediction.tolist() == [True, True, False, False, True, True]


def test_oracle_and_baseline_can_share_components_without_state_leakage() -> None:
    components = [(0, 0), (2, 2), (4, 4)]
    baseline = decode_components(components, {}, line_count=5)
    oracle = decode_components(
        components, {(0, 2): True, (2, 4): True}, line_count=5,
    )
    assert not np.any(baseline)
    assert oracle.tolist() == [True, True, True, True, True]


def test_main_bibliography_header_attaches_only_above_existing_block() -> None:
    prediction = np.asarray([False, False, True, True])
    heading = np.asarray([
        [0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    attached = attach_main_bib_headers(
        prediction, heading, np.arange(4), np.zeros(4, dtype=bool), threshold=0.5,
    )
    assert attached.tolist() == [False, True, True, True]


def test_non_bibliography_heading_blocks_header_attachment() -> None:
    prediction = np.asarray([False, False, False, True])
    heading = np.asarray([
        [0.9, 0.0, 0.0],
        [0.0, 0.0, 0.9],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    attached = attach_main_bib_headers(
        prediction, heading, np.arange(4), np.zeros(4, dtype=bool),
        threshold=0.5, window=3,
    )
    assert attached.tolist() == prediction.tolist()

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sequence_models.bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    blocks_from_mask,
    decode_b0_document,
    evaluate_prediction,
)


CONFIG = BlockConfig(
    anchor_probability=0.8,
    seed_length_limit=330,
    anchors_required=2,
    anchor_window=3,
    maximum_bridge_gap=2,
    inside_probability=0.25,
    adjacent_expansion=1,
    header_window=2,
)


def test_long_line_cannot_seed_but_is_absorbed_between_anchors() -> None:
    probability = np.asarray([0.95, 0.01, 0.96, 0.99])
    lengths = np.asarray([100, 900, 100, 900])
    absolute = np.asarray([0, 1, 2, 3])
    predicted = decode_b0_document(probability, lengths, absolute, CONFIG)
    assert predicted.tolist() == [True, True, True, True]
    assert not decode_b0_document(
        np.asarray([0.99, 0.99]),
        np.asarray([900, 800]),
        np.asarray([0, 1]),
        CONFIG,
    ).any()


def test_h0_attaches_to_a_block_but_cannot_create_one() -> None:
    b0 = np.asarray([False, True, True, False])
    probability = np.asarray([0.01, 0.95, 0.96, 0.01])
    headings = np.asarray([1, 0, 0, 0])
    absolute = np.arange(4)
    assert attach_h0_document(b0, probability, headings, absolute, CONFIG).tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert not attach_h0_document(
        np.zeros(4, dtype=bool), probability, headings, absolute, CONFIG
    ).any()


def test_physical_gap_splits_predicted_blocks() -> None:
    mask = np.asarray([True, True, True])
    assert blocks_from_mask(mask, np.asarray([0, 1, 100])) == [(0, 1), (2, 2)]


def test_perfect_prediction_has_perfect_line_token_and_block_metrics() -> None:
    table = SimpleNamespace(
        documents=(
            {
                "line_start": 0,
                "line_end": 5,
                "source": "openarchives",
                "coverage": "full_document",
            },
        ),
        original_labels=np.asarray([0, 1, 1, 1, 0], dtype=np.uint8),
        token_counts=np.asarray([3, 1, 5, 5, 3], dtype=np.uint32),
        char_lengths=np.asarray([20, 12, 80, 90, 20], dtype=np.uint32),
        abs_indices=np.arange(5, dtype=np.uint32),
    )
    prediction = np.asarray([False, True, True, True, False])
    metrics = evaluate_prediction(table, prediction)
    assert metrics["line_precision"] == metrics["line_recall"] == 1.0
    assert metrics["token_precision"] == metrics["token_recall"] == 1.0
    assert metrics["exact_block_precision"] == metrics["exact_block_recall"] == 1.0
    assert metrics["split_error_count"] == metrics["merge_error_count"] == 0

from __future__ import annotations

import numpy as np

from sequence_models.bibliography_nextgen_decode import DecoderConfig, decode_document


NAMES = (
    "probability:bib_header",
    "probability:bib_subheader",
    "probability:continuation_specialist",
    "presence:numbered_entry_count",
    "structure:markdown_heading",
    "structure:image_marker",
    "structure:table_row",
)


def _features(length: int) -> np.ndarray:
    return np.zeros((length, len(NAMES)), dtype=np.float32)


def _config(**changes: object) -> DecoderConfig:
    values = {
        "anchor_probability": 0.8,
        "inside_probability": 0.3,
        "anchor_window": 8,
        "maximum_bridge_gap": 8,
        "adjacent_expansion": 1,
        "allow_guarded_single_anchor": False,
    }
    values.update(changes)
    return DecoderConfig(**values)


def test_terminal_subheader_cannot_extend_bibliography_into_cv() -> None:
    probability = np.asarray((0.9, 0.9, 0.1, 0.1, 0.1), dtype=np.float32)
    features = _features(len(probability))
    features[2, NAMES.index("structure:markdown_heading")] = 1
    features[2, NAMES.index("probability:bib_subheader")] = 0.9
    predicted = decode_document(
        probability,
        features,
        NAMES,
        np.asarray((100, 100, 25, 20, 8), dtype=np.uint32),
        np.arange(len(probability), dtype=np.uint32),
        _config(),
    )
    assert predicted.tolist() == [True, True, False, False, False]


def test_subheader_connects_supported_bibliography_on_both_sides() -> None:
    probability = np.asarray((0.9, 0.9, 0.05, 0.9, 0.9), dtype=np.float32)
    features = _features(len(probability))
    features[2, NAMES.index("structure:markdown_heading")] = 1
    features[2, NAMES.index("probability:bib_subheader")] = 0.9
    predicted = decode_document(
        probability,
        features,
        NAMES,
        np.asarray((100, 100, 20, 100, 100), dtype=np.uint32),
        np.arange(len(probability), dtype=np.uint32),
        _config(),
    )
    assert predicted.all()


def test_long_markdown_table_rows_can_seed_a_block() -> None:
    probability = np.asarray((0.9, 0.9), dtype=np.float32)
    features = _features(len(probability))
    features[:, NAMES.index("structure:table_row")] = 1
    predicted = decode_document(
        probability,
        features,
        NAMES,
        np.asarray((490, 490), dtype=np.uint32),
        np.arange(len(probability), dtype=np.uint32),
        _config(),
    )
    assert predicted.all()


def test_image_marker_is_internal_only() -> None:
    probability = np.asarray((0.9, 0.9, 0.99), dtype=np.float32)
    features = _features(len(probability))
    features[2, NAMES.index("structure:image_marker")] = 1
    predicted = decode_document(
        probability,
        features,
        NAMES,
        np.asarray((100, 100, 14), dtype=np.uint32),
        np.arange(len(probability), dtype=np.uint32),
        _config(),
    )
    assert predicted.tolist() == [True, True, False]

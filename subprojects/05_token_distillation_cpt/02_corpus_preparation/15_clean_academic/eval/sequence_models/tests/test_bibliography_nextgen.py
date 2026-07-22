from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sequence_models.bibliography_nextgen_decode import DecoderConfig, decode_document
from sequence_models.bibliography_nextgen_models import (
    CONTEXT_SIGNALS,
    build_context_features,
    context_feature_names,
)
from sequence_models.bibliography_nextgen_scope import (
    SIGNALS,
    apply_component_threshold,
    build_component_table,
    component_feature_names,
)
from sequence_models.bibliography_nextgen_table import _load_specialist


NAMES = (
    "probability:bib_header",
    "probability:bib_subheader",
    "probability:continuation_specialist",
    "presence:numbered_entry_count",
    "presence:year_count",
    "presence:url_count",
    "presence:doi_count",
    "presence:page_range_count",
    "structure:markdown_heading",
    "structure:bib_heading_lexicon",
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


def test_sparse_continuation_specialist_preserves_fallback(tmp_path) -> None:
    connector = tmp_path / "connector"
    specialist = tmp_path / "specialist"
    connector.mkdir()
    specialist.mkdir()
    np.save(connector / "row_indices.npy", np.asarray([1, 3, 4], dtype=np.int64))
    np.save(
        specialist / "special.npy",
        np.asarray([0.8, np.nan, 0.2], dtype=np.float32),
    )
    fallback = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

    result, digest, count = _load_specialist(
        specialist, connector, fallback, 5, "special.npy"
    )

    np.testing.assert_allclose(result, [0.1, 0.8, 0.3, 0.4, 0.2])
    assert digest is not None
    assert count == 2


def test_strict_decoder_can_use_but_not_emit_markdown_subheader() -> None:
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
        _config(emit_markdown_headings=False),
    )
    assert predicted.tolist() == [True, True, False, True, True]


def test_gated_heading_emits_only_bibliography_heading_attached_to_block() -> None:
    probability = np.asarray((0.05, 0.9, 0.9, 0.05), dtype=np.float32)
    features = _features(len(probability))
    markdown = NAMES.index("structure:markdown_heading")
    features[[0, 3], markdown] = 1
    features[0, NAMES.index("probability:bib_header")] = 0.9
    features[0, NAMES.index("structure:bib_heading_lexicon")] = 1
    predicted = decode_document(
        probability,
        features,
        NAMES,
        np.asarray((12, 100, 100, 12), dtype=np.uint32),
        np.arange(len(probability), dtype=np.uint32),
        _config(
            emit_markdown_headings=False,
            gated_bib_heading_window=2,
            gated_bib_heading_require_lexicon=True,
        ),
    )
    assert predicted.tolist() == [True, True, True, False]


def test_lexicon_gate_rejects_model_only_heading() -> None:
    probability = np.asarray((0.05, 0.9, 0.9), dtype=np.float32)
    features = _features(len(probability))
    features[0, NAMES.index("structure:markdown_heading")] = 1
    features[0, NAMES.index("probability:bib_header")] = 0.9
    predicted = decode_document(
        probability,
        features,
        NAMES,
        np.asarray((12, 100, 100), dtype=np.uint32),
        np.arange(len(probability), dtype=np.uint32),
        _config(
            emit_markdown_headings=False,
            gated_bib_heading_window=2,
            gated_bib_heading_require_lexicon=True,
        ),
    )
    assert predicted.tolist() == [False, True, True]


def test_conditioned_expansion_blocks_long_evidence_free_prose() -> None:
    probability = np.asarray((0.9, 0.9, 0.7), dtype=np.float32)
    features = _features(len(probability))
    predicted = decode_document(
        probability,
        features,
        NAMES,
        np.asarray((100, 100, 500), dtype=np.uint32),
        np.arange(len(probability), dtype=np.uint32),
        _config(conditioned_long_line_expansion=True),
    )
    assert predicted.tolist() == [True, True, False]


def test_conditioned_expansion_keeps_long_line_with_bibliographic_evidence() -> None:
    probability = np.asarray((0.9, 0.9, 0.7), dtype=np.float32)
    features = _features(len(probability))
    features[2, NAMES.index("presence:year_count")] = 1
    predicted = decode_document(
        probability,
        features,
        NAMES,
        np.asarray((100, 100, 500), dtype=np.uint32),
        np.arange(len(probability), dtype=np.uint32),
        _config(conditioned_long_line_expansion=True),
    )
    assert predicted.tolist() == [True, True, True]


def test_auxiliary_scope_veto_removes_only_scoped_component() -> None:
    probability = np.asarray((0.9, 0.9, 0.0, 0.9, 0.9), dtype=np.float32)
    features = _features(len(probability))
    predicted = decode_document(
        probability,
        features,
        NAMES,
        np.asarray((100, 100, 10, 100, 100), dtype=np.uint32),
        np.asarray((0, 1, 100, 101, 102), dtype=np.uint32),
        _config(apply_auxiliary_scope_veto=True),
        np.asarray((True, True, False, False, False)),
    )
    assert predicted.tolist() == [False, False, False, True, True]


def test_context_features_encode_document_and_segment_position() -> None:
    names = (*CONTEXT_SIGNALS, "structure:markdown_heading")
    features = np.zeros((3, len(names)), dtype=np.float32)
    table = SimpleNamespace(
        documents=({"line_start": 0, "line_end": 3},),
        abs_indices=np.asarray((10, 11, 12), dtype=np.uint32),
    )

    result = build_context_features(features, names, table)
    result_names = context_feature_names(names)

    assert result.shape == (3, len(result_names))
    np.testing.assert_allclose(
        result[:, result_names.index("document:relative_line_position")],
        (0.0, 0.5, 1.0),
    )
    np.testing.assert_allclose(
        result[:, result_names.index("document:relative_lines_remaining")],
        (1.0, 0.5, 0.0),
    )
    np.testing.assert_allclose(
        result[:, result_names.index("document:physical_segment_relative_position")],
        (0.0, 0.5, 1.0),
    )


def test_component_scope_features_and_veto_are_component_aligned() -> None:
    names = SIGNALS
    features = np.zeros((4, len(names)), dtype=np.float32)
    table = SimpleNamespace(
        targets=np.zeros(4, dtype=np.uint8),
        original_labels=np.asarray((1, 1, 0, 0), dtype=np.uint8),
        char_lengths=np.asarray((10, 20, 30, 40), dtype=np.uint32),
        abs_indices=np.asarray((0, 1, 2, 3), dtype=np.uint32),
        documents=({"line_start": 0, "line_end": 4},),
    )
    prediction = np.asarray((True, True, False, True))
    probability = np.asarray((0.9, 0.8, 0.1, 0.7), dtype=np.float32)

    matrix, documents, bounds, target = build_component_table(
        table, prediction, probability, features, names
    )
    scoped = apply_component_threshold(
        prediction, bounds, np.asarray((0.9, 0.1), dtype=np.float32), 0.5
    )

    assert matrix.shape[0] == 2
    assert documents.tolist() == [0, 0]
    assert bounds.tolist() == [[0, 1], [3, 3]]
    assert target.tolist() == [True, False]
    assert scoped.tolist() == [True, True, False, False]


def test_component_scope_target_requires_configured_gold_purity() -> None:
    names = SIGNALS
    features = np.zeros((2, len(names)), dtype=np.float32)
    table = SimpleNamespace(
        original_labels=np.asarray((1, 0), dtype=np.uint8),
        char_lengths=np.asarray((10, 10), dtype=np.uint32),
        abs_indices=np.asarray((0, 1), dtype=np.uint32),
        documents=({"line_start": 0, "line_end": 2},),
    )
    _, _, _, target = build_component_table(
        table,
        np.asarray((True, True)),
        np.asarray((0.9, 0.8), dtype=np.float32),
        features,
        names,
        target_min_gold_fraction=0.75,
    )
    assert target.tolist() == [False]


def test_component_scope_heading_rescue_and_structural_veto() -> None:
    names = component_feature_names()
    x = np.zeros((3, len(names)), dtype=np.float32)
    x[0, names.index("signal:structure:bib_heading_lexicon:max")] = 1
    x[1, names.index("signal:structure:image_marker:active_fraction")] = 0.2
    x[2, names.index("signal:structure:rule_line:active_fraction")] = 0.1
    prediction = np.ones(3, dtype=bool)
    bounds = np.asarray(((0, 0), (1, 1), (2, 2)), dtype=np.uint32)
    result = apply_component_threshold(
        prediction,
        bounds,
        np.asarray((0.65, 0.99, 0.99), dtype=np.float32),
        0.9,
        component_features=x,
        heading_rescue_floor=0.6,
        image_fraction_veto=0.15,
        rule_fraction_veto=0.05,
    )
    assert result.tolist() == [True, False, False]

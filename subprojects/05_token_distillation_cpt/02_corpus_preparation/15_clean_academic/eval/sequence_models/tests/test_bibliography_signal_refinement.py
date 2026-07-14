from types import SimpleNamespace

import numpy as np

from sequence_models.bibliography_deterministic_roles import ROLE_NAMES
from sequence_models.bibliography_entry_blocks import BlockConfig
from sequence_models.bibliography_entry_component_gate import CandidateSet
from sequence_models.bibliography_signal_refinement import (
    EXPECTED_DIRECTIONS,
    FEATURE_NAMES,
    _split_span_at_headings,
    component_feature_vector,
    decode_component_candidates,
    refine_outer_edges,
)
from sequence_models.bibliography_signal_refinement_unseen import (
    _core_prediction,
    _summarize,
)


def _table(length: int):
    return SimpleNamespace(
        targets=np.zeros(length, dtype=np.int8),
        documents=({"line_start": 0, "line_end": length},),
        abs_indices=np.arange(length, dtype=np.uint32),
    )


def _roles(length: int) -> np.ndarray:
    return np.zeros((length, len(ROLE_NAMES)), dtype=np.uint8)


def test_edge_policy_removes_only_fringe_at_and_beyond_structural_role() -> None:
    table = _table(9)
    base = np.asarray([False, True, True, True, True, True, True, True, False])
    core = np.asarray([False, False, False, True, True, True, False, False, False])
    roles = _roles(9)
    heading = ROLE_NAMES.index("generic_markdown_heading")
    roles[1, heading] = 1
    roles[6, heading] = 1
    result = refine_outer_edges(
        table,
        base,
        core,
        roles,
        role_names=("generic_markdown_heading",),
        side="both",
        qualified_documents={0},
    )
    assert result.tolist() == [False, False, True, True, True, True, False, False, False]


def test_edge_policy_never_removes_a_role_inside_the_anchored_core() -> None:
    table = _table(7)
    base = np.asarray([False, True, True, True, True, True, False])
    core = np.asarray([False, False, True, True, True, False, False])
    roles = _roles(7)
    roles[3, ROLE_NAMES.index("generic_markdown_heading")] = 1
    result = refine_outer_edges(
        table,
        base,
        core,
        roles,
        role_names=("generic_markdown_heading",),
        side="both",
        qualified_documents={0},
    )
    assert np.array_equal(result, base)


def test_heading_split_assigns_heading_to_following_component() -> None:
    headings = np.zeros(10, dtype=bool)
    headings[[4, 7]] = True
    assert _split_span_at_headings(2, 8, headings) == [(2, 3), (4, 6), (7, 8)]


def test_component_features_are_fixed_structural_summaries() -> None:
    signal = np.asarray([0.1, 0.4, 0.8, 0.9])
    entry = np.asarray([0.2, 0.6, 0.7, 0.1])
    roles = _roles(4)
    roles[0, ROLE_NAMES.index("generic_markdown_heading")] = 1
    roles[3, ROLE_NAMES.index("footnote")] = 1
    headers = np.asarray([0, 1, 0, 0], dtype=np.uint8)
    config = BlockConfig(0.3, 1, 2, 16, 8, 0.05, 2, 2)
    features = component_feature_vector(
        signal,
        entry,
        roles,
        headers,
        start=0,
        end=3,
        config=config,
    )
    assert len(features) == len(FEATURE_NAMES) == len(EXPECTED_DIRECTIONS)
    assert np.isclose(features[FEATURE_NAMES.index("signal_anchor_fraction")], 0.75)
    assert np.isclose(features[FEATURE_NAMES.index("entry_positive_fraction")], 0.5)
    assert np.isclose(features[FEATURE_NAMES.index("longest_entry_run_fraction")], 0.5)
    assert np.isclose(
        features[FEATURE_NAMES.index("longest_hard_negative_run_fraction")], 0.25
    )
    assert features[FEATURE_NAMES.index("starts_with_generic_heading")] == 1
    assert features[FEATURE_NAMES.index("exact_header_at_or_before_start")] == 0
    assert np.isclose(features[FEATURE_NAMES.index("role_footnote_fraction")], 0.25)


def test_component_gate_cannot_add_a_header_outside_frozen_proposal() -> None:
    table = SimpleNamespace(
        targets=np.zeros(6, dtype=np.int8),
        documents=({"line_start": 0, "line_end": 6},),
        abs_indices=np.arange(6, dtype=np.uint32),
        header_kinds=np.asarray([0, 1, 0, 0, 0, 0], dtype=np.uint8),
    )
    candidates = CandidateSet(
        features=np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
        document_indices=np.asarray([0], dtype=np.uint32),
        starts=np.asarray([2], dtype=np.uint32),
        ends=np.asarray([4], dtype=np.uint32),
        labels=np.asarray([1], dtype=np.int8),
    )
    prediction = decode_component_candidates(
        table,
        candidates,
        np.asarray([0.9]),
        np.asarray([0.0, 0.0, 0.8, 0.8, 0.8, 0.0]),
        BlockConfig(0.3, 1, 2, 16, 8, 0.05, 2, 2),
        threshold=0.5,
        qualified_documents={0},
    )
    assert prediction.tolist() == [False, False, True, True, True, False]


def test_unseen_core_reconstruction_drops_only_expansion_fringe() -> None:
    base = np.asarray([False, True, True, True, True, True, False])
    signal = np.asarray([0.0, 0.1, 0.8, 0.1, 0.8, 0.1, 0.0])
    core = _core_prediction(
        base,
        signal,
        np.zeros(7),
        np.zeros(7, dtype=np.uint8),
        np.arange(7, dtype=np.uint32),
        BlockConfig(0.3, 1, 2, 16, 8, 0.05, 2, 2),
    )
    assert core.tolist() == [False, False, True, True, True, False, False]


def test_unseen_summary_keeps_marks_distinct_from_unreviewed_removals() -> None:
    summary = _summarize(
        {"a", "b", "c", "d"},
        {"b", "d"},
        {"a", "b"},
        {"d"},
        {"a"},
    )
    assert summary["marked_wrong_removed"] == 1
    assert summary["marked_wrong_retained"] == 1
    assert summary["unmarked_prediction_removed_review_risk"] == 1
    assert summary["marked_whole_block_wrong_removed"] == 1

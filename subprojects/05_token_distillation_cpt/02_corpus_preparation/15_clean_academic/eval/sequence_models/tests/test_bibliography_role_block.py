from __future__ import annotations

import numpy as np

from sequence_models.bibliography_role_block import (
    FEATURE_NAMES, BlockExample, DecoderConfig, StructuredConfig, StructuredModel,
    candidate_spans, decode, gold_is_proposal_reachable, gold_is_seed_reachable,
    prediction_mask,
)
from sequence_models.bibliography_role_v2 import OPERATIONAL_ROLES, ROLE_TO_ID


INDEX = {role: index for index, role in enumerate(OPERATIONAL_ROLES)}


def example(probabilities: list[dict[str, float]], gold: list[str]) -> BlockExample:
    p = np.full((len(probabilities), len(OPERATIONAL_ROLES)), 0.001, dtype=np.float32)
    for row, values in enumerate(probabilities):
        for role, value in values.items():
            p[row, INDEX[role]] = value
        p[row] /= p[row].sum()
    return BlockExample(
        document_id="d", work_id="w", source="s", fold=0,
        role_probability=p, connector_probability=np.maximum(p[:, INDEX["CONTINUATION"]], p[:, INDEX["FILLER"]]),
        abs_indices=np.arange(len(p), dtype=np.uint32),
        char_lengths=np.full(len(p), 40, dtype=np.uint32),
        gold_roles=np.asarray([ROLE_TO_ID[value] for value in gold], dtype=np.uint8),
        trusted=np.ones(len(p), dtype=np.uint8),
    )


def test_non_bibliography_heading_is_a_candidate_barrier() -> None:
    item = example(
        [
            {"ENTRY": .9}, {"ENTRY": .9}, {"NON_BIB_HEADER": .9},
            {"ENTRY": .9}, {"ENTRY": .9},
        ],
        ["ENTRY", "ENTRY", "NON_BIB_HEADER", "ENTRY", "ENTRY"],
    )
    assert all(not (start < 2 < end) for start, end in candidate_spans(item, DecoderConfig()))


def test_heading_barrier_is_independent_of_entry_probability() -> None:
    item = example(
        [
            {"ENTRY": .9}, {"ENTRY": .9},
            {"ENTRY": .95, "NON_BIB_HEADER": .8},
            {"ENTRY": .9}, {"ENTRY": .9},
        ],
        ["ENTRY", "ENTRY", "NON_BIB_HEADER", "ENTRY", "ENTRY"],
    )
    config = DecoderConfig(heading_assignment_threshold=0.4)
    assert all(not (start < 2 < end) for start, end in candidate_spans(item, config))


def test_bibliography_header_can_only_be_the_start_of_a_candidate() -> None:
    item = example(
        [{"OTHER": .8}, {"BIB_HEADER": .9}, {"ENTRY": .9}, {"ENTRY": .9}],
        ["OTHER", "BIB_HEADER", "ENTRY", "ENTRY"],
    )
    spans = candidate_spans(item, DecoderConfig())
    assert spans
    assert all(start >= 1 for start, _ in spans)


def test_subheader_does_not_split_supported_entries() -> None:
    item = example(
        [{"ENTRY": .9}, {"ENTRY": .9}, {"BIB_SUBHEADER": .9}, {"ENTRY": .9}, {"ENTRY": .9}],
        ["ENTRY", "ENTRY", "BIB_SUBHEADER", "ENTRY", "ENTRY"],
    )
    spans = candidate_spans(item, DecoderConfig())
    assert (0, 4) in spans


def test_decoder_never_emits_span_without_two_entry_seeds() -> None:
    item = example(
        [{"BIB_HEADER": .9}, {"ENTRY": .9}, {"CONTINUATION": .9}],
        ["BIB_HEADER", "ENTRY", "CONTINUATION"],
    )
    model = StructuredModel(
        np.ones(len(FEATURE_NAMES), dtype=np.float32), DecoderConfig(), StructuredConfig(4, 1),
    )
    assert decode(item, model) == []
    assert not prediction_mask(3, []).any()
    assert not gold_is_seed_reachable(item, DecoderConfig())


def test_proposal_reachability_accounts_for_candidate_radius() -> None:
    item = example(
        [
            {"BIB_HEADER": .9}, {"CONTINUATION": .9}, {"CONTINUATION": .9},
            {"ENTRY": .9}, {"ENTRY": .9}, {"CONTINUATION": .9},
            {"CONTINUATION": .9}, {"CONTINUATION": .9}, {"CONTINUATION": .9},
        ],
        [
            "BIB_HEADER", "CONTINUATION", "CONTINUATION", "ENTRY", "ENTRY",
            "CONTINUATION", "CONTINUATION", "CONTINUATION", "CONTINUATION",
        ],
    )
    assert gold_is_seed_reachable(item, DecoderConfig(candidate_radius=3))
    assert not gold_is_proposal_reachable(item, DecoderConfig(candidate_radius=3))
    assert gold_is_proposal_reachable(item, DecoderConfig(candidate_radius=5))


def test_zero_bibliography_sequence_has_a_reachable_empty_path() -> None:
    item = example([{"OTHER": .9}, {"OTHER": .9}], ["OTHER", "OTHER"])
    assert gold_is_proposal_reachable(item, DecoderConfig())

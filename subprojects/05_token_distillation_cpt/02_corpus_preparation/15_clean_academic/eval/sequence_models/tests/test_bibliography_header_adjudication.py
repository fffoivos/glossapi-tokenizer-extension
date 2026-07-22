from __future__ import annotations

import pytest

from sequence_models.bibliography_header_adjudication import adjudicate


def _packet() -> dict:
    return {
        "cases": [
            {
                "candidate_id": "entry",
                "stratum": "internal_sparse_probe",
                "source": "source",
                "document_id": "doc",
                "abs_idx": 1,
                "text": "https://doi.org/example",
            },
            {
                "candidate_id": "header",
                "stratum": "exact_heading",
                "source": "source",
                "document_id": "doc",
                "abs_idx": 2,
                "text": "ΒΙΒΛΙΟΓΡΑΦΙΑ",
            },
            {
                "candidate_id": "disputed",
                "stratum": "block_start_probe",
                "source": "source",
                "document_id": "doc",
                "abs_idx": 3,
                "text": "PUBLICATIONS",
            },
        ]
    }


def _review(reviewer: str, labels: list[str]) -> dict:
    return {
        "reviewer": reviewer,
        "cases": [
            {
                "candidate_id": candidate_id,
                "label": label,
                "confidence": "high",
                "reason": "test",
            }
            for candidate_id, label in zip(
                ["entry", "header", "disputed"], labels, strict=True
            )
        ],
    }


def test_entry_header_and_disagreement_have_distinct_actions() -> None:
    result = adjudicate(
        _packet(),
        _review("a", ["ENTRY", "BIB_HEADER", "BIB_SUBHEADER"]),
        _review("b", ["ENTRY", "BIB_HEADER", "OTHER_STRUCTURE"]),
    )
    decisions = {row["candidate_id"]: row for row in result["decisions"]}
    assert decisions["entry"]["training_action"] == "KEEP_ENTRY_POSITIVE"
    assert decisions["entry"]["boundary_cue"] is None
    assert decisions["header"]["training_action"] == "MASK_HEADER_ENABLE_BOUNDARY_CUE"
    assert decisions["header"]["boundary_cue"] == "BIB_HEADER"
    assert decisions["disputed"]["training_action"] == "MASK_DISAGREEMENT_NO_CUE"
    assert decisions["disputed"]["boundary_cue"] is None


def test_exact_mask_audit_counts_any_entry_vote_as_a_failure() -> None:
    result = adjudicate(
        _packet(),
        _review("a", ["ENTRY", "BIB_HEADER", "BIB_SUBHEADER"]),
        _review("b", ["ENTRY", "ENTRY", "BIB_SUBHEADER"]),
    )
    assert result["conservative_exact_mask_audit"] == {
        "sample_count": 1,
        "entry_vote_count_from_either_reviewer": 1,
        "observed_entry_false_exclusion_rate": 1.0,
        "verified_scope": "sampled exact heading and exact subheading rules only",
    }


def test_rejects_mismatched_candidate_sets() -> None:
    review_b = _review("b", ["ENTRY", "BIB_HEADER", "OTHER_STRUCTURE"])
    review_b["cases"].pop()
    with pytest.raises(ValueError, match="candidate_id sets differ"):
        adjudicate(
            _packet(),
            _review("a", ["ENTRY", "BIB_HEADER", "BIB_SUBHEADER"]),
            review_b,
        )

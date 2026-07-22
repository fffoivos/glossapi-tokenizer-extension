from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
)
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.bibliography_role_adjudication import merge_reviews  # noqa: E402
from sequence_models.bibliography_role_review_runner import (  # noqa: E402
    make_batches,
    validate_review_payload,
)


def _case(case_id: str, block: str, start: int = 10) -> dict:
    return {
        "case_id": case_id,
        "block_case_id": block,
        "chunk_index": 0,
        "chunk_count": 1,
        "document_id": "doc",
        "work_id": "work",
        "source": "greek_phd",
        "n_physical_lines": 100,
        "lines": [
            {
                "line_id": f"line-{start}",
                "abs_idx": start,
                "document_position_percent": 10.0,
                "text": "1. Author A. 2020. Title.",
            },
            {
                "line_id": f"line-{start + 1}",
                "abs_idx": start + 1,
                "document_position_percent": 11.0,
                "text": "continued pages 1-2",
            },
        ],
    }


def _packet(cases: list[dict]) -> dict:
    return {
        "schema_version": "bibliography-role-review-packet-v1",
        "status": "ready_for_blind_contextual_review",
        "blinding": {
            "detector_features_hidden": True,
            "model_predictions_hidden": True,
            "nomination_strata_hidden": True,
            "original_region_labels_hidden": True,
        },
        "selection": {},
        "instructions": {},
        "cases": cases,
    }


def _review(case: dict, reviewer: str, roles=("ENTRY_ANCHOR", "CONTINUATION")) -> dict:
    return {
        "schema_version": "bibliography-role-review-v1",
        "reviewer": reviewer,
        "cases": [
            {
                "case_id": case["case_id"],
                "notes": "",
                "lines": [
                    {
                        "line_id": line["line_id"],
                        "abs_idx": line["abs_idx"],
                        "role": role,
                        "boundary_flag": "NONE",
                        "confidence": 0.9,
                        "reason": "fixture",
                    }
                    for line, role in zip(case["lines"], roles)
                ],
            }
        ],
    }


def _provenance(case: dict) -> dict:
    return {
        "schema_version": "bibliography-role-review-selection-v1",
        "cases": [
            {
                "document_id": case["document_id"],
                "review_chunks": [
                    {
                        "context_source_labels": [
                            {
                                "line_id": line["line_id"],
                                "abs_idx": line["abs_idx"],
                                "original_region_label": "BIB",
                            }
                            for line in case["lines"]
                        ]
                    }
                ],
            }
        ],
    }


def test_pass_order_and_batch_contracts_are_distinct() -> None:
    cases = [_case(f"case-{index}", f"block-{index}", 10 + 10 * index) for index in range(5)]
    common = {
        "model": "gpt-5.6-sol",
        "prompt_sha256": "a" * 64,
        "output_schema_sha256": "b" * 64,
        "batch_size": 2,
    }
    a = make_batches(cases, pass_id="pass-a", reviewer_id="a", **common)
    b = make_batches(cases, pass_id="pass-b", reviewer_id="b", **common)
    assert [len(row["cases"]) for row in a] == [2, 2, 1]
    assert [row["batch_id"] for row in a] != [row["batch_id"] for row in b]
    assert [case["case_id"] for row in a for case in row["cases"]] != [
        case["case_id"] for row in b for case in row["cases"]
    ]


def test_review_validation_canonicalizes_order_and_fails_on_invented_id() -> None:
    case = _case("case", "block")
    batch = make_batches(
        [case], pass_id="pass-a", reviewer_id="a", model="gpt-5.6-sol",
        prompt_sha256="a" * 64, output_schema_sha256="b" * 64, batch_size=1,
    )[0]
    review = _review(case, "a")
    assert validate_review_payload(batch, review, reviewer_id="a")["reviewer"] == "a"
    review["cases"][0]["lines"].reverse()
    normalized = validate_review_payload(batch, review, reviewer_id="a")
    assert [line["line_id"] for line in normalized["cases"][0]["lines"]] == [
        line["line_id"] for line in case["lines"]
    ]
    review["cases"][0]["lines"][0]["line_id"] = "invented"
    with pytest.raises(ValueError, match="omits or invents"):
        validate_review_payload(batch, review, reviewer_id="a")


def test_review_validation_repairs_near_copy_id_only_with_exact_coordinates() -> None:
    case = _case("case", "block")
    case["lines"][0]["line_id"] = "a" * 35 + "135" + "b" * 26
    batch = make_batches(
        [case], pass_id="pass-a", reviewer_id="a", model="gpt-5.6-sol",
        prompt_sha256="a" * 64, output_schema_sha256="b" * 64, batch_size=1,
    )[0]
    review = _review(case, "a")
    review["cases"][0]["lines"][0]["line_id"] = "a" * 35 + "b" * 26
    normalized = validate_review_payload(batch, review, reviewer_id="a")
    assert normalized["cases"][0]["lines"][0]["line_id"] == case["lines"][0]["line_id"]

    review["cases"][0]["lines"][0]["abs_idx"] += 1
    with pytest.raises(ValueError, match="omits or invents"):
        validate_review_payload(batch, review, reviewer_id="a")


def test_merge_agrees_role_and_boundary_independently() -> None:
    case = _case("case", "block")
    packet = _packet([case])
    a = _review(case, "a")
    b = _review(case, "b", roles=("ENTRY_ANCHOR", "FILLER"))
    b["cases"][0]["lines"][0]["boundary_flag"] = "HARD_STOP"
    overlays, report = merge_reviews(packet, _provenance(case), a, b)
    first, second = overlays
    assert first["role"] == "ENTRY_ANCHOR"
    assert first["role_status"] == "AGREED_REVIEW"
    assert first["boundary_status"] == "UNRESOLVED"
    assert second["role"] == "UNKNOWN"
    assert second["role_status"] == "UNRESOLVED"
    assert second["boundary_status"] == "AGREED_REVIEW"
    assert report["entry_seed_eligibility_agreement"] == 1.0
    assert report["exact_role_agreement"] == 0.5


def test_overlap_conflict_within_reviewer_is_unresolved() -> None:
    first = _case("case-1", "block", 10)
    second = dict(first)
    second["case_id"] = "case-2"
    second["chunk_index"] = 1
    second["chunk_count"] = 2
    first["chunk_count"] = 2
    packet = _packet([first, second])
    provenance = _provenance(first)
    a = _review(first, "a")
    a["cases"].append(_review(second, "a", roles=("FILLER", "CONTINUATION"))["cases"][0])
    b = _review(first, "b")
    b["cases"].append(_review(second, "b")["cases"][0])
    overlays, _ = merge_reviews(packet, provenance, a, b)
    assert overlays[0]["role_status"] == "UNRESOLVED"
    assert overlays[0]["raw_role_votes"]["a"] == ["ENTRY_ANCHOR", "FILLER"]

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
)
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.bibliography_role_artifacts import subset_packet, union_overlays  # noqa: E402
from sequence_models.bibliography_role_readiness import evaluate  # noqa: E402


def _case(case_id: str, block_id: str, source: str = "kallipos") -> dict:
    return {
        "case_id": case_id, "block_case_id": block_id, "chunk_index": 0,
        "chunk_count": 1, "document_id": f"doc-{block_id}", "work_id": f"work-{block_id}",
        "source": source, "n_physical_lines": 10,
        "lines": [{"line_id": f"line-{case_id}", "abs_idx": 1,
                   "document_position_percent": 10.0, "text": "citation"}],
    }


def _provenance(case: dict, stratum: str) -> dict:
    return {
        "block_case_id": case["block_case_id"], "source": case["source"],
        "bootstrap_stratum": stratum, "work_id": case["work_id"],
        "review_chunks": [{"case_id": case["case_id"], "context_source_labels": []}],
    }


def _overlay(line: str, *, role: str = "ENTRY_ANCHOR", status: str = "AGREED_REVIEW") -> dict:
    return {
        "schema_version": "bibliography-role-overlay-v2", "document_id": "doc",
        "work_id": "work", "line_id": line, "abs_idx": 1, "text_sha256": "a" * 64,
        "original_region_label": "BIB", "role": role, "role_status": status,
        "boundary_flag": "NONE", "boundary_status": status,
        "role_confidence": 1.0, "boundary_confidence": 1.0, "label_origin": "fixture",
        "reviewers": [status], "review_case_ids": ["case"],
        "raw_role_votes": {status: [role]}, "raw_boundary_votes": {status: ["NONE"]},
    }


def test_targeted_subset_keeps_complete_selected_blocks() -> None:
    filler, continuation, random = (
        _case("f", "bf"), _case("c", "bc"), _case("r", "br")
    )
    packet = {"schema_version": "bibliography-role-review-packet-v1", "blinding": {},
              "selection": {}, "instructions": {}, "status": "ready",
              "cases": [filler, continuation, random]}
    provenance = {
        "schema_version": "bibliography-role-review-selection-v1",
        "cases": [_provenance(filler, "underrepresented_filler_proxy"),
                  _provenance(continuation, "underrepresented_continuation_proxy"),
                  _provenance(random, "random")],
    }
    subset, selected_provenance, report = subset_packet(
        packet, provenance, source="kallipos",
        strata=("underrepresented_filler_proxy", "underrepresented_continuation_proxy"),
        expected_block_count=2,
    )
    assert {row["case_id"] for row in subset["cases"]} == {"f", "c"}
    assert len(selected_provenance["cases"]) == report["block_count"] == 2


def test_overlay_union_prefers_human_and_rejects_trusted_ties() -> None:
    automatic, human = _overlay("line"), _overlay("line", role="CONTINUATION", status="ADJUDICATED")
    rows, report = union_overlays((("auto", [automatic]), ("human", [human])))
    assert rows[0]["role"] == "CONTINUATION"
    assert rows[0]["role_status"] == "ADJUDICATED"
    assert report["adjudicated_precedence_count"] == 1
    conflicting = _overlay("line", role="FILLER")
    with pytest.raises(ValueError, match="equal-trust"):
        union_overlays((("left", [automatic]), ("right", [conflicting])))


def test_readiness_header_gate_is_global_with_two_source_coverage() -> None:
    rows = []
    for source, count in (("greek_phd", 50), ("openarchives", 50), ("kallipos", 0)):
        rows.extend(
            {"_source": source, "role": "HEADER", "role_status": "AGREED_REVIEW",
             "boundary_flag": "NONE", "boundary_status": "AGREED_REVIEW"}
            for _ in range(count)
        )
    report = evaluate(rows)
    gates = {row["gate"]: row for row in report["gates"]}
    assert gates["HEADER_OR_SUBHEADER:overall"]["passed"] is True
    assert gates["HEADER_OR_SUBHEADER:sources_ge_30"]["passed"] is True
    assert not any(row["gate"] == "HEADER_OR_SUBHEADER:kallipos" for row in report["gates"])

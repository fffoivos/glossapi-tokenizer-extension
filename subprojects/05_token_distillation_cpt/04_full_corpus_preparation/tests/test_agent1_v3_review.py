from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module():
    path = HERE / "scripts" / "agent1_v3_review.py"
    spec = importlib.util.spec_from_file_location("agent1_v3_review_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REVIEW = load_module()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def uid(index: int, source: str = "source-a") -> str:
    return digest(f"{source}:{index}")


def roster(*sources: str) -> dict:
    return {
        "schema_version": "agent1_full_corpus_v3_candidate_roster_v1",
        "candidate_source_ids": list(sources),
        "review_routes": {source: "pdf_ocr" for source in sources},
        "inventory_only_exclusions": [
            {"repo_id": "metadata-only", "reason": "metadata_only_no_training_text"}
        ],
    }


def metric_rows(count: int, *, source: str = "source-a") -> list[dict]:
    return [
        {
            "source_id": source,
            "source_dataset": source,
            "source_revision": "a" * 40,
            "stable_uid": uid(index, source),
            "review_route": "pdf_ocr",
            "review_risk_score": float(index),
            "structural_template_id": f"cluster-{index % 7}",
        }
        for index in range(count)
    ]


def sample() -> dict:
    return {
        "source_id": "source-a",
        "source_dataset": "source-a",
        "source_revision": "a" * 40,
        "stable_uid": uid(1),
        "source_route": "pdf_ocr",
        "sampling_stratum": "random",
    }


def request(slot: str = "primary") -> dict:
    text = "Καθαρό κείμενο χωρίς προσωπικό στοιχείο."
    return REVIEW.make_review_request(
        sample(),
        reviewer_slot=slot,
        original_text_sha256=digest("original"),
        review_copy_sha256=digest(text),
        prompt_sha256=digest("prompt"),
        response_schema_sha256=digest("schema"),
        model="gpt-5.6-luna",
        code_commit="b" * 40,
        review_copy=text,
    )


def response_for(req: dict, **overrides: object) -> dict:
    result = {
        "schema_version": "agent1_v3_review_response_v1",
        **{
            key: req[key]
            for key in (
                "review_id",
                "sample_id",
                "reviewer_slot",
                "source_id",
                "source_dataset",
                "source_revision",
                "source_route",
                "sampling_stratum",
                "original_text_sha256",
                "review_copy_sha256",
                "prompt_sha256",
                "response_schema_sha256",
                "model",
                "code_commit",
                "attempt",
            )
        },
        "cleanliness_score": 5,
        "quality_score": 4,
        "diversity_contribution_score": 3,
        "issues": [],
        "recommendation": "include",
        "confidence_score": 5,
        "evidence": "Συνεκτικό ελληνικό κείμενο χωρίς εμφανές πρόβλημα εξαγωγής.",
    }
    result.update(overrides)
    return result


def test_review_copy_masks_verified_identifiers_without_moving_offsets() -> None:
    text = (
        "Επικοινωνία: maria@example.gr\n"
        "IP 192.168.1.25, IBAN GR16 0110 1250 0000 0001 2300 695.\n"
        "ΑΦΜ: 094259216, ΑΜΚΑ: 01019012345, ΑΔΤ: ΑΒ 123456.\n"
        "Τηλ. +30 210 123 4567 και 6912345678.\n"
        "Το μη έγκυρο ΑΦΜ: 123456789 και IBAN GR0000000000000000000000000 μένουν.\n"
    )
    review_copy, report = REVIEW.redact_review_copy(text)
    assert len(review_copy) == len(text)
    assert [index for index, char in enumerate(text) if char in "\r\n"] == [
        index for index, char in enumerate(review_copy) if char in "\r\n"
    ]
    for secret in (
        "maria@example.gr",
        "192.168.1.25",
        "GR16 0110 1250 0000 0001 2300 695",
        "094259216",
        "01019012345",
        "ΑΒ 123456",
        "+30 210 123 4567",
        "6912345678",
    ):
        assert secret not in review_copy
    assert "123456789" in review_copy
    assert "GR0000000000000000000000000" in review_copy
    assert report["offset_unit"] == "unicode_codepoints"
    assert report["redaction_counts"] == {
        "afm": 1,
        "amka": 1,
        "email": 1,
        "iban": 1,
        "identity_or_passport": 1,
        "ip": 1,
        "phone": 2,
    }
    for span in report["redaction_spans"]:
        assert review_copy[span["char_start"] : span["char_end"]] == "█" * (
            span["char_end"] - span["char_start"]
        )


def test_review_copy_masks_compressed_ipv6_but_not_a_clock() -> None:
    review_copy, report = REVIEW.redact_review_copy("IPv6 fe80::1, ώρα 12:34:56")
    assert "fe80::1" not in review_copy
    assert "12:34:56" in review_copy
    assert report["redaction_counts"] == {"ip": 1}


def test_candidate_roster_requires_exact_route_coverage() -> None:
    good = roster("source-a", "source-b")
    good["review_routes"]["source-b"] = "mixed"
    report = REVIEW.validate_candidate_roster_routes(good)
    assert report["candidate_count"] == 2
    del good["review_routes"]["source-b"]
    with pytest.raises(ValueError, match="coverage mismatch"):
        REVIEW.validate_candidate_roster_routes(good)


def test_sampling_is_deterministic_disjoint_and_records_full_denominator() -> None:
    rows = metric_rows(140)
    first = REVIEW.build_sample_manifest(rows, roster("source-a"), seed="frozen-seed")
    second = REVIEW.build_sample_manifest(reversed(rows), roster("source-a"), seed="frozen-seed")
    assert first == second
    selected = first["selected_documents"]
    assert len(selected) == 100
    assert len({row["stable_uid"] for row in selected}) == 100
    assert Counter(row["sampling_stratum"] for row in selected) == {
        "random": 60,
        "risk": 20,
        "cluster": 20,
    }
    expected_risk = {
        row["stable_uid"]
        for row in sorted(rows, key=lambda row: -row["review_risk_score"])[:20]
    }
    actual_risk = {row["stable_uid"] for row in selected if row["sampling_stratum"] == "risk"}
    assert actual_risk == expected_risk
    source = first["sources"][0]
    assert source["review_denominator"]["eligible_document_count"] == 140
    assert source["review_denominator"]["selection_is_exhaustive"] is False
    assert source["review_denominator"]["minimum_requirement_status"] == "met"
    assert source["eligible_inventory_sha256"]
    assert source["selected_inventory_sha256"]


def test_large_and_small_sources_have_exact_or_exhaustive_evidence() -> None:
    large = REVIEW.build_sample_manifest(
        metric_rows(250),
        roster("source-a"),
        seed="frozen-seed",
        large_or_heterogeneous_sources=["source-a"],
    )
    assert Counter(row["sampling_stratum"] for row in large["selected_documents"]) == {
        "random": 100,
        "risk": 50,
        "cluster": 50,
    }
    small = REVIEW.build_sample_manifest(metric_rows(17), roster("source-a"), seed="frozen-seed")
    source = small["sources"][0]
    assert len(small["selected_documents"]) == 17
    assert source["review_denominator"] == {
        "eligible_document_count": 17,
        "minimum_required_documents": 100,
        "configured_review_target": 100,
        "selected_unique_documents": 17,
        "selection_is_exhaustive": True,
        "minimum_requirement_status": "unattainable_exhaustive",
        "target_status": "unattainable_exhaustive",
        "denominator_exception": "eligible_inventory_below_100_all_documents_selected",
    }


def test_sampling_rejects_missing_roster_source_without_explicit_override() -> None:
    with pytest.raises(ValueError, match="missing candidate roster sources"):
        REVIEW.build_sample_manifest(metric_rows(100), roster("source-a", "source-b"), seed="frozen-seed")


def test_fallback_risk_is_source_route_aware() -> None:
    metrics = {"original_characters": 500, "raw_html_tags_per_1000_chars": 3.5}
    assert REVIEW.risk_score_from_metrics(metrics, source_route="html_web") > 3
    assert REVIEW.risk_score_from_metrics(metrics, source_route="pdf_ocr") == 0
    structured = {
        "original_characters": 500,
        "structured_missing_field_count": 2,
        "repeated_parent_context_fraction": 0.5,
    }
    assert REVIEW.risk_score_from_metrics(structured, source_route="structured") > 3


def test_secondary_selection_is_deterministically_stratified() -> None:
    manifest = REVIEW.build_sample_manifest(metric_rows(140), roster("source-a"), seed="frozen-seed")
    first = REVIEW.select_secondary_samples(manifest["selected_documents"], seed="second-seed")
    second = REVIEW.select_secondary_samples(reversed(manifest["selected_documents"]), seed="second-seed")
    assert first == second
    assert Counter(row["sampling_stratum"] for row in first) == {
        "random": 6,
        "risk": 2,
        "cluster": 2,
    }
    assert {row["stable_uid"] for row in first} <= {
        row["stable_uid"] for row in manifest["selected_documents"]
    }
    assert {row["reviewer_slot"] for row in first} == {"secondary"}


def test_strict_response_requires_one_to_five_and_exact_request_binding() -> None:
    req = request()
    valid = response_for(req)
    assert REVIEW.validate_review_response(valid, req) == []
    bad_score = response_for(req, quality_score=0)
    assert any("quality_score must be an integer in [1, 5]" in error for error in REVIEW.validate_review_response(bad_score, req))
    bad_extra = response_for(req, unexpected=True)
    assert any("unexpected fields" in error for error in REVIEW.validate_review_response(bad_extra, req))
    bad_identity = response_for(req, source_revision="c" * 40)
    assert any("response/request identity drift: source_revision" == error for error in REVIEW.validate_review_response(bad_identity, req))


def test_adjudication_manifest_blocks_low_confidence_and_material_disagreement() -> None:
    primary_request = request("primary")
    secondary_request = request("secondary")
    primary = response_for(primary_request, quality_score=5, confidence_score=5)
    secondary = response_for(
        secondary_request,
        quality_score=2,
        confidence_score=2,
        recommendation="exclude",
        issues=[{"code": "ocr_corruption", "severity_score": 4}],
    )
    pending = REVIEW.build_adjudication_manifest(
        [primary_request, secondary_request], [primary, secondary]
    )
    assert pending["pending_count"] == 1
    case = pending["cases"][0]
    assert case["status"] == "pending_adjudication"
    assert "secondary_low_confidence" in case["reasons"]
    assert "material_disagreement:recommendation" in case["reasons"]
    assert "material_disagreement:quality_score" in case["reasons"]
    adjudication = response_for(
        case["adjudication_request"],
        reviewer_slot="adjudicator",
        recommendation="include_after_cleaning",
        confidence_score=5,
    )
    complete = REVIEW.build_adjudication_manifest(
        [primary_request, secondary_request], [primary, secondary, adjudication]
    )
    assert complete["pending_count"] == 0
    assert complete["status"] == "complete"
    REVIEW.assert_adjudication_closed(complete)


def test_response_schema_matches_strict_v3_contract() -> None:
    schema = json.loads(
        (HERE / "schemas" / "agent1_v3_review_response.schema.json").read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["quality_score"]["$ref"] == "#/$defs/score"
    assert schema["$defs"]["score"] == {"type": "integer", "minimum": 1, "maximum": 5}
    assert "confidence_score" in schema["required"]

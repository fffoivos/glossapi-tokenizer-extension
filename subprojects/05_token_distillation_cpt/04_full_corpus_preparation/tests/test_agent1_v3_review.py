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
        "extraction_route": "pdf_ocr",
        "observed_extraction_route": "pdf_ocr",
        "observed_extraction_route_basis": "declared_extraction_route_fallback",
        "observed_extraction_route_evidence": "roster:extraction_route",
        "observed_extraction_route_priority": "logical_primary",
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
                "extraction_route",
                "observed_extraction_route",
                "observed_extraction_route_basis",
                "observed_extraction_route_evidence",
                "observed_extraction_route_priority",
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
    html_score = REVIEW.risk_score_from_metrics(metrics, source_route="html_web")
    pdf_score = REVIEW.risk_score_from_metrics(metrics, source_route="pdf_ocr")
    assert html_score > 3
    # A visible HTML defect remains a bounded secondary signal for a logical
    # PDF/OCR source; it must not displace its primary diagnostic model.
    assert 0 < pdf_score < html_score
    structured = {
        "original_characters": 500,
        "structured_missing_field_count": 2,
        "repeated_parent_context_fraction": 0.5,
    }
    assert REVIEW.risk_score_from_metrics(structured, source_route="structured") > 3


def test_logical_source_route_drives_selection_with_extraction_as_secondary_signal() -> None:
    explicit_roster = roster("source-a")
    explicit_roster.update(
        {
            "source_routes": {"source-a": "pdf_ocr"},
            "review_routes": {"source-a": "pdf_ocr"},
            "extraction_routes": {"source-a": "html_web"},
            "route_policy": {"priority": "logical_source_then_observed_extraction"},
        }
    )
    metrics = {
        "original_characters": 500,
        "raw_html_tags_per_1000_chars": 4.0,
        "raw_mojibake_per_1000_chars": 1.0,
    }
    logical_pdf_score = REVIEW.risk_score_from_metrics(
        metrics, source_route="pdf_ocr", extraction_route="html_web"
    )
    logical_html_score = REVIEW.risk_score_from_metrics(
        metrics, source_route="html_web", extraction_route="pdf_ocr"
    )
    assert logical_pdf_score > REVIEW.risk_score_from_metrics(metrics, source_route="pdf_ocr")
    assert logical_pdf_score < logical_html_score

    row = {
        **metric_rows(1)[0],
        **metrics,
        "source_route": "pdf_ocr",
        "review_route": "pdf_ocr",
        "extraction_route": "html_web",
    }
    normalized = REVIEW.normalize_metric_rows([row], explicit_roster)
    assert normalized[0].source_route == "pdf_ocr"
    assert normalized[0].review_route == "pdf_ocr"
    assert normalized[0].extraction_route == "html_web"

    manifest = REVIEW.build_sample_manifest([row], explicit_roster, seed="frozen-seed")
    assert manifest["sources"][0]["source_route"] == "pdf_ocr"
    assert manifest["sources"][0]["extraction_route"] == "html_web"
    assert manifest["selected_documents"][0]["source_route"] == "pdf_ocr"
    assert manifest["selected_documents"][0]["review_route"] == "pdf_ocr"
    assert manifest["selected_documents"][0]["extraction_route"] == "html_web"


def test_explicit_risk_base_cannot_bypass_logical_route_or_mixed_extraction_diagnostics() -> None:
    metrics = {
        "review_risk_score": 2.0,
        "original_characters": 500,
        "raw_html_tags_per_1000_chars": 4.0,
        "raw_mojibake_per_1000_chars": 1.0,
    }
    logical_pdf = REVIEW.risk_score_from_metrics(metrics, source_route="pdf_ocr")
    observed_mixed = REVIEW.risk_score_from_metrics(
        metrics, source_route="pdf_ocr", extraction_route="mixed"
    )
    logical_html = REVIEW.risk_score_from_metrics(metrics, source_route="html_web")
    assert logical_pdf > metrics["review_risk_score"]
    assert observed_mixed > logical_pdf
    assert observed_mixed < logical_html


def test_school_books_logical_modes_keep_html_and_structure_secondary_to_pdf() -> None:
    """PDF-linked school-book fields must not invent HTML/structured primacy."""

    school_modes = ["pdf_ocr"]
    html_only = {
        "original_characters": 1000,
        "raw_html_entity_per_1000_chars": 4.0,
        "raw_script_style_tag_count": 1,
        "raw_navigation_markup_tag_count": 2,
    }
    pdf_only = {
        "original_characters": 1000,
        "raw_line_break_hyphenation_fraction": 1.0,
        "raw_repeated_short_line_fraction": 1.0,
    }
    structured_only = {
        "original_characters": 1000,
        "structured_missing_required_field_count": 3,
    }

    school_html = REVIEW.risk_score_from_metrics(
        html_only,
        source_route="pdf_ocr",
        logical_error_modes=school_modes,
        observed_extraction_route="pdf_ocr",
    )
    primary_html = REVIEW.risk_score_from_metrics(
        html_only,
        source_route="html_web",
        logical_error_modes=["html_web"],
    )
    school_pdf = REVIEW.risk_score_from_metrics(
        pdf_only,
        source_route="pdf_ocr",
        logical_error_modes=school_modes,
        observed_extraction_route="pdf_ocr",
    )
    html_pdf = REVIEW.risk_score_from_metrics(
        pdf_only,
        source_route="html_web",
        logical_error_modes=["html_web"],
    )
    school_structured = REVIEW.risk_score_from_metrics(
        structured_only,
        source_route="pdf_ocr",
        logical_error_modes=school_modes,
        observed_extraction_route="pdf_ocr",
    )
    structured_primary = REVIEW.risk_score_from_metrics(
        structured_only,
        source_route="structured",
        logical_error_modes=["structured"],
    )
    assert school_html < primary_html
    assert school_pdf > html_pdf
    assert school_structured < structured_primary

    school_roster = roster("school_books_new_editions")
    school_roster.update(
        {
            "source_routes": {"school_books_new_editions": "pdf_ocr"},
            "review_routes": {"school_books_new_editions": "pdf_ocr"},
            "extraction_routes": {"school_books_new_editions": "pdf_ocr"},
            "logical_error_modes": {"school_books_new_editions": school_modes},
            "route_policy": {"priority": "logical_source_then_observed_extraction"},
        }
    )
    metric = {
        **metric_rows(1, source="school_books_new_editions")[0],
        **html_only,
        "source_route": "pdf_ocr",
        "review_route": "pdf_ocr",
        "extraction_route": "pdf_ocr",
        "observed_extraction_route": "pdf_ocr",
        "observed_extraction_route_basis": "explicit_row_route",
        "observed_extraction_route_evidence": "raw_field:format",
        "observed_extraction_route_priority": "logical_primary",
    }
    normalized = REVIEW.normalize_metric_rows([metric], school_roster)
    assert normalized[0].risk_score == school_html


def test_structured_depth_requires_a_concrete_flattening_failure_to_affect_risk() -> None:
    base = {
        "original_characters": 1000,
        "structured_metadata_max_depth": 12,
    }
    depth_only = REVIEW.risk_score_from_metrics(
        base,
        source_route="structured",
        logical_error_modes=["structured"],
    )
    flat = REVIEW.risk_score_from_metrics(
        {"original_characters": 1000},
        source_route="structured",
        logical_error_modes=["structured"],
    )
    with_flattening_failure = REVIEW.risk_score_from_metrics(
        {**base, "field_flattening_failures": 1},
        source_route="structured",
        logical_error_modes=["structured"],
    )
    assert depth_only == flat
    assert with_flattening_failure > depth_only


def test_metric_rows_reject_logical_or_extraction_route_drift() -> None:
    explicit_roster = roster("source-a")
    explicit_roster.update(
        {
            "source_routes": {"source-a": "pdf_ocr"},
            "extraction_routes": {"source-a": "html_web"},
            "route_policy": {"priority": "logical_source_then_observed_extraction"},
        }
    )
    valid = {
        **metric_rows(1)[0],
        "source_route": "pdf_ocr",
        "review_route": "pdf_ocr",
        "extraction_route": "html_web",
    }
    source_drift = {**valid, "source_route": "html_web"}
    with pytest.raises(ValueError, match="source_route drift"):
        REVIEW.normalize_metric_rows([source_drift], explicit_roster)
    extraction_drift = {**valid, "extraction_route": "pdf_ocr"}
    with pytest.raises(ValueError, match="extraction_route drift"):
        REVIEW.normalize_metric_rows([extraction_drift], explicit_roster)


def test_document_observed_route_requires_documented_secondary_exception() -> None:
    documented = roster("source-a")
    documented.update(
        {
            "source_routes": {"source-a": "pdf_ocr"},
            "review_routes": {"source-a": "pdf_ocr"},
            "extraction_routes": {"source-a": "pdf_ocr"},
            "route_policy": {"priority": "logical_source_then_observed_extraction"},
            "route_basis": {
                "schema_version": "agent1_v3_source_route_basis_v1",
                "priority": "logical_source_then_observed_extraction",
                "sources": {
                    "source-a": {
                        "logical_acquisition_type": "pdf_ocr",
                        "rationale": "PDF extraction is the logical source.",
                        "expected_observed_extraction_exceptions": [
                            {
                                "observed_extraction_route": "html_web",
                                "rationale": "A repository shell can remain in text.",
                                "secondary_only": True,
                            }
                        ],
                    }
                },
            },
        }
    )
    observed_html = {
        **metric_rows(1)[0],
        "source_route": "pdf_ocr",
        "review_route": "pdf_ocr",
        "extraction_route": "pdf_ocr",
        "observed_extraction_route": "html_web",
        "observed_extraction_route_basis": "row_representation_metadata",
        "observed_extraction_route_evidence": "raw_metadata:mime_type=text_html",
    }
    normalized = REVIEW.normalize_metric_rows([observed_html], documented)
    assert normalized[0].source_route == "pdf_ocr"
    assert normalized[0].observed_extraction_route == "html_web"
    assert normalized[0].observed_extraction_route_priority == "secondary_exception_only"

    undocumented = {**observed_html, "observed_extraction_route": "structured"}
    with pytest.raises(ValueError, match="documented secondary exception"):
        REVIEW.normalize_metric_rows([undocumented], documented)

    false_declared_fallback = {
        **observed_html,
        "observed_extraction_route_basis": "declared_extraction_route_fallback",
        "observed_extraction_route_evidence": "roster:extraction_route",
    }
    with pytest.raises(ValueError, match="declared extraction route fallback must equal"):
        REVIEW.normalize_metric_rows([false_declared_fallback], documented)

    unavailable_with_route = {
        **observed_html,
        "observed_extraction_route": "pdf_ocr",
        "observed_extraction_route_basis": "unavailable",
        "observed_extraction_route_evidence": "none",
        "observed_extraction_route_priority": "logical_primary",
    }
    with pytest.raises(ValueError, match="unavailable observed extraction cannot carry"):
        REVIEW.normalize_metric_rows([unavailable_with_route], documented)


def test_request_provenance_rejects_false_declared_fallback_or_unavailable_route() -> None:
    false_declared_fallback = {
        **sample(),
        "extraction_route": "pdf_ocr",
        "observed_extraction_route": "html_web",
        "observed_extraction_route_basis": "declared_extraction_route_fallback",
        "observed_extraction_route_evidence": "roster:extraction_route",
        "observed_extraction_route_priority": "secondary_exception_only",
    }
    with pytest.raises(ValueError, match="declared extraction route fallback must equal"):
        REVIEW._request_observed_route_fields(
            false_declared_fallback,
            source_route="pdf_ocr",
            extraction_route="pdf_ocr",
        )

    unavailable_with_route = {
        **sample(),
        "observed_extraction_route_basis": "unavailable",
        "observed_extraction_route_evidence": "none",
    }
    with pytest.raises(ValueError, match="unavailable observed extraction route cannot carry"):
        REVIEW._request_observed_route_fields(
            unavailable_with_route,
            source_route="pdf_ocr",
            extraction_route="pdf_ocr",
        )


def test_observed_secondary_risk_bonus_is_strictly_bounded() -> None:
    metrics = {
        "original_characters": 1000,
        "raw_mojibake_per_1000_chars": 4.0,
        "raw_control_per_1000_chars": 3.0,
        "raw_repeated_line_fraction": 1.0,
        "raw_one_token_line_fraction": 1.0,
        "cleaner_removed_character_fraction": 1.0,
        "cleaner_badness_score": 3.0,
        "toc_header_detected": True,
    }
    logical_html = REVIEW.risk_score_from_metrics(metrics, source_route="html_web")
    observed_pdf = REVIEW.risk_score_from_metrics(
        metrics,
        source_route="html_web",
        observed_extraction_route="pdf_ocr",
    )
    assert 0 < observed_pdf - logical_html <= REVIEW.MAX_OBSERVED_SECONDARY_RISK_BONUS


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
    bad_priority = response_for(req, observed_extraction_route_priority="secondary_exception_only")
    assert any(
        "observed_extraction_route_priority reverses logical-source priority" in error
        for error in REVIEW.validate_review_response(bad_priority, req)
    )


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

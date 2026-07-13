from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agent1_v3_admission as ADMISSION
import agent1_v3_review as REVIEW
import agent1_v3_review_aggregate as AGGREGATE


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def binding(path: Path, *, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def review_response(request: dict[str, object], *, recommendation: str = "include") -> dict[str, object]:
    return {
        key: request[key]
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
    } | {
        "schema_version": REVIEW.RESPONSE_SCHEMA,
        "cleanliness_score": 4,
        "quality_score": 5,
        "diversity_contribution_score": 3,
        "issues": [{"code": "template_replay", "severity_score": 2}],
        "recommendation": recommendation,
        "confidence_score": 5,
        "evidence": "Compact packet is coherent Greek with a bounded template concern.",
    }


def make_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Write a complete one-source Stage30/35 fixture without corpus text."""

    roster_path = tmp_path / "roster.json"
    roster = {
        "schema_version": REVIEW.ROSTER_SCHEMA,
        "candidate_source_ids": ["source-a"],
        "review_routes": {"source-a": "structured"},
        "inventory_only_exclusions": [],
    }
    write_json(roster_path, roster)
    route_validation = REVIEW.validate_candidate_roster_routes(roster)

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Review the compact sample.", encoding="utf-8")
    schema_path = tmp_path / "schema.json"
    write_json(schema_path, {"type": "object", "additionalProperties": False})
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, {"schema_version": REVIEW.POLICY_SCHEMA, "review": {"required_model": "gpt-5.6-luna"}})
    sample_id = digest("source-a:sample")
    original_sha = digest("source-a:original")
    review_copy = "Σύντομο ελληνικό κείμενο."
    prompt_sha = binding(prompt_path)["sha256"]
    schema_sha = binding(schema_path)["sha256"]
    source_revision = "a" * 40
    selected = {
        "source_id": "source-a",
        "source_dataset": "source-a",
        "source_revision": source_revision,
        "stable_uid": sample_id,
        "source_route": "structured",
        "sampling_stratum": "random",
        "risk_score": 0.0,
        "review_cluster_id": "singleton:" + sample_id,
        "review_cluster_size": 1,
        "selection_rank": 1,
    }
    denominator = {
        "eligible_document_count": 1,
        "minimum_required_documents": 100,
        "configured_review_target": 100,
        "selected_unique_documents": 1,
        "selection_is_exhaustive": True,
        "minimum_requirement_status": "unattainable_exhaustive",
        "target_status": "unattainable_exhaustive",
        "denominator_exception": "eligible_inventory_below_100_all_documents_selected",
    }
    source_selection = {
        "source_id": "source-a",
        "source_dataset": "source-a",
        "source_revision": source_revision,
        "source_route": "structured",
        "large_or_heterogeneous": False,
        "review_denominator": denominator,
        "requested_strata": {"random": 1, "risk": 0, "cluster": 0},
        "actual_strata": {"random": 1, "risk": 0, "cluster": 0},
        "eligible_inventory_sha256": digest("eligible"),
        "selected_inventory_sha256": digest("selected"),
    }
    selection = {
        "schema_version": REVIEW.SAMPLE_MANIFEST_SCHEMA,
        "seed": "seed",
        "candidate_roster_sha256": route_validation["roster_sha256"],
        "full_scan_metrics_sha256": digest("metrics"),
        "route_validation": route_validation,
        "sources": [source_selection],
        "selected_documents": [selected],
        "selected_document_count": 1,
        "missing_candidate_sources": [],
    }
    selection["manifest_sha256"] = REVIEW.sha256_json(selection)
    primary = REVIEW.make_review_request(
        selected,
        reviewer_slot="primary",
        original_text_sha256=original_sha,
        review_copy_sha256=digest(review_copy),
        prompt_sha256=str(prompt_sha),
        response_schema_sha256=str(schema_sha),
        model="gpt-5.6-luna",
        code_commit="b" * 40,
        review_copy=review_copy,
        comparison_bundle=[],
    )
    secondary = REVIEW.make_review_request(
        selected,
        reviewer_slot="secondary",
        original_text_sha256=original_sha,
        review_copy_sha256=digest(review_copy),
        prompt_sha256=str(prompt_sha),
        response_schema_sha256=str(schema_sha),
        model="gpt-5.6-luna",
        code_commit="b" * 40,
        review_copy=review_copy,
        comparison_bundle=[],
    )
    requests_path = tmp_path / "requests.jsonl"
    write_jsonl(requests_path, [primary, secondary])
    coverage = {
        "source_id": "source-a",
        "source_dataset": "source-a",
        "source_revision": source_revision,
        "source_route": "structured",
        "review_denominator": denominator,
        "requested_strata": {"random": 1, "risk": 0, "cluster": 0},
        "primary_requests_by_stratum": {"random": 1, "risk": 0, "cluster": 0},
        "secondary_requests_by_stratum": {"random": 1, "risk": 0, "cluster": 0},
    }
    packet = {
        "schema_version": AGGREGATE.PACKET_SCHEMA,
        "status": "materialized_no_model_invocation",
        "inputs": {
            "candidate_roster": binding(roster_path),
            "review_policy": binding(policy_path),
            "prompt": binding(prompt_path),
            "response_schema": binding(schema_path),
        },
        "selection": selection,
        "review_execution": {
            "model_environment_variable": "CODEX_REVIEW_MODEL",
            "model": "gpt-5.6-luna",
            "reasoning_effort": "low",
            "no_model_fallback": True,
            "model_invocation": "not_run",
            "prompt_sha256": prompt_sha,
            "response_schema_sha256": schema_sha,
        },
        "source_review_coverage": [coverage],
        "request_counts": {
            "primary": 1,
            "secondary": 1,
            "total": 2,
            "primary_by_stratum": {"random": 1, "risk": 0, "cluster": 0},
            "secondary_by_stratum": {"random": 1, "risk": 0, "cluster": 0},
        },
        "request_inventory": [
            {key: row[key] for key in ("review_id", "request_sha256", "sample_id", "reviewer_slot")}
            for row in (primary, secondary)
        ],
    }
    packet["requests"] = binding(requests_path)
    packet["manifest_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in packet.items() if key != "manifest_sha256"}
    )
    packet_path = tmp_path / "packet.json"
    write_json(packet_path, packet)

    primary_response = review_response(primary)
    secondary_response = review_response(secondary)
    responses_path = tmp_path / "responses.jsonl"
    write_jsonl(responses_path, [primary_response, secondary_response])
    final_manifest = REVIEW.build_adjudication_manifest([primary, secondary], [primary_response, secondary_response])
    assert final_manifest["pending_count"] == 0
    calibration_fields = (
        "review_id",
        "request_sha256",
        "reviewer_slot",
        "sample_id",
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
    calibration_pairs = [
        {
            "primary_request": {field: primary[field] for field in calibration_fields},
            "secondary_request": {field: secondary[field] for field in calibration_fields},
        }
    ]
    calibration = {
        "schema_version": AGGREGATE.CALIBRATION_RECEIPT_SCHEMA,
        "status": "passed",
        "input_scope": "compact_v3_request_jsonl_only_no_corpus_files",
        "corpus_files_read": False,
        "inputs": {
            "requests": binding(requests_path),
            "policy": binding(policy_path),
            "prompt": binding(prompt_path),
            "response_schema": binding(schema_path),
        },
        "model": {
            "environment_variable": "CODEX_REVIEW_MODEL",
            "required_model": "gpt-5.6-luna",
            "accepted_model": "gpt-5.6-luna",
            "no_fallback": True,
        },
        "assessment": {
            "status": "passed",
            "failure_count": 0,
            "criteria": {
                "all_primary_secondary_response_identities_valid": True,
                "all_represented_logical_routes_sampled": True,
                "low_confidence_or_uncertain_is_a_failure": True,
                "material_score_issue_or_recommendation_disagreement_is_a_failure": True,
                "prompt_or_schema_tuning_performed": False,
                "admission_decision_performed": False,
            },
            "logical_route_coverage": [
                {
                    "logical_source_route": "structured",
                    "pair_count": 1,
                    "source_ids": ["source-a"],
                    "distinct_source_count": 1,
                }
            ],
        },
        "selection": {
            "algorithm": "route_spanning_source_round_robin_request_hash_v1",
            "selected_pairs": calibration_pairs,
            "selected_pair_inventory_sha256": AGGREGATE.sha256_json(calibration_pairs),
        },
        "prompt_schema_frozen_for_full_review": True,
        "prompt_or_schema_tuning_performed": False,
        "admission_decision_performed": False,
        "primary_secondary_sessions_separated": True,
    }
    calibration["receipt_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in calibration.items() if key != "receipt_sha256"}
    )
    calibration_path = tmp_path / "calibration-receipt.json"
    write_json(calibration_path, calibration)
    calibration_binding = binding(calibration_path)
    adjudication_receipt = {
        "schema_version": AGGREGATE.ADJUDICATION_RECEIPT_SCHEMA,
        "status": "complete",
        "model": "gpt-5.6-luna",
        "initial_request_rows": 2,
        "adjudication_request_rows": 0,
        "response_rows": 2,
        "response_slot_counts": {"primary": 1, "secondary": 1},
        "responses": binding(responses_path),
        "final_adjudication_manifest": final_manifest,
        "primary_secondary_sessions_separated": True,
        "adjudication_sessions_separated": True,
        "passed_calibration_receipt": calibration_binding,
    }
    adjudication_receipt["receipt_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in adjudication_receipt.items() if key != "receipt_sha256"}
    )
    adjudication_path = tmp_path / "adjudication.json"
    write_json(adjudication_path, adjudication_receipt)
    response_receipt = {
        "schema_version": AGGREGATE.RESPONSE_RECEIPT_SCHEMA,
        "status": "complete",
        "inputs": {
            "requests": binding(requests_path),
            "policy": binding(policy_path),
            "prompt": binding(prompt_path),
            "response_schema": binding(schema_path),
        },
        "model": {
            "environment_variable": "CODEX_REVIEW_MODEL",
            "required_model": "gpt-5.6-luna",
            "accepted_model": "gpt-5.6-luna",
            "no_fallback": True,
        },
        "responses": binding(responses_path, rows=2) | {"slot_counts": {"primary": 1, "secondary": 1}},
        "primary_secondary_sessions_separated": True,
        "adjudication_sessions_separated": True,
        "adjudication_receipt": binding(adjudication_path),
        "passed_calibration_receipt": calibration_binding,
    }
    response_receipt["receipt_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in response_receipt.items() if key != "receipt_sha256"}
    )
    response_receipt_path = tmp_path / "response-receipt.json"
    write_json(response_receipt_path, response_receipt)

    # The Stage-35 closure, masked sample receipt, and diagnostic summary must
    # all bind the exact primary inventory selected in Stage 30.
    primary_inventory_sha = AGGREGATE.sha256_json(sorted([sample_id]))
    closure_path = tmp_path / "stage35-closure.json"
    closure = {
        "schema_version": AGGREGATE.STAGE35_CLOSURE_SCHEMA,
        "status": "passed",
        "run_id": "run-v3",
        "code_commit": "b" * 40,
        "inputs": {
            "review_requests": binding(requests_path),
            "review_packet_manifest": binding(packet_path),
            "external_responses": binding(responses_path),
            "external_response_receipt": binding(response_receipt_path),
            "external_adjudication_receipt": binding(adjudication_path),
            "external_calibration_receipt": calibration_binding,
        },
        "packet": {
            "manifest_sha256": packet["manifest_sha256"],
            "primary_sample_inventory_sha256": primary_inventory_sha,
        },
        "review_execution": {
            "model_environment_variable": "CODEX_REVIEW_MODEL",
            "required_model": "gpt-5.6-luna",
            "accepted_model": "gpt-5.6-luna",
            "no_model_fallback": True,
            "prompt_sha256": prompt_sha,
            "response_schema_sha256": schema_sha,
            "calibration_receipt_sha256": calibration["receipt_sha256"],
            "calibration_prompt_schema_frozen_for_full_review": True,
            "code_commit": "b" * 40,
        },
        "response_closure": {
            "pending_adjudication_count": 0,
            "final_adjudication_manifest": final_manifest,
            "response_execution_receipt_sha256": response_receipt["receipt_sha256"],
            "adjudication_execution_receipt_sha256": adjudication_receipt["receipt_sha256"],
        },
        "privacy": {"external_bundle_contains_raw_corpus": False},
        "admission_decision": "not_evaluated_in_stage35",
    }
    closure["closure_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in closure.items() if key != "closure_sha256"}
    )
    write_json(closure_path, closure)

    sample_receipt_path = tmp_path / "masked-sample-receipt.json"
    sample_receipt = {
        "schema_version": "agent1_v3_masked_review_sample_receipt_v1",
        "status": "passed",
        "primary_sample_count": 1,
        "primary_sample_inventory_sha256": primary_inventory_sha,
        "raw_corpus_included": False,
        "text_variant": "high_precision_identifier_masked_review_sample",
        "admission_decision": "not_evaluated_in_stage35",
        "inputs": {"quality_review_evidence_closure": binding(closure_path)},
    }
    sample_receipt["receipt_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in sample_receipt.items() if key != "receipt_sha256"}
    )
    write_json(sample_receipt_path, sample_receipt)
    review_sample_quality_summary_path = tmp_path / "review-sample-quality.json"
    review_sample_quality_summary = {
        "schema_version": AGGREGATE.REVIEW_SAMPLE_QUALITY_SUMMARY_SCHEMA,
        "status": "passed",
        "scan_mode": "exact_v3_masked_review_sample",
        "diagnostic_only": True,
        "admission_decision": "not_evaluated_in_stage35",
        "sample": {
            "primary_samples": 1,
            "primary_sample_inventory_sha256": primary_inventory_sha,
            "raw_corpus_included": False,
            "text_variant": "high_precision_identifier_masked_review_sample",
            "receipt": binding(sample_receipt_path),
        },
        "source_summaries": [
            {
                "repo_id": "source-a",
                "source_datasets": ["source-a"],
                "documents": 1,
                "document_rates": {"html_rate": 0.0},
                "distributions": {"original_characters": {"p50_approx": 20.0}},
                "template_concentration": {
                    "documents_with_template": 1,
                    "unique_templates": 1,
                    "top_1_fraction": 1.0,
                    "top_10_fraction": 1.0,
                },
            }
        ],
    }
    review_sample_quality_summary["summary_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in review_sample_quality_summary.items() if key != "summary_sha256"}
    )
    write_json(review_sample_quality_summary_path, review_sample_quality_summary)
    review_sample_quality_handoff_path = tmp_path / "review-sample-quality-handoff.json"
    review_sample_quality_handoff = {
        "schema_version": AGGREGATE.REVIEW_SAMPLE_QUALITY_HANDOFF_SCHEMA,
        "status": "passed",
        "summary": binding(review_sample_quality_summary_path),
        "sample_receipt": binding(sample_receipt_path),
        "diagnostic_only": True,
        "raw_corpus_included": False,
        "admission_decision": "not_evaluated_in_stage35",
    }
    review_sample_quality_handoff["handoff_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in review_sample_quality_handoff.items() if key != "handoff_sha256"}
    )
    write_json(review_sample_quality_handoff_path, review_sample_quality_handoff)

    quality_path = tmp_path / "quality.json"
    quality = {
        "selected_source_ids": ["source-a"],
        "excluded_source_ids": ["nanochat_base"],
        "global": {"source_datasets": ["source-a"]},
        "repositories": [
            {
                "repo_id": "glossAPI/source-a",
                "source_datasets": ["source-a"],
                "documents": 1,
                "document_rates": {"html_rate": 0.0},
                "distributions": {"original_characters": {"p50_approx": 20.0}},
                "template_concentration": {
                    "documents_with_template": 1,
                    "unique_templates": 1,
                    "top_1_fraction": 1.0,
                    "top_10_fraction": 1.0,
                },
            }
        ],
    }
    write_json(quality_path, quality)
    monkeypatch.setattr(AGGREGATE.quality_runtime, "validate_and_project_quality_summary", lambda value: {"scan_mode": "full_scan"})

    novelty_path = tmp_path / "novelty.json"
    novelty = {
        "schema_version": AGGREGATE.NOVELTY_SCHEMA,
        "near_duplicate_novelty_deferred_to_global_dedup": True,
        "sources": [
            {
                "source_dataset": "source-a",
                "source_ids": ["source-a"],
                "rows": 1,
                "identity_word_tokens": 5,
                "exact_unique_rows": 1,
                "exact_unique_word_tokens": 5,
                "novel_rows_after_lineage_resolution": 1,
                "novel_word_tokens_after_lineage_resolution": 5,
                "novel_token_fraction": 1.0,
                "document_action_counts": {},
            }
        ],
    }
    write_json(novelty_path, novelty)
    lineage_path = tmp_path / "lineage.json"
    lineage = {
        "schema_version": AGGREGATE.LINEAGE_SCHEMA,
        "blind_append_allowed": False,
        "source_novelty": {"sha256": binding(novelty_path)["sha256"]},
        "sources": [
            {
                "source_id": "source-a",
                "origin": "candidate",
                "rows": 1,
                "distinct_source_dataset_names": 1,
                "distinct_source_families": 1,
                "base_candidate_exact_clusters": 0,
                "base_candidate_work_clusters": 0,
                "double_add_hazard_reasons": ["new_family_still_requires_global_exact_near_dedup"],
            }
        ],
    }
    write_json(lineage_path, lineage)
    license_path = tmp_path / "license.json"
    license = {
        "schema_version": AGGREGATE.LICENSE_SCHEMA,
        "status": "technical_audit_complete",
        "sources": [
            {
                "source_id": "source-a",
                "repo_id": "glossAPI/source-a",
                "revision": source_revision,
                "declared_license": "cc-by-4.0",
                "registry_training_eligibility": "eligible_open",
                "local_training": {"eligible": True, "status": "allowed_for_pipeline", "conditions": ["attribution"]},
                "redistribution": {"eligible": True, "status": "allowed_for_pipeline", "conditions": ["attribution"]},
            }
        ],
    }
    write_json(license_path, license)
    return {
        "roster": roster_path,
        "packet": packet_path,
        "requests": requests_path,
        "responses": responses_path,
        "response_receipt": response_receipt_path,
        "adjudication_receipt": adjudication_path,
        "calibration_receipt": calibration_path,
        "stage35_closure": closure_path,
        "review_sample_quality_summary": review_sample_quality_summary_path,
        "review_sample_quality_handoff": review_sample_quality_handoff_path,
        "quality": quality_path,
        "lineage": lineage_path,
        "novelty": novelty_path,
        "license": license_path,
    }


def aggregate_from_fixture(fixture: dict[str, object]) -> dict[str, object]:
    return AGGREGATE.build_aggregate(
        run_id="run-v3",
        roster_path=fixture["roster"],
        packet_path=fixture["packet"],
        requests_path=fixture["requests"],
        responses_path=fixture["responses"],
        response_receipt_path=fixture["response_receipt"],
        adjudication_receipt_path=fixture["adjudication_receipt"],
        stage35_closure_path=fixture["stage35_closure"],
        review_sample_quality_summary_path=fixture["review_sample_quality_summary"],
        review_sample_quality_handoff_path=fixture["review_sample_quality_handoff"],
        quality_summary_path=fixture["quality"],
        lineage_summary_path=fixture["lineage"],
        novelty_summary_path=fixture["novelty"],
        license_adjudication_path=fixture["license"],
    )


def build_aggregate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, object], dict[str, object]]:
    fixture = make_fixture(tmp_path, monkeypatch)
    result = aggregate_from_fixture(fixture)
    return result, fixture


def test_aggregate_requires_exact_stage30_stage35_closure_and_derives_source_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = build_aggregate(tmp_path, monkeypatch)

    assert result["status"] == "passed_review_evidence_no_admission_decision"
    assert result["review_closure"]["pending_count"] == 0
    assert result["source_ids"] == ["source-a"]
    source = result["sources"][0]
    assert source["review"]["resolved_documents"] == 1
    assert source["review"]["recommendation_counts"]["include"] == 1
    assert source["review"]["issue_document_counts"] == {"template_replay": 1}
    assert source["cluster_and_template"]["quality_template_concentration"]["top_1_fraction"] == 1.0
    AGGREGATE.validate_aggregate(result, roster=json.loads(Path(_["roster"]).read_text(encoding="utf-8")))


def test_aggregate_rejects_self_hashed_source_evidence_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recomputed aggregate hash must not make invented reviews admissible."""

    result, fixture = build_aggregate(tmp_path, monkeypatch)
    forged = copy.deepcopy(result)
    source = forged["sources"][0]
    source["review"]["recommendation_counts"]["include"] = 0
    source["review"]["recommendation_counts"]["exclude"] = 1
    source["review"]["recommendation_rates"]["include"] = 0.0
    source["review"]["recommendation_rates"]["exclude"] = 1.0
    source["source_evidence_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in source.items() if key != "source_evidence_sha256"}
    )
    forged["aggregate_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in forged.items() if key != "aggregate_sha256"}
    )

    with pytest.raises(ValueError, match="deterministic recomputation"):
        AGGREGATE.validate_aggregate(
            forged,
            roster=json.loads(Path(fixture["roster"]).read_text(encoding="utf-8")),
        )


def test_aggregate_rejects_response_receipt_not_bound_to_exact_stage30_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    receipt = json.loads(Path(fixture["response_receipt"]).read_text(encoding="utf-8"))
    receipt["inputs"]["requests"]["sha256"] = "0" * 64
    receipt["receipt_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    write_json(Path(fixture["response_receipt"]), receipt)

    with pytest.raises(ValueError, match="Stage 35 requests do not bind|bound file bytes"):
        AGGREGATE.build_aggregate(
            run_id="run-v3",
            roster_path=fixture["roster"],
            packet_path=fixture["packet"],
            requests_path=fixture["requests"],
            responses_path=fixture["responses"],
            response_receipt_path=fixture["response_receipt"],
            adjudication_receipt_path=fixture["adjudication_receipt"],
            stage35_closure_path=fixture["stage35_closure"],
            review_sample_quality_summary_path=fixture["review_sample_quality_summary"],
            review_sample_quality_handoff_path=fixture["review_sample_quality_handoff"],
            quality_summary_path=fixture["quality"],
            lineage_summary_path=fixture["lineage"],
            novelty_summary_path=fixture["novelty"],
            license_adjudication_path=fixture["license"],
        )


def test_aggregate_rejects_unclosed_stage35_receipt_even_when_raw_responses_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    closure_path = Path(fixture["stage35_closure"])
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    closure["response_closure"]["pending_adjudication_count"] = 1
    closure["closure_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in closure.items() if key != "closure_sha256"}
    )
    write_json(closure_path, closure)

    with pytest.raises(ValueError, match="review-response/adjudication closure drift"):
        AGGREGATE.build_aggregate(
            run_id="run-v3",
            roster_path=fixture["roster"],
            packet_path=fixture["packet"],
            requests_path=fixture["requests"],
            responses_path=fixture["responses"],
            response_receipt_path=fixture["response_receipt"],
            adjudication_receipt_path=fixture["adjudication_receipt"],
            stage35_closure_path=fixture["stage35_closure"],
            review_sample_quality_summary_path=fixture["review_sample_quality_summary"],
            review_sample_quality_handoff_path=fixture["review_sample_quality_handoff"],
            quality_summary_path=fixture["quality"],
            lineage_summary_path=fixture["lineage"],
            novelty_summary_path=fixture["novelty"],
            license_adjudication_path=fixture["license"],
        )


def test_aggregate_requires_route_spanning_calibration_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    closure_path = Path(fixture["stage35_closure"])
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    del closure["inputs"]["external_calibration_receipt"]
    del closure["review_execution"]["calibration_receipt_sha256"]
    closure["closure_sha256"] = AGGREGATE.sha256_json(
        {key: value for key, value in closure.items() if key != "closure_sha256"}
    )
    write_json(closure_path, closure)

    with pytest.raises(ValueError, match="calibration"):
        aggregate_from_fixture(fixture)


def test_admission_freezes_only_complete_hash_bound_proposal_then_exact_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_aggregate, fixture = build_aggregate(tmp_path, monkeypatch)
    aggregate_path = tmp_path / "aggregate.json"
    write_json(aggregate_path, review_aggregate)
    source = review_aggregate["sources"][0]
    proposal_path = tmp_path / "proposal.json"
    proposal = {
        "schema_version": ADMISSION.PROPOSAL_SCHEMA,
        "review_aggregate": binding(aggregate_path),
        "sources": [
            {
                "source_id": "source-a",
                "source_dataset": "source-a",
                "source_revision": "a" * 40,
                "decision": "include",
                "source_evidence_sha256": source["source_evidence_sha256"],
                "review_support": "supported",
                "manual_override_reason": None,
                "rationale": "The closed source review, full scan, lineage, and licence evidence support inclusion.",
                "required_cleaning": [],
                "expected_token_loss": {
                    "kind": "not_applicable",
                    "estimated_fraction": None,
                    "basis": "No pre-admission cleaning action is proposed for this source.",
                },
            }
        ],
    }
    write_json(proposal_path, proposal)
    packet_path = tmp_path / "admission-packet.json"
    ADMISSION.cmd_build_packet(
        type(
            "Args",
            (),
            {
                "output": packet_path,
                "roster": fixture["roster"],
                "review_aggregate": aggregate_path,
                "run_id": "run-v3",
                "proposed_decisions": proposal_path,
            },
        )()
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["status"] == "pending_user_confirmation"
    assert packet["destructive_progression"]["deduplication_permitted"] is False

    confirmation_path = tmp_path / "confirmation.json"
    ADMISSION.cmd_confirm(
        type(
            "Args",
            (),
            {
                "packet": packet_path,
                "packet_sha256": binding(packet_path)["sha256"],
                "confirm_user_reviewed_packet_sha256": binding(packet_path)["sha256"],
                "confirmation_note": "User inspected the packet and explicitly confirmed this exact hash.",
                "roster": fixture["roster"],
                "output": confirmation_path,
            },
        )()
    )
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    ADMISSION.validate_confirmation(confirmation, roster=json.loads(Path(fixture["roster"]).read_text(encoding="utf-8")))

    # Even a recomputed confirmation hash cannot alter the packet's frozen
    # proposal after user inspection.
    confirmation["sources"][0]["decision"] = "exclude"
    confirmation["confirmation_sha256"] = ADMISSION.sha256_json(
        {key: value for key, value in confirmation.items() if key != "confirmation_sha256"}
    )
    with pytest.raises(ValueError, match="differ from the exact confirmed proposal"):
        ADMISSION.validate_confirmation(confirmation, roster=json.loads(Path(fixture["roster"]).read_text(encoding="utf-8")))


def test_admission_rejects_incomplete_or_license_forbidden_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_aggregate, fixture = build_aggregate(tmp_path, monkeypatch)
    aggregate_path = tmp_path / "aggregate.json"
    write_json(aggregate_path, review_aggregate)
    incomplete = {
        "schema_version": ADMISSION.PROPOSAL_SCHEMA,
        "review_aggregate": binding(aggregate_path),
        "sources": [],
    }
    with pytest.raises(ValueError, match="non-empty list|must decide every roster source"):
        ADMISSION.validate_proposal(
            incomplete,
            aggregate_value=review_aggregate,
            aggregate_path=aggregate_path,
            roster=json.loads(Path(fixture["roster"]).read_text(encoding="utf-8")),
        )

    license_value = json.loads(Path(fixture["license"]).read_text(encoding="utf-8"))
    license_value["sources"][0]["local_training"]["eligible"] = False
    write_json(Path(fixture["license"]), license_value)
    review_aggregate = aggregate_from_fixture(fixture)
    write_json(aggregate_path, review_aggregate)
    source = review_aggregate["sources"][0]
    forbidden = {
        "schema_version": ADMISSION.PROPOSAL_SCHEMA,
        "review_aggregate": binding(aggregate_path),
        "sources": [
            {
                "source_id": "source-a",
                "source_dataset": "source-a",
                "source_revision": "a" * 40,
                "decision": "include",
                "source_evidence_sha256": source["source_evidence_sha256"],
                "review_support": "supported",
                "manual_override_reason": None,
                "rationale": "This would otherwise have evidence, but the licence must remain a hard gate.",
                "required_cleaning": [],
                "expected_token_loss": {
                    "kind": "not_applicable",
                    "estimated_fraction": None,
                    "basis": "No cleaning action is proposed before a licence-authorized admission.",
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="licence excludes local training"):
        ADMISSION.validate_proposal(
            forbidden,
            aggregate_value=review_aggregate,
            aggregate_path=aggregate_path,
            roster=json.loads(Path(fixture["roster"]).read_text(encoding="utf-8")),
        )

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "scripts" / "agent1_v3_review_evidence.py"


def load_module():
    spec = importlib.util.spec_from_file_location("agent1_v3_review_evidence_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVIDENCE = load_module()
import agent1_v3_review as REVIEW  # noqa: E402


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(EVIDENCE.canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def response_for(request: dict[str, object]) -> dict[str, object]:
    identity = {
        field: request[field]
        for field in (
            "review_id",
            "sample_id",
            "reviewer_slot",
            "source_id",
            "source_dataset",
            "source_revision",
            "source_route",
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
    }
    return {
        "schema_version": REVIEW.RESPONSE_SCHEMA,
        **identity,
        "cleanliness_score": 4,
        "quality_score": 4,
        "diversity_contribution_score": 3,
        "issues": [],
        "recommendation": "include_after_cleaning",
        "confidence_score": 5,
        "evidence": "Σύντομη τεκμηρίωση από το μασκαρισμένο δείγμα.",
    }


def receipt(value: dict[str, object]) -> dict[str, object]:
    value["receipt_sha256"] = EVIDENCE.receipt_digest(value)
    return value


def fixture(tmp_path: Path) -> dict[str, object]:
    code_commit = "a" * 40
    prompt = tmp_path / "prompt.md"
    prompt.write_text("review prompt", encoding="utf-8")
    schema = tmp_path / "schema.json"
    write_json(
        schema,
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_version",
                "observed_extraction_route",
                "observed_extraction_route_basis",
                "observed_extraction_route_evidence",
                "observed_extraction_route_priority",
            ],
            "properties": {
                "schema_version": {"const": REVIEW.RESPONSE_SCHEMA},
                "observed_extraction_route": {"enum": ["pdf_ocr"]},
                "observed_extraction_route_basis": {
                    "enum": ["row_representation_metadata"]
                },
                "observed_extraction_route_evidence": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 256,
                },
                "observed_extraction_route_priority": {
                    "enum": ["logical_primary"]
                },
            },
        },
    )
    policy = tmp_path / "policy.json"
    write_json(
        policy,
        {
            "schema_version": REVIEW.POLICY_SCHEMA,
            "review": {
                "required_model": "gpt-5.6-luna",
                "model_environment_variable": "CODEX_REVIEW_MODEL",
                "no_model_fallback": True,
                "reasoning_effort": "low",
            },
        },
    )
    prompt_binding = EVIDENCE.file_binding(prompt)
    schema_binding = EVIDENCE.file_binding(schema)
    primary_copy = "Μασκαρισμένο κείμενο [EMAIL_0001]."
    secondary_copy = primary_copy
    primary = REVIEW.make_review_request(
        {
            "source_id": "source-a",
            "source_dataset": "dataset-a",
            "source_revision": "rev-a",
            "stable_uid": digest("sample-a"),
            "source_route": "pdf_ocr",
            "observed_extraction_route": "pdf_ocr",
            "observed_extraction_route_basis": "row_representation_metadata",
            "observed_extraction_route_evidence": "raw_metadata:mime_type=application_pdf",
            "observed_extraction_route_priority": "logical_primary",
            "sampling_stratum": "random",
        },
        reviewer_slot="primary",
        original_text_sha256=digest("original-a"),
        review_copy_sha256=digest(primary_copy),
        prompt_sha256=str(prompt_binding["sha256"]),
        response_schema_sha256=str(schema_binding["sha256"]),
        model="gpt-5.6-luna",
        code_commit=code_commit,
        review_copy=primary_copy,
        comparison_bundle=[],
    )
    secondary = REVIEW.make_review_request(
        {
            "source_id": "source-a",
            "source_dataset": "dataset-a",
            "source_revision": "rev-a",
            "stable_uid": digest("sample-a"),
            "source_route": "pdf_ocr",
            "observed_extraction_route": "pdf_ocr",
            "observed_extraction_route_basis": "row_representation_metadata",
            "observed_extraction_route_evidence": "raw_metadata:mime_type=application_pdf",
            "observed_extraction_route_priority": "logical_primary",
            "sampling_stratum": "random",
        },
        reviewer_slot="secondary",
        original_text_sha256=digest("original-a"),
        review_copy_sha256=digest(secondary_copy),
        prompt_sha256=str(prompt_binding["sha256"]),
        response_schema_sha256=str(schema_binding["sha256"]),
        model="gpt-5.6-luna",
        code_commit=code_commit,
        review_copy=secondary_copy,
        comparison_bundle=[],
    )
    requests = tmp_path / "requests.jsonl"
    write_jsonl(requests, [primary, secondary])
    requests_binding = EVIDENCE.file_binding(requests)
    packet = {
        "schema_version": EVIDENCE.PACKET_MANIFEST_SCHEMA,
        "status": "materialized_no_model_invocation",
        "inputs": {
            "review_policy": EVIDENCE.file_binding(policy),
            "prompt": prompt_binding,
            "response_schema": schema_binding,
        },
        "requests": requests_binding,
        "review_execution": {
            "model_environment_variable": "CODEX_REVIEW_MODEL",
            "model": "gpt-5.6-luna",
            "no_model_fallback": True,
            "model_invocation": "not_run",
            "code_commit": code_commit,
            "prompt_sha256": prompt_binding["sha256"],
            "response_schema_sha256": schema_binding["sha256"],
        },
        "request_inventory": [
            {
                "review_id": request["review_id"],
                "request_sha256": request["request_sha256"],
                "sample_id": request["sample_id"],
                "reviewer_slot": request["reviewer_slot"],
            }
            for request in (primary, secondary)
        ],
        "request_counts": {"primary": 1, "secondary": 1, "total": 2},
        "privacy": {
            "raw_canonical_text_in_manifest": False,
            "raw_source_document_identifier_in_manifest": False,
            "review_copy_masking": "high_precision_direct_identifiers_position_preserving",
        },
        "review_copy_attestations": [
            {
                "stable_uid": primary["sample_id"],
                "original_text_sha256": primary["original_text_sha256"],
                "review_copy_sha256": primary["review_copy_sha256"],
                "positions_preserved": True,
            }
        ],
    }
    packet["manifest_sha256"] = EVIDENCE.manifest_digest(packet)
    packet_path = tmp_path / "packet.json"
    write_json(packet_path, packet)

    responses = [response_for(primary), response_for(secondary)]
    responses_path = tmp_path / "responses.jsonl"
    write_jsonl(responses_path, responses)
    final = REVIEW.build_adjudication_manifest([primary, secondary], responses)
    assert final["status"] == "complete"
    response_binding = EVIDENCE.file_binding(responses_path)

    def calibration_request_summary(request: dict[str, object]) -> dict[str, object]:
        return {
            field: request[field]
            for field in (
                "review_id",
                "request_sha256",
                "reviewer_slot",
                "sample_id",
                "source_id",
                "source_dataset",
                "source_revision",
                "source_route",
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
        }

    selected_pairs = [
        {
            "primary_request": calibration_request_summary(primary),
            "secondary_request": calibration_request_summary(secondary),
        }
    ]
    calibration = receipt(
        {
            "schema_version": EVIDENCE.CALIBRATION_RECEIPT_SCHEMA,
            "status": "passed",
            "input_scope": "compact_v3_request_jsonl_only_no_corpus_files",
            "corpus_files_read": False,
            "inputs": {
                "requests": requests_binding,
                "policy": EVIDENCE.file_binding(policy),
                "prompt": prompt_binding,
                "response_schema": schema_binding,
            },
            "model": {
                "environment_variable": "CODEX_REVIEW_MODEL",
                "required_model": "gpt-5.6-luna",
                "accepted_model": "gpt-5.6-luna",
                "no_fallback": True,
            },
            "execution_settings": {"settings_sha256": digest("calibration-settings")},
            "selection": {
                "algorithm": "route_spanning_source_round_robin_request_hash_v1",
                "selection_namespace": "agent1_v3_route_spanning_prompt_calibration_v1",
                "calibration_per_route": 1,
                "minimum_required_routes": 1,
                "selected_pairs": selected_pairs,
                "selected_pair_inventory_sha256": EVIDENCE.sha256_json(selected_pairs),
            },
            "assessment": {
                "status": "passed",
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
                        "logical_source_route": "pdf_ocr",
                        "pair_count": 1,
                        "source_ids": ["source-a"],
                        "distinct_source_count": 1,
                    }
                ],
                "case_count": 1,
                "failure_count": 0,
                "cases": [],
            },
            "execution_batches": [],
            "usage_for_calibration": {},
            "prompt_schema_frozen_for_full_review": True,
            "prompt_or_schema_tuning_performed": False,
            "admission_decision_performed": False,
            "primary_secondary_sessions_separated": True,
        }
    )
    calibration_path = tmp_path / "calibration-receipt.json"
    write_json(calibration_path, calibration)
    calibration_binding = EVIDENCE.file_binding(calibration_path)
    adjudication = receipt(
        {
            "schema_version": EVIDENCE.ADJUDICATION_RECEIPT_SCHEMA,
            "status": "complete",
            "model": "gpt-5.6-luna",
            "initial_request_rows": 2,
            "adjudication_request_rows": 0,
            "response_rows": 2,
            "response_slot_counts": {"primary": 1, "secondary": 1},
            "responses": response_binding,
            "pending_before_execution": {
                "case_count": 0,
                "pending_count": 0,
                "manifest_sha256": final["manifest_sha256"],
            },
            "final_adjudication_manifest": final,
            "execution_batches": [],
            "no_adjudication_noop": True,
            "primary_secondary_sessions_separated": True,
            "adjudication_sessions_separated": True,
            "passed_calibration_receipt": calibration_binding,
        }
    )
    adjudication_path = tmp_path / "adjudication.json"
    write_json(adjudication_path, adjudication)
    response_receipt = receipt(
        {
            "schema_version": EVIDENCE.RESPONSE_RECEIPT_SCHEMA,
            "status": "complete",
            "implementation_version": "agent1_v3_codex_review_runner_v1",
            "input_scope": "compact_v3_request_jsonl_only_no_corpus_files",
            "corpus_files_read": False,
            "inputs": {
                "requests": requests_binding,
                "policy": EVIDENCE.file_binding(policy),
                "prompt": prompt_binding,
                "response_schema": {
                    **schema_binding,
                    "normalizer": "agent1_v3_openai_schema_compat_v1",
                    "normalized_execution_schema_sha256": digest("normalized-schema"),
                },
            },
            "model": {
                "environment_variable": "CODEX_REVIEW_MODEL",
                "required_model": "gpt-5.6-luna",
                "accepted_model": "gpt-5.6-luna",
                "no_fallback": True,
            },
            "execution_settings": {
                "model": "gpt-5.6-luna",
                "sandbox": "read-only",
                "ephemeral": True,
                "schema_normalizer": "agent1_v3_openai_schema_compat_v1",
                "settings_sha256": digest("settings"),
            },
            "preflight": {
                "synthetic_fixture": True,
                "request_sha256": digest("preflight-request"),
                "review_id": digest("preflight-id"),
                "response_sha256": digest("preflight-response"),
                "execution_schema_sha256": digest("preflight-schema"),
                "accepted_model": "gpt-5.6-luna",
            },
            "requests": {
                "initial_rows": 2,
                "initial_slot_counts": {"primary": 1, "secondary": 1},
                "adjudication_rows": 0,
            },
            "responses": {**response_binding, "rows": 2, "slot_counts": {"primary": 1, "secondary": 1}},
            "adjudication_receipt": EVIDENCE.file_binding(adjudication_path),
            "passed_calibration_receipt": calibration_binding,
        }
    )
    response_receipt_path = tmp_path / "response-receipt.json"
    write_json(response_receipt_path, response_receipt)
    return {
        "run_id": "agent1-full-corpus-v3-20260713T120000Z-abcdef1",
        "code_commit": code_commit,
        "requests": requests,
        "requests_path": requests,
        "packet_path": packet_path,
        "policy": policy,
        "prompt": prompt,
        "schema": schema,
        "responses": responses_path,
        "response_receipt": response_receipt_path,
        "adjudication_receipt": adjudication_path,
        "calibration_receipt": calibration_path,
    }


def test_stage35_packages_imports_validates_and_materializes_masked_sample(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    bundle = tmp_path / "external-bundle"
    EVIDENCE.package_external(
        run_id=str(data["run_id"]),
        requests_path=data["requests_path"],  # type: ignore[arg-type]
        responses_path=data["responses"],  # type: ignore[arg-type]
        response_receipt_path=data["response_receipt"],  # type: ignore[arg-type]
        adjudication_receipt_path=data["adjudication_receipt"],  # type: ignore[arg-type]
        calibration_receipt_path=data["calibration_receipt"],  # type: ignore[arg-type]
        output_dir=bundle,
    )
    import_destination = tmp_path / "imported"
    import_receipt = tmp_path / "import-receipt.json"
    EVIDENCE.import_external_bundle(
        external_dir=bundle, destination=import_destination, receipt_path=import_receipt
    )
    closure_path = tmp_path / "closure.json"
    closure = EVIDENCE.validate_closure(
        run_id=str(data["run_id"]),
        requests_path=data["requests_path"],  # type: ignore[arg-type]
        packet_manifest_path=data["packet_path"],  # type: ignore[arg-type]
        external_dir=import_destination,
        policy_path=data["policy"],  # type: ignore[arg-type]
        prompt_path=data["prompt"],  # type: ignore[arg-type]
        schema_path=data["schema"],  # type: ignore[arg-type]
        code_commit=str(data["code_commit"]),
        output=closure_path,
    )
    assert closure["response_closure"]["pending_adjudication_count"] == 0
    assert "review_copy" not in closure_path.read_text(encoding="utf-8")
    sample = tmp_path / "masked-sample.jsonl"
    sample_receipt = tmp_path / "masked-sample-receipt.json"
    result = EVIDENCE.materialize_masked_sample(
        requests_path=data["requests_path"],  # type: ignore[arg-type]
        packet_manifest_path=data["packet_path"],  # type: ignore[arg-type]
        closure_path=closure_path,
        output=sample,
        receipt=sample_receipt,
    )
    assert result["raw_corpus_included"] is False
    rows = EVIDENCE.read_jsonl(sample, label="sample")
    assert len(rows) == 1
    assert rows[0]["text_variant"] == "high_precision_identifier_masked_review_sample"
    assert rows[0]["review_copy"] == "Μασκαρισμένο κείμενο [EMAIL_0001]."


def test_stage35_rejects_tampered_external_response_after_package(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    bundle = tmp_path / "external-bundle"
    EVIDENCE.package_external(
        run_id=str(data["run_id"]),
        requests_path=data["requests_path"],  # type: ignore[arg-type]
        responses_path=data["responses"],  # type: ignore[arg-type]
        response_receipt_path=data["response_receipt"],  # type: ignore[arg-type]
        adjudication_receipt_path=data["adjudication_receipt"],  # type: ignore[arg-type]
        calibration_receipt_path=data["calibration_receipt"],  # type: ignore[arg-type]
        output_dir=bundle,
    )
    (bundle / EVIDENCE.RESPONSES_NAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte/hash binding drift"):
        EVIDENCE.validate_bundle(bundle)

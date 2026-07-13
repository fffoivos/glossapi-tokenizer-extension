from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.codex56_audit import REQUEST_SCHEMA, RESPONSE_SCHEMA  # noqa: E402
from sequence_models.contract import canonical_json_sha256  # noqa: E402
from sequence_models.run_codex56_audit import (  # noqa: E402
    make_batches,
    validate_batch_payload,
    validate_request_manifest,
)


def test_codex_output_schema_const_and_enum_have_explicit_types() -> None:
    schema_path = Path(__file__).parents[1] / "codex56_audit_batch.schema.json"
    schema = json.loads(schema_path.read_text())
    properties = schema["properties"]["responses"]["items"]["properties"]
    assert properties["schema_version"]["type"] == "string"
    assert properties["label"]["type"] == "string"
    assert "uniqueItems" not in properties["evidence_abs_indices"]


def _request(index: int) -> dict:
    value = f"{index:064x}"
    request = {
        "schema_version": REQUEST_SCHEMA,
        "request_id": value,
        "prompt_version": "p1",
        "source": "fixture",
        "opaque_document_id": f"{index + 200:064x}",
        "target_abs_idx": index,
        "context_start_abs_idx": index,
        "context_end_abs_idx": index,
        "lines": [{"line_id": f"l{index}", "abs_idx": index, "text": "κείμενο"}],
    }
    request["request_sha256"] = canonical_json_sha256(request)
    return request


def _response(request: dict, model: str) -> dict:
    return {
        "schema_version": RESPONSE_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "reviewer_model": model,
        "label": "OTHER",
        "start_abs_idx": None,
        "end_abs_idx": None,
        "should_remove": False,
        "confidence": 0.8,
        "structural_cues": ["prose"],
        "evidence_abs_indices": [request["target_abs_idx"]],
    }


def test_batches_are_stable_bounded_and_model_bound() -> None:
    requests = [_request(index) for index in reversed(range(25))]
    batches = make_batches(
        requests,
        model="gpt-5.6-luna",
        prompt_sha256="a" * 64,
        output_schema_sha256="b" * 64,
    )
    assert [len(batch["requests"]) for batch in batches] == [12, 12, 1]
    assert batches == make_batches(
        requests,
        model="gpt-5.6-luna",
        prompt_sha256="a" * 64,
        output_schema_sha256="b" * 64,
    )
    assert batches != make_batches(
        requests,
        model="different",
        prompt_sha256="a" * 64,
        output_schema_sha256="b" * 64,
    )


def test_batch_validation_rejects_missing_and_wrong_model() -> None:
    requests = [_request(index) for index in range(2)]
    batch = make_batches(
        requests,
        model="gpt-5.6-luna",
        prompt_sha256="a" * 64,
        output_schema_sha256="b" * 64,
    )[0]
    payload = {
        "responses": [_response(request, "gpt-5.6-luna") for request in requests]
    }
    assert len(validate_batch_payload(batch, payload, model="gpt-5.6-luna")) == 2
    with pytest.raises(ValueError, match="omit"):
        validate_batch_payload(
            batch, {"responses": payload["responses"][:1]}, model="gpt-5.6-luna"
        )
    bad = [dict(row) for row in payload["responses"]]
    bad[0]["reviewer_model"] = "fallback"
    with pytest.raises(ValueError, match="model mismatch"):
        validate_batch_payload(batch, {"responses": bad}, model="gpt-5.6-luna")


def test_batches_recompute_request_content_hash() -> None:
    request = _request(1)
    request["lines"][0]["text"] = "tampered"

    with pytest.raises(ValueError, match="content hash mismatch"):
        make_batches(
            [request],
            model="gpt-5.6-luna",
            prompt_sha256="a" * 64,
            output_schema_sha256="b" * 64,
        )


def test_request_manifest_binds_the_complete_request_set() -> None:
    requests = [_request(1), _request(2)]
    manifest = {
        "schema_version": "academic-structure-codex56-audit-manifest-v1",
        "request_count": 2,
        "request_set_sha256": canonical_json_sha256(requests),
    }
    validate_request_manifest(requests, manifest)
    manifest["request_set_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="set hash mismatch"):
        validate_request_manifest(requests, manifest)

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
SCHEMA_PATH = HERE / "schemas" / "agent1_v3_review_response.schema.json"
POLICY_PATH = HERE / "configs" / "agent1_v3_policy.json"


def load_runner():
    path = SCRIPTS / "run_agent1_v3_codex_reviews.py"
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("agent1_v3_codex_runner_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


RUN = load_runner()
REVIEW = RUN.review


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "prompt.md"
    path.write_text("Review the compact request conservatively.\n", encoding="utf-8")
    return path


def request(
    slot: str = "primary",
    *,
    prompt_sha256: str,
    response_schema_sha256: str,
    source: str = "source-a",
    source_route: str = "pdf_ocr",
    observed_extraction_route: str | None = None,
    observed_extraction_route_basis: str = "explicit_row_route",
    observed_extraction_route_evidence: str = "raw_field:representation_kind",
    text: str = "Καθαρό σύντομο ελληνικό κείμενο.",
) -> dict:
    # A secondary review must bind the same sampled document as its primary;
    # only reviewer_slot changes the immutable review identity.
    stable_uid = digest(f"{source}:{text}")
    observed_route = observed_extraction_route or source_route
    return REVIEW.make_review_request(
        {
            "source_id": source,
            "source_dataset": source,
            "source_revision": "a" * 40,
            "stable_uid": stable_uid,
            "source_route": source_route,
            "observed_extraction_route": observed_route,
            "observed_extraction_route_basis": observed_extraction_route_basis,
            "observed_extraction_route_evidence": observed_extraction_route_evidence,
            "observed_extraction_route_priority": (
                "logical_primary"
                if observed_route == source_route
                else "secondary_exception_only"
            ),
            "sampling_stratum": "random",
        },
        reviewer_slot=slot,
        original_text_sha256=digest("original:" + stable_uid),
        review_copy_sha256=digest(text),
        prompt_sha256=prompt_sha256,
        response_schema_sha256=response_schema_sha256,
        model="gpt-5.6-luna",
        code_commit="b" * 40,
        review_copy=text,
        comparison_bundle=[],
    )


def response_for(request_row: dict, **overrides: object) -> dict:
    response = {
        "schema_version": REVIEW.RESPONSE_SCHEMA,
        **{
            field: request_row[field]
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
        },
        "cleanliness_score": 5,
        "quality_score": 4,
        "diversity_contribution_score": 3,
        "issues": [],
        "recommendation": "include",
        "confidence_score": 5,
        "evidence": "Συνεκτικό ελληνικό δείγμα χωρίς ορατό πρόβλημα εξαγωγής.",
    }
    response.update(overrides)
    return response


def test_schema_normalization_adds_types_and_keeps_defs_resolvable(tmp_path: Path) -> None:
    raw, binding = RUN.load_response_schema(SCHEMA_PATH)
    normalized = RUN.openai_schema_compat(raw)
    assert normalized["properties"]["schema_version"]["type"] == "string"
    assert normalized["properties"]["reviewer_slot"]["type"] == "string"
    for field in RUN.OBSERVED_EXTRACTION_ROUTE_FIELDS:
        assert field in raw["required"]
        assert field in raw["properties"]
    assert binding["sha256"] == RUN.sha256_file(SCHEMA_PATH)
    assert binding["normalized_execution_schema_sha256"]

    batch = RUN.make_batch_schema(raw, 2)
    assert batch["properties"]["responses"]["minItems"] == 2
    assert batch["$defs"]["response"]["properties"]["issues"]["items"]["$ref"] == "#/$defs/issue"
    assert batch["$defs"]["issue"]["properties"]["code"]["type"] == "string"

    missing_observed_field = schema()
    missing_observed_field["required"].remove("observed_extraction_route_evidence")
    missing_observed_field["properties"].pop("observed_extraction_route_evidence")
    invalid_schema = tmp_path / "missing-observed-route.schema.json"
    invalid_schema.write_text(json.dumps(missing_observed_field), encoding="utf-8")
    with pytest.raises(ValueError, match="expected strict"):
        RUN.load_response_schema(invalid_schema)


def test_model_must_be_explicit_policy_value_without_fallback() -> None:
    policy, _ = RUN.load_policy(POLICY_PATH)
    with pytest.raises(ValueError, match="CODEX_REVIEW_MODEL is required"):
        RUN.resolve_review_model(policy, {})
    with pytest.raises(ValueError, match="must equal policy required_model exactly"):
        RUN.resolve_review_model(policy, {"CODEX_REVIEW_MODEL": "another-model"})
    assert RUN.resolve_review_model(policy, {"CODEX_REVIEW_MODEL": "gpt-5.6-luna"}) == "gpt-5.6-luna"


def test_compact_observed_route_provenance_is_bound_and_secondary_only(
    tmp_path: Path,
) -> None:
    prompt = prompt_file(tmp_path)
    row = request(
        prompt_sha256=RUN.sha256_file(prompt),
        response_schema_sha256=RUN.sha256_file(SCHEMA_PATH),
        source_route="pdf_ocr",
        observed_extraction_route="html_web",
        observed_extraction_route_basis="row_representation_metadata",
        observed_extraction_route_evidence="raw_metadata:mime_type=text_html",
    )

    RUN.validate_execution_request(
        row,
        model="gpt-5.6-luna",
        prompt_sha256=RUN.sha256_file(prompt),
        response_schema_sha256=RUN.sha256_file(SCHEMA_PATH),
        initial_only=True,
    )
    assert row["source_route"] == "pdf_ocr"
    assert row["observed_extraction_route"] == "html_web"
    assert row["observed_extraction_route_priority"] == "secondary_exception_only"
    assert all(field in RUN.REQUEST_IDENTITY_FIELDS for field in RUN.OBSERVED_EXTRACTION_ROUTE_FIELDS)

    rendered = RUN.compose_prompt("Committed prompt.", [row])
    assert "secondary_exception_only" in rendered
    assert "raw_metadata:mime_type=text_html" in rendered

    calibration_secondary = RUN._calibration_secondary_request(row)
    for field in RUN.OBSERVED_EXTRACTION_ROUTE_FIELDS:
        assert calibration_secondary[field] == row[field]

    unbounded_or_textual = dict(row)
    unbounded_or_textual["observed_extraction_route_evidence"] = "raw metadata text is forbidden"
    unbounded_or_textual["request_sha256"] = REVIEW._request_hash(unbounded_or_textual)
    with pytest.raises(ValueError, match="bounded text-free audit code"):
        RUN.validate_execution_request(
            unbounded_or_textual,
            model="gpt-5.6-luna",
            prompt_sha256=RUN.sha256_file(prompt),
            response_schema_sha256=RUN.sha256_file(SCHEMA_PATH),
            initial_only=True,
        )

    priority_drift = dict(row)
    priority_drift["observed_extraction_route_priority"] = "logical_primary"
    priority_drift["request_sha256"] = REVIEW._request_hash(priority_drift)
    with pytest.raises(ValueError, match="preserve source_route as logical primary"):
        RUN.validate_execution_request(
            priority_drift,
            model="gpt-5.6-luna",
            prompt_sha256=RUN.sha256_file(prompt),
            response_schema_sha256=RUN.sha256_file(SCHEMA_PATH),
            initial_only=True,
        )


def test_batch_plan_isolates_slots_and_cache_never_stores_review_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = prompt_file(tmp_path)
    prompt_sha256 = RUN.sha256_file(prompt)
    schema_sha256 = RUN.sha256_file(SCHEMA_PATH)
    primary = request("primary", prompt_sha256=prompt_sha256, response_schema_sha256=schema_sha256)
    secondary = request("secondary", prompt_sha256=prompt_sha256, response_schema_sha256=schema_sha256)
    batches = RUN.batch_plan([secondary, primary], 12)
    assert len(batches) == 2
    assert {batch[0]["reviewer_slot"] for batch in batches} == {"primary", "secondary"}
    assert all(len({row["reviewer_slot"] for row in batch}) == 1 for batch in batches)

    calls: list[list[str]] = []

    def fake_invoke(requests, **kwargs):
        calls.append([str(row["review_id"]) for row in requests])
        _, execution_schema_sha256 = RUN.execution_schema_bytes(kwargs["response_schema"], len(requests))
        return (
            RUN.InvocationResult(
                responses=tuple(response_for(dict(row)) for row in requests),
                usage={"input_tokens": 7},
                execution_schema_sha256=execution_schema_sha256,
            ),
            1,
        )

    monkeypatch.setattr(RUN, "invoke_with_retries", fake_invoke)
    settings, settings_sha256 = RUN.execution_settings(
        model="gpt-5.6-luna", reasoning_effort="low", codex_bin="codex", batch_size=12
    )
    del settings
    state = tmp_path / "cache"
    first = RUN.execute_batch(
        [primary],
        state_dir=state,
        settings_sha256=settings_sha256,
        prompt_text=prompt.read_text(encoding="utf-8"),
        response_schema=schema(),
        model="gpt-5.6-luna",
        reasoning_effort="low",
        codex_bin="codex",
        timeout_seconds=30,
        max_attempts=3,
    )
    second = RUN.execute_batch(
        [primary],
        state_dir=state,
        settings_sha256=settings_sha256,
        prompt_text=prompt.read_text(encoding="utf-8"),
        response_schema=schema(),
        model="gpt-5.6-luna",
        reasoning_effort="low",
        codex_bin="codex",
        timeout_seconds=30,
        max_attempts=3,
    )
    assert first.cache_status == "created"
    assert second.cache_status == "hit"
    assert len(calls) == 1
    cache_text = first.cache_path.read_text(encoding="utf-8")
    assert primary["review_copy"] not in cache_text
    assert primary["request_sha256"] in cache_text


def test_invoke_batch_uses_ephemeral_read_only_codex_and_validates_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = prompt_file(tmp_path)
    request_row = request(
        prompt_sha256=RUN.sha256_file(prompt),
        response_schema_sha256=RUN.sha256_file(SCHEMA_PATH),
    )
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["prompt"] = kwargs["input"]
        schema_path = Path(command[command.index("--output-schema") + 1])
        execution_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert execution_schema["$defs"]["response"]["properties"]["schema_version"]["type"] == "string"
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps({"responses": [response_for(request_row)]}), encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4}}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = RUN.invoke_batch(
        [request_row],
        prompt_text=prompt.read_text(encoding="utf-8"),
        response_schema=schema(),
        model="gpt-5.6-luna",
        reasoning_effort="low",
        codex_bin="codex-test",
        timeout_seconds=30,
    )
    command = observed["command"]
    assert command[0] == "codex-test"
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-5.6-luna"
    assert "--ask-for-approval" not in command
    assert "REQUESTS_JSON_BEGIN" in str(observed["prompt"])
    assert result.responses[0]["review_id"] == request_row["review_id"]
    assert result.usage == {"input_tokens": 4}


def test_main_emits_response_and_adjudication_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = prompt_file(tmp_path)
    prompt_sha256 = RUN.sha256_file(prompt)
    schema_sha256 = RUN.sha256_file(SCHEMA_PATH)
    primary = request(
        "primary",
        prompt_sha256=prompt_sha256,
        response_schema_sha256=schema_sha256,
        source="source-a",
        source_route="pdf_ocr",
    )
    secondary = request(
        "secondary",
        prompt_sha256=prompt_sha256,
        response_schema_sha256=schema_sha256,
        source="source-a",
        source_route="pdf_ocr",
    )
    html_primary = request(
        "primary",
        prompt_sha256=prompt_sha256,
        response_schema_sha256=schema_sha256,
        source="source-b",
        source_route="html_web",
    )
    requests_path = tmp_path / "requests.jsonl"
    requests_path.write_text(
        "".join(RUN.canonical_json(row) + "\n" for row in (primary, secondary, html_primary)),
        encoding="utf-8",
    )
    observed_slots: list[str] = []

    def fake_invoke(requests, **kwargs):
        request_row = dict(requests[0])
        observed_slots.append(str(request_row["reviewer_slot"]))
        if request_row["source_id"] == "agent1_v3_codex_preflight_fixture":
            response = response_for(request_row)
        elif request_row["reviewer_slot"] == "secondary":
            response = response_for(request_row)
        elif request_row["reviewer_slot"] == "adjudicator":
            response = response_for(request_row, recommendation="include_after_cleaning")
        else:
            response = response_for(request_row, quality_score=5)
        _, execution_schema_sha256 = RUN.execution_schema_bytes(kwargs["response_schema"], len(requests))
        return (
            RUN.InvocationResult(
                responses=(response,), usage={"input_tokens": 3}, execution_schema_sha256=execution_schema_sha256
            ),
            1,
        )

    monkeypatch.setattr(RUN, "invoke_with_retries", fake_invoke)
    output = tmp_path / "responses.jsonl"
    assert (
        RUN.main(
            [
                "--requests",
                str(requests_path),
                "--output",
                str(output),
                "--policy",
                str(POLICY_PATH),
                "--prompt",
                str(prompt),
                "--response-schema",
                str(SCHEMA_PATH),
                "--workers",
                "1",
                "--batch-size",
                "1",
            ],
            environ={"CODEX_REVIEW_MODEL": "gpt-5.6-luna"},
        )
        == 0
    )
    rows = RUN.read_jsonl(output)
    assert [row["reviewer_slot"] for row in rows] == ["primary", "secondary", "primary"]
    assert observed_slots[0] == "primary"  # synthetic preflight
    assert "secondary" in observed_slots

    response_receipt = output.with_name(output.name + ".response_receipt.json")
    adjudication_receipt = output.with_name(output.name + ".adjudication_receipt.json")
    calibration_receipt = output.with_name(output.name + ".calibration_receipt.json")
    response_payload = json.loads(response_receipt.read_text(encoding="utf-8"))
    adjudication_payload = json.loads(adjudication_receipt.read_text(encoding="utf-8"))
    calibration_payload = json.loads(calibration_receipt.read_text(encoding="utf-8"))
    assert response_payload["preflight"]["synthetic_fixture"] is True
    assert response_payload["model"]["accepted_model"] == "gpt-5.6-luna"
    assert response_payload["inputs"]["prompt"]["sha256"] == prompt_sha256
    assert response_payload["corpus_files_read"] is False
    assert adjudication_payload["final_adjudication_manifest"]["status"] == "complete"
    assert adjudication_payload["adjudication_request_rows"] == 0
    assert calibration_payload["status"] == "passed"
    assert {row["logical_source_route"] for row in calibration_payload["assessment"]["logical_route_coverage"]} == {
        "html_web",
        "pdf_ocr",
    }
    assert response_payload["passed_calibration_receipt"]["sha256"] == RUN.sha256_file(calibration_receipt)


def test_failed_route_spanning_calibration_stops_before_full_primary_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = prompt_file(tmp_path)
    prompt_sha256 = RUN.sha256_file(prompt)
    schema_sha256 = RUN.sha256_file(SCHEMA_PATH)
    primary = request(
        "primary",
        prompt_sha256=prompt_sha256,
        response_schema_sha256=schema_sha256,
        source="source-a",
        source_route="pdf_ocr",
    )
    secondary = request(
        "secondary",
        prompt_sha256=prompt_sha256,
        response_schema_sha256=schema_sha256,
        source="source-a",
        source_route="pdf_ocr",
    )
    html_primary = request(
        "primary",
        prompt_sha256=prompt_sha256,
        response_schema_sha256=schema_sha256,
        source="source-b",
        source_route="html_web",
    )
    requests_path = tmp_path / "requests.jsonl"
    requests_path.write_text(
        "".join(RUN.canonical_json(row) + "\n" for row in (primary, secondary, html_primary)),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_invoke(requests, **kwargs):
        request_row = dict(requests[0])
        calls.append(str(request_row["review_id"]))
        if request_row["source_id"] == "agent1_v3_codex_preflight_fixture":
            response = response_for(request_row)
        elif request_row["source_id"] == "source-a" and request_row["reviewer_slot"] == "secondary":
            response = response_for(
                request_row,
                quality_score=1,
                recommendation="exclude",
                confidence_score=2,
            )
        else:
            response = response_for(request_row)
        _, execution_schema_sha256 = RUN.execution_schema_bytes(kwargs["response_schema"], len(requests))
        return (
            RUN.InvocationResult(
                responses=(response,), usage={"input_tokens": 3}, execution_schema_sha256=execution_schema_sha256
            ),
            1,
        )

    monkeypatch.setattr(RUN, "invoke_with_retries", fake_invoke)
    output = tmp_path / "responses.jsonl"
    with pytest.raises(ValueError, match="prompt calibration consistency gate failed"):
        RUN.main(
            [
                "--requests",
                str(requests_path),
                "--output",
                str(output),
                "--policy",
                str(POLICY_PATH),
                "--prompt",
                str(prompt),
                "--response-schema",
                str(SCHEMA_PATH),
                "--workers",
                "1",
                "--batch-size",
                "1",
            ],
            environ={"CODEX_REVIEW_MODEL": "gpt-5.6-luna"},
        )
    calibration_path = output.with_name(output.name + ".calibration_receipt.json")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert calibration["status"] == "failed"
    assert calibration["assessment"]["failure_count"] == 1
    assert not output.exists()
    assert not output.with_name(output.name + ".response_receipt.json").exists()
    # One synthetic preflight plus exactly two route-local primary/secondary
    # pairs; no later full-review invocation may happen after a failed gate.
    assert len(calls) == 5

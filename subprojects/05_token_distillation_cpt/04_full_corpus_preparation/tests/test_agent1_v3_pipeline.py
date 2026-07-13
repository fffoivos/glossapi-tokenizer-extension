from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


PHASE = Path(__file__).resolve().parents[1]
PIPELINE = PHASE / "scripts" / "agent1_v3_pipeline.py"
CONTRACT = PHASE / "scripts" / "agent1_v3_contract.py"

RUN_ID = "agent1-full-corpus-v3-20260713T123456Z-abcdef0"
CHILD_RUN_ID = "agent1-full-corpus-v3-20260713T123457Z-abcdef1"
ORDERED_STAGES = (
    "10-normalize",
    "20-lineage",
    "30-review-packet",
    "35-quality-review-evidence",
    "40-admission",
    "50-dedup",
    "55-greekmmlu-freeze",
    "60-decontamination",
    "65-anonymization-sanitization",
    "70-prestructural-freeze",
    "75-structural-detection-audit",
    "78-structural-apply",
    "80-final-validation",
)


def _write(path: Path, value: object = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(command), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _pipeline(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(PIPELINE, *args, check=check)


def _freeze(tmp_path: Path, *, run_id: str = RUN_ID, prestructural_only: bool = False) -> Path:
    inputs = {
        name: _write(tmp_path / "inputs" / f"{name}.json")
        for name in (
            "source_registry",
            "source_aliases",
            "candidate_roster",
            "post_cutoff_inventory",
            "nanochat_initial_roster",
            "acquisition_receipt",
            "tokenizer",
            "review_policy",
            "review_prompt",
            "review_response_schema",
            "glossapi_build_receipt",
            "license_adjudication",
            "training_eligibility_policy",
            "policy",
        )
    }
    root = tmp_path / "run"
    args = [
        "freeze-run",
        "--run-root",
        str(root),
        "--data-root",
        str(tmp_path / "data" / run_id),
        "--run-id",
        run_id,
        "--code-commit",
        "a" * 40,
        "--source-registry",
        str(inputs["source_registry"]),
        "--source-aliases",
        str(inputs["source_aliases"]),
        "--candidate-roster",
        str(inputs["candidate_roster"]),
        "--post-cutoff-inventory",
        str(inputs["post_cutoff_inventory"]),
        "--nanochat-initial-roster",
        str(inputs["nanochat_initial_roster"]),
        "--acquisition-receipt",
        str(inputs["acquisition_receipt"]),
        "--tokenizer",
        str(inputs["tokenizer"]),
        "--review-policy",
        str(inputs["review_policy"]),
        "--review-prompt",
        str(inputs["review_prompt"]),
        "--review-response-schema",
        str(inputs["review_response_schema"]),
        "--glossapi-build-receipt",
        str(inputs["glossapi_build_receipt"]),
        "--license-adjudication",
        str(inputs["license_adjudication"]),
        "--training-eligibility-policy",
        str(inputs["training_eligibility_policy"]),
        "--dedup-policy",
        str(inputs["policy"]),
        "--greekmmlu-policy",
        str(inputs["policy"]),
        "--anonymization-policy",
        str(inputs["policy"]),
        "--structural-policy",
        str(inputs["policy"]),
    ]
    if prestructural_only:
        args.append("--prestructural-only")
    _run(CONTRACT, *args)
    return root


def _common(root: Path, run_id: str = RUN_ID) -> list[str]:
    return ["--run-root", str(root), "--run-id", run_id]


def _complete_through(root: Path, stage: str, *, run_id: str = RUN_ID) -> None:
    for index, current in enumerate(ORDERED_STAGES):
        attempt = f"attempt-{index:02d}"
        _pipeline("begin", *_common(root, run_id), "--stage", current, "--attempt-id", attempt, "--execute")
        output = _write(root / "stages" / current / "attempts" / attempt / f"{current}.json")
        _pipeline(
            "finish",
            *_common(root, run_id),
            "--stage",
            current,
            "--attempt-id",
            attempt,
            "--output",
            str(output),
            "--execute",
        )
        if current == stage:
            return
    raise AssertionError(f"unknown stage: {stage}")


def test_begin_finish_default_to_dry_run_and_enforce_fixed_order(tmp_path: Path) -> None:
    root = _freeze(tmp_path)

    early = _pipeline(
        "begin", *_common(root), "--stage", "20-lineage", "--attempt-id", "early", check=False
    )
    assert early.returncode != 0
    assert "requires 10-normalize next" in early.stderr

    planned = _pipeline("begin", *_common(root), "--stage", "10-normalize", "--attempt-id", "a0")
    plan = json.loads(planned.stdout)
    assert plan["mode"] == "dry_run"
    assert plan["expected_upstream"] == []
    assert not (root / "stages" / "10-normalize").exists()

    begun = _pipeline(
        "begin", *_common(root), "--stage", "10-normalize", "--attempt-id", "a0", "--execute"
    )
    assert json.loads(begun.stdout)["mode"] == "executed"
    attempt_dir = root / "stages" / "10-normalize" / "attempts" / "a0"
    output = _write(attempt_dir / "normalization_manifest.json")

    finish_plan = _pipeline(
        "finish",
        *_common(root),
        "--stage",
        "10-normalize",
        "--attempt-id",
        "a0",
        "--output",
        str(output),
    )
    assert json.loads(finish_plan.stdout)["mode"] == "dry_run"
    assert not (root / "stages" / "10-normalize" / "stage_receipt.json").exists()

    finished = _pipeline(
        "finish",
        *_common(root),
        "--stage",
        "10-normalize",
        "--attempt-id",
        "a0",
        "--output",
        str(output),
        "--execute",
    )
    assert json.loads(finished.stdout)["mode"] == "executed"
    assert (root / "stages" / "10-normalize" / "COMPLETED").is_file()

    status = json.loads(_pipeline("status", *_common(root)).stdout)
    assert status["completed_stages"] == ["10-normalize"]
    assert status["next_stage"] == "20-lineage"


def test_status_fails_closed_when_a_receipted_output_hash_drifts(tmp_path: Path) -> None:
    root = _freeze(tmp_path)
    _complete_through(root, "10-normalize")
    output = root / "stages" / "10-normalize" / "attempts" / "attempt-00" / "10-normalize.json"
    output.write_text("tampered after receipt\n", encoding="utf-8")

    status = _pipeline("status", *_common(root), check=False)
    assert status.returncode != 0
    assert "bound input drift" in status.stderr


def test_pipeline_wrapper_preserves_exact_retry_contract_and_finishes_retry_attempt(tmp_path: Path) -> None:
    root = _freeze(tmp_path)
    source = _write(tmp_path / "inputs" / "retry-source.json", "fixed input\n")
    first = _pipeline(
        "begin",
        *_common(root),
        "--stage",
        "10-normalize",
        "--attempt-id",
        "failed-job",
        "--parameters-json",
        '{"batch_size":128}',
        "--input",
        "source",
        str(source),
        "--execute",
    )
    assert json.loads(first.stdout)["mode"] == "executed"

    altered_bytes = _pipeline(
        "begin",
        *_common(root),
        "--stage",
        "10-normalize",
        "--attempt-id",
        "bad-retry",
        "--parameters-json",
        '{ "batch_size": 128 }',
        "--input",
        "source",
        str(source),
        "--execute",
        check=False,
    )
    assert altered_bytes.returncode != 0
    assert "not byte-identical" in altered_bytes.stderr

    retry = _pipeline(
        "begin",
        *_common(root),
        "--stage",
        "10-normalize",
        "--attempt-id",
        "retry-job",
        "--parameters-json",
        '{"batch_size":128}',
        "--input",
        "source",
        str(source),
        "--execute",
    )
    assert json.loads(retry.stdout)["mode"] == "executed"
    output = _write(root / "stages" / "10-normalize" / "attempts" / "retry-job" / "manifest.json")
    completed = _pipeline(
        "finish",
        *_common(root),
        "--stage",
        "10-normalize",
        "--attempt-id",
        "retry-job",
        "--output",
        str(output),
        "--execute",
    )
    assert json.loads(completed.stdout)["mode"] == "executed"
    receipt = json.loads((root / "stages" / "10-normalize" / "stage_receipt.json").read_text())
    assert receipt["attempt_id"] == "retry-job"


def test_structural_apply_needs_hash_pinned_agent2_handoff_and_child_manifest(tmp_path: Path) -> None:
    root = _freeze(tmp_path, run_id=CHILD_RUN_ID)
    _complete_through(root, "75-structural-detection-audit", run_id=CHILD_RUN_ID)

    planned_without_handoff = json.loads(_pipeline("plan", *_common(root, CHILD_RUN_ID)).stdout)
    assert planned_without_handoff["stage"] == "78-structural-apply"
    assert planned_without_handoff["status"] == "blocked"
    assert "Agent 2 handoff" in planned_without_handoff["blocked_reason"]

    absent = _pipeline(
        "begin",
        *_common(root, CHILD_RUN_ID),
        "--stage",
        "78-structural-apply",
        "--attempt-id",
        "structural",
        check=False,
    )
    assert absent.returncode != 0
    assert "Agent 2 handoff" in absent.stderr
    assert not (root / "stages" / "78-structural-apply").exists()

    prestructural = _write(
        tmp_path / "parent-prestructural.json",
        {
            "schema_version": "agent1_full_corpus_v3_prestructural_manifest_v1",
            "status": "prestructural_frozen",
            "run_id": RUN_ID,
            "publish_permitted": False,
            "structural_state": "awaiting_agent2_handoff",
        },
    )
    handoff = _write(
        tmp_path / "agent2-handoff.json",
        {
            "schema_version": "agent2_immutable_structural_model_handoff_v1",
            "ready_for_corpus_application": True,
            "python_rust_probability_parity_passed": True,
            "python_rust_decoded_span_parity_passed": True,
            "source_balanced_safety_metrics_passed": True,
            "false_deletion_audit_passed": True,
        },
    )
    common_args = [
        "--agent2-handoff",
        str(handoff),
        "--agent2-handoff-sha256",
        _sha256(handoff),
        "--prestructural-manifest",
        str(prestructural),
        "--prestructural-manifest-sha256",
        _sha256(prestructural),
    ]
    wrong_hash = _pipeline(
        "begin",
        *_common(root, CHILD_RUN_ID),
        "--stage",
        "78-structural-apply",
        "--attempt-id",
        "structural",
        "--agent2-handoff",
        str(handoff),
        "--agent2-handoff-sha256",
        "0" * 64,
        "--prestructural-manifest",
        str(prestructural),
        "--prestructural-manifest-sha256",
        _sha256(prestructural),
        check=False,
    )
    assert wrong_hash.returncode != 0
    assert "Agent 2 immutable handoff SHA-256 mismatch" in wrong_hash.stderr
    assert not (root / "stages" / "78-structural-apply").exists()

    dry = json.loads(
        _pipeline(
            "begin",
            *_common(root, CHILD_RUN_ID),
            "--stage",
            "78-structural-apply",
            "--attempt-id",
            "structural",
            *common_args,
        ).stdout
    )
    assert dry["mode"] == "dry_run"
    assert dry["structural_apply_gate"] == {
        "agent2_handoff_bound": True,
        "imported_prestructural_manifest_bound": True,
        "child_run_required": True,
    }
    assert not (root / "stages" / "78-structural-apply").exists()

    executed = _pipeline(
        "begin",
        *_common(root, CHILD_RUN_ID),
        "--stage",
        "78-structural-apply",
        "--attempt-id",
        "structural",
        *common_args,
        "--execute",
    )
    assert json.loads(executed.stdout)["mode"] == "executed"
    stage_contract = json.loads(
        (root / "stages" / "78-structural-apply" / "stage_contract.json").read_text(encoding="utf-8")
    )
    assert stage_contract["inputs"]["agent2_immutable_handoff"]["sha256"] == _sha256(handoff)
    assert stage_contract["inputs"]["imported_prestructural_manifest"]["sha256"] == _sha256(prestructural)


def test_prestructural_only_run_cannot_plan_structural_apply(tmp_path: Path) -> None:
    root = _freeze(tmp_path, prestructural_only=True)
    result = _pipeline(
        "plan", *_common(root), "--stage", "78-structural-apply", check=False
    )
    assert result.returncode != 0
    assert "unavailable in this run mode" in result.stderr

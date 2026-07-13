from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PHASE = Path(__file__).resolve().parents[1]
SCRIPT = PHASE / "scripts" / "agent1_v3_structural_child.py"
SHELL = PHASE / "clariden" / "agent1_v3_structural_child.sh"

PARENT_RUN_ID = "agent1-full-corpus-v3-20260713T123456Z-abcdef0"
CHILD_RUN_ID = "agent1-full-corpus-v3-20260713T123457Z-abcdef1"
REQUIRED_RELEASE_EVIDENCE_NAMES = (
    "candidate_roster",
    "review_packet",
    "review_requests",
    "review_responses",
    "response_execution_receipt",
    "adjudication_execution_receipt",
    "stage35_review_closure",
    "review_sample_quality_summary",
    "review_sample_quality_handoff",
    "quality_summary",
    "lineage_summary",
    "source_novelty",
    "license_adjudication",
    "review_aggregate",
    "admission_confirmation",
    "transformation_waterfall",
    "anonymization_semantic_clearance",
)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _write(path: Path, value: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _phase10_evidence() -> tuple[dict[str, object], dict[str, object]]:
    bindings = {
        name: {"bytes": index + 1, "sha256": f"{index + 1:064x}"}
        for index, name in enumerate(REQUIRED_RELEASE_EVIDENCE_NAMES)
    }
    clearance = {
        "status": "cleared",
        "sha256": bindings["anonymization_semantic_clearance"]["sha256"],
        "waterfall_sha256": bindings["transformation_waterfall"]["sha256"],
    }
    release = {
        "status": "passed",
        "required_categories": list(REQUIRED_RELEASE_EVIDENCE_NAMES),
        "source_bindings": bindings,
        "anonymization_semantic_false_positive_clearance": clearance,
    }
    handoff = {
        **release,
        "compact_files": {name: dict(binding) for name, binding in bindings.items()},
    }
    return release, handoff


def _structural_module() -> object:
    spec = importlib.util.spec_from_file_location("agent1_v3_structural_child_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], check=check, capture_output=True, text=True
    )


def _fixtures(tmp_path: Path) -> dict[str, Path]:
    corpus = tmp_path / "parent-corpus"
    corpus.mkdir()
    _write(corpus / "source" / "part.parquet", "parquet fixture\n")
    postmask = _write_json(
        tmp_path / "parent" / "postmask-duplicate-report.json",
        {
            "schema_version": "agent1_v3_postmask_duplicate_verification_v1",
            "verification_only": True,
            "dedup_applied": False,
            "material_new_duplicate_count": 0,
        },
    )
    parent = _write_json(
        tmp_path / "parent" / "prestructural.json",
        {
            "schema_version": "agent1_full_corpus_v3_prestructural_manifest_v1",
            "status": "prestructural_frozen",
            "run_id": PARENT_RUN_ID,
            "publish_permitted": False,
            "structural_state": "awaiting_agent2_handoff",
            "corpus_root": str(corpus.resolve()),
            "inputs": {"postmask_duplicate_report": _binding(postmask)},
        },
    )
    handoff = _write_json(
        tmp_path / "agent2" / "structural_model_handoff.json",
        {
            "schema_version": "agent2_immutable_structural_model_handoff_v1",
            "status": "passed",
            "ready_for_corpus_application": True,
            "python_rust_probability_parity_passed": True,
            "python_rust_decoded_span_parity_passed": True,
            "source_balanced_safety_metrics_passed": True,
            "false_deletion_audit_passed": True,
            "artifact_hashes": {"model": "1" * 64, "config": "2" * 64, "code": "3" * 64},
        },
    )
    base_policy = _write_json(
        tmp_path / "policy" / "agent1_v3_policy.json",
        {
            "schema_version": "agent1_full_corpus_v3_policy_v1",
            "structural": {
                "default": "no_op",
                "requires_agent2_immutable_handoff": True,
                "allowlisted_profiles": ["academic_ocr", "academic_sectioned"],
            },
        },
    )
    application_policy = _write_json(
        tmp_path / "policy" / "structural_application.json",
        {
            "schema_version": "agent1_full_corpus_v3_structural_application_policy_v1",
            "status": "approved",
            "explicit_application_approved": True,
            "base_structural_policy_sha256": _sha(base_policy),
            "materialization": {"toc": True, "bibliography": False},
            "source_profile_allowlist": ["academic_ocr"],
            "publish_permitted": False,
        },
    )
    return {
        "corpus": corpus,
        "parent": parent,
        "handoff": handoff,
        "base_policy": base_policy,
        "application_policy": application_policy,
    }


def _init(tmp_path: Path, fixtures: dict[str, Path]) -> tuple[Path, Path]:
    run_root = tmp_path / "child-run"
    data_root = tmp_path / "child-data"
    _run(
        "init",
        "--run-root",
        str(run_root),
        "--data-root",
        str(data_root),
        "--run-id",
        CHILD_RUN_ID,
        "--prestructural-manifest",
        str(fixtures["parent"]),
        "--prestructural-manifest-sha256",
        _sha(fixtures["parent"]),
        "--agent2-handoff",
        str(fixtures["handoff"]),
        "--agent2-handoff-sha256",
        _sha(fixtures["handoff"]),
        "--base-structural-policy",
        str(fixtures["base_policy"]),
        "--application-policy",
        str(fixtures["application_policy"]),
    )
    return run_root, data_root


def _contract(run_root: Path) -> dict[str, object]:
    return json.loads((run_root / "structural_child_run_contract.json").read_text(encoding="utf-8"))


def _attempt(run_root: Path, data_root: Path, stage: str, attempt: str) -> tuple[Path, Path]:
    return (
        run_root / "stages" / stage / "attempts" / attempt,
        data_root / "stages" / stage / "attempts" / attempt,
    )


def test_child_init_requires_handoff_hash_and_explicit_materialization_policy(tmp_path: Path) -> None:
    fixtures = _fixtures(tmp_path)
    run_root, _ = _init(tmp_path, fixtures)
    contract = _contract(run_root)
    assert contract["publish_permitted"] is False
    assert contract["materialization"] == {"toc": True, "bibliography": False}
    assert contract["source_profile_allowlist"] == ["academic_ocr"]

    bad_handoff = _run(
        "init",
        "--run-root",
        str(tmp_path / "bad-handoff-run"),
        "--data-root",
        str(tmp_path / "bad-handoff-data"),
        "--run-id",
        "agent1-full-corpus-v3-20260713T123458Z-abcdef2",
        "--prestructural-manifest",
        str(fixtures["parent"]),
        "--prestructural-manifest-sha256",
        _sha(fixtures["parent"]),
        "--agent2-handoff",
        str(fixtures["handoff"]),
        "--agent2-handoff-sha256",
        "0" * 64,
        "--base-structural-policy",
        str(fixtures["base_policy"]),
        "--application-policy",
        str(fixtures["application_policy"]),
        check=False,
    )
    assert bad_handoff.returncode != 0
    assert "Agent 2 immutable handoff SHA-256 mismatch" in bad_handoff.stderr

    disabled_policy = _write_json(
        tmp_path / "policy" / "disabled-application.json",
        {
            "schema_version": "agent1_full_corpus_v3_structural_application_policy_v1",
            "status": "approved",
            "explicit_application_approved": True,
            "base_structural_policy_sha256": _sha(fixtures["base_policy"]),
            "materialization": {"toc": False, "bibliography": False},
            "source_profile_allowlist": ["academic_ocr"],
            "publish_permitted": False,
        },
    )
    disabled = _run(
        "init",
        "--run-root",
        str(tmp_path / "disabled-run"),
        "--data-root",
        str(tmp_path / "disabled-data"),
        "--run-id",
        "agent1-full-corpus-v3-20260713T123459Z-abcdef3",
        "--prestructural-manifest",
        str(fixtures["parent"]),
        "--prestructural-manifest-sha256",
        _sha(fixtures["parent"]),
        "--agent2-handoff",
        str(fixtures["handoff"]),
        "--agent2-handoff-sha256",
        _sha(fixtures["handoff"]),
        "--base-structural-policy",
        str(fixtures["base_policy"]),
        "--application-policy",
        str(disabled_policy),
        check=False,
    )
    assert disabled.returncode != 0
    assert "must enable ToC and/or bibliography" in disabled.stderr


def test_child_stages_close_only_with_receipted_audit_apply_and_private_validation(tmp_path: Path) -> None:
    fixtures = _fixtures(tmp_path)
    run_root, data_root = _init(tmp_path, fixtures)
    contract = _contract(run_root)
    parent_binding = contract["parent"]["prestructural_manifest"]
    handoff_binding = contract["agent2_handoff"]
    application_binding = contract["application_policy"]
    allowlist = contract["source_profile_allowlist"]

    stage75 = "75-structural-detection-audit"
    _run("begin", "--run-root", str(run_root), "--stage", stage75, "--attempt-id", "audit")
    metadata75, bulk75 = _attempt(run_root, data_root, stage75, "audit")
    detection_root = bulk75 / "raw-detection"
    detection_root.mkdir(parents=True)
    detected_spans = _write(bulk75 / "detected-spans.jsonl", "{}\n")
    detection = _write_json(
        metadata75 / "detection-manifest.json",
        {
            "schema_version": "agent1_full_corpus_v3_structural_detection_manifest_v1",
            "status": "passed",
            "child_run_id": CHILD_RUN_ID,
            "input_root": str(fixtures["corpus"].resolve()),
            "output_root": str(detection_root.resolve()),
            "parent_prestructural_manifest": parent_binding,
            "agent2_handoff": handoff_binding,
            "application_policy": application_binding,
            "source_profile_allowlist": allowlist,
            "non_allowlisted_mode": "explicit_no_op",
            "detected_span_ledger": _binding(detected_spans),
        },
    )
    manual_audit = _write(metadata75 / "manual-audit.json")
    audit = _write_json(
        metadata75 / "audit-manifest.json",
        {
            "schema_version": "agent1_full_corpus_v3_structural_audit_manifest_v1",
            "status": "passed",
            "child_run_id": CHILD_RUN_ID,
            "detection_manifest": _binding(detection),
            "agent2_handoff": handoff_binding,
            "application_policy": application_binding,
            "source_profile_allowlist": allowlist,
            "python_rust_probability_parity_passed": True,
            "python_rust_decoded_span_parity_passed": True,
            "source_balanced_safety_metrics_passed": True,
            "false_deletion_audit_passed": True,
            "independent_source_balanced_safety_metrics_passed": True,
            "targeted_false_deletion_audit_cases": 100,
            "manual_false_deletion_audit_receipt": _binding(manual_audit),
        },
    )
    _run(
        "finish-audit",
        "--run-root",
        str(run_root),
        "--detection-manifest",
        str(detection),
        "--audit-manifest",
        str(audit),
    )

    stage78 = "78-structural-apply"
    _run("begin", "--run-root", str(run_root), "--stage", stage78, "--attempt-id", "apply")
    metadata78, bulk78 = _attempt(run_root, data_root, stage78, "apply")
    final_root = bulk78 / "structural-output"
    final_root.mkdir(parents=True)
    _write(final_root / "source" / "part.parquet", "parquet fixture\n")
    structural_ledger = _write(bulk78 / "structural-ledger.parquet")
    span_ledger = _write(bulk78 / "structural-spans.jsonl", "{}\n")
    noop_ledger = _write(bulk78 / "nonallowlisted-noops.jsonl", "{}\n")
    duplicate_report = _write_json(
        bulk78 / "poststructural-duplicates.json",
        {
            "verification_only": True,
            "dedup_applied": False,
            "material_new_duplicate_count": 0,
            "input_root": str(final_root.resolve()),
        },
    )
    apply = _write_json(
        metadata78 / "apply-manifest.json",
        {
            "schema_version": "agent1_full_corpus_v3_structural_apply_manifest_v1",
            "status": "passed",
            "mode": "applied",
            "child_run_id": CHILD_RUN_ID,
            "input_root": str(fixtures["corpus"].resolve()),
            "output_root": str(final_root.resolve()),
            "parent_prestructural_manifest": parent_binding,
            "agent2_handoff": handoff_binding,
            "application_policy": application_binding,
            "audit_manifest": _binding(audit),
            "structural_ledger": _binding(structural_ledger),
            "structural_span_ledger": _binding(span_ledger),
            "nonallowlisted_noop_ledger": _binding(noop_ledger),
            "poststructural_duplicate_report": _binding(duplicate_report),
            "source_profile_allowlist": allowlist,
            "ready_for_application": True,
            "python_rust_probability_parity_passed": True,
            "python_rust_decoded_span_parity_passed": True,
            "source_balanced_safety_metrics_passed": True,
            "false_deletion_audit_passed": True,
            "non_allowlisted_mode": "explicit_no_op",
        },
    )
    _run(
        "finish-apply",
        "--run-root",
        str(run_root),
        "--audit-manifest",
        str(audit),
        "--apply-manifest",
        str(apply),
        "--structural-ledger",
        str(structural_ledger),
        "--structural-span-ledger",
        str(span_ledger),
        "--nonallowlisted-noop-ledger",
        str(noop_ledger),
        "--poststructural-duplicate-report",
        str(duplicate_report),
    )

    stage80 = "80-final-validation"
    _run("begin", "--run-root", str(run_root), "--stage", stage80, "--attempt-id", "validate")
    metadata80, _ = _attempt(run_root, data_root, stage80, "validate")
    release_contract_sha = "a" * 64
    release_evidence, handoff_evidence = _phase10_evidence()
    release = _write_json(
        metadata80 / "release-manifest.json",
        {
            "schema_version": "agent1_full_corpus_v3_release_manifest_v1",
            "status": "passed",
            "release_kind": "local_private_training_release",
            "publish_permitted": False,
            "release_contract_sha256": release_contract_sha,
            "ordered_transform_contract": {
                "dedup_before_greekmmlu": True,
                "greekmmlu_before_anonymization": True,
                "anonymization_before_structural": True,
                "structural_mode": "applied",
            },
            "upstream_bindings": {
                f"evidence_{name}": binding
                for name, binding in release_evidence["source_bindings"].items()
            },
            "required_evidence": release_evidence,
        },
    )
    release_validation = _write_json(
        metadata80 / "release-validation.json",
        {
            "schema_version": "agent1_full_corpus_v3_release_validation_v1",
            "status": "passed",
            "failed_checks": [],
            "release_contract_sha256": release_contract_sha,
        },
    )
    handoff = _write_json(
        metadata80 / "site-handoff.json",
        {
            "schema_version": "agent1_full_corpus_v3_dataset_review_site_handoff_v1",
            "status": "passed",
            "release_is_public_dataset": False,
            "release_contract_sha256": release_contract_sha,
            "required_evidence": handoff_evidence,
        },
    )
    final_validation = _write_json(
        metadata80 / "final-validation-manifest.json",
        {
            "schema_version": "agent1_full_corpus_v3_final_validation_manifest_v1",
            "status": "passed",
            "child_run_id": CHILD_RUN_ID,
            "local_release_only": True,
            "publish_permitted": False,
            "no_hf_publish_without_user_confirmation": True,
            "structural_apply_manifest": _binding(apply),
            "release_manifest": _binding(release),
            "release_validation": _binding(release_validation),
            "site_handoff": _binding(handoff),
        },
    )
    _run(
        "finish-final",
        "--run-root",
        str(run_root),
        "--apply-manifest",
        str(apply),
        "--final-validation-manifest",
        str(final_validation),
        "--release-manifest",
        str(release),
        "--release-validation",
        str(release_validation),
        "--site-handoff",
        str(handoff),
    )
    status = json.loads(_run("status", "--run-root", str(run_root)).stdout)
    assert status["completed_stages"] == [stage75, stage78, stage80]
    assert status["next_stage"] is None
    assert status["publish_permitted"] is False


def test_shell_handler_is_manual_cpu_gate_not_legacy_or_publish_path() -> None:
    subprocess.run(["bash", "-n", str(SHELL)], check=True)
    text = SHELL.read_text(encoding="utf-8")
    assert "CONFIRM_STRUCTURAL_CHILD_EXECUTION=1" in text
    assert "agent1_v3_require_compute_cpu" in text
    assert "agent1_v3_mask_gpu_visibility" in text
    assert 'python3 "$CHILD_SCRIPT"' in text
    assert "require_child_runtime" in text
    assert '"$AGENT1_V3_RUNTIME_VENV/bin/python" "$CHILD_SCRIPT"' not in text
    assert "refuse_publish" in text
    assert "finish-audit" in text and "finish-apply" in text and "finish-final" in text
    assert "structural_span_production.py" not in text
    assert "finalize_structural_cleaning.py" not in text
    assert "agent1_v3_submit.sh" not in text


def test_structural_final_gate_rejects_pending_or_missing_phase10_evidence() -> None:
    module = _structural_module()
    release_evidence, handoff_evidence = _phase10_evidence()
    release = {
        "required_evidence": release_evidence,
        "upstream_bindings": {
            f"evidence_{name}": binding
            for name, binding in release_evidence["source_bindings"].items()
        },
    }
    module._validate_required_release_evidence(release, {"required_evidence": handoff_evidence})

    pending = json.loads(json.dumps(handoff_evidence))
    pending["anonymization_semantic_false_positive_clearance"]["status"] = "pending"
    with pytest.raises(ValueError, match="semantic false-positive clearance is incomplete"):
        module._validate_required_release_evidence(release, {"required_evidence": pending})

    missing = json.loads(json.dumps(handoff_evidence))
    del missing["compact_files"]["quality_summary"]
    with pytest.raises(ValueError, match="complete compact evidence inventory"):
        module._validate_required_release_evidence(release, {"required_evidence": missing})

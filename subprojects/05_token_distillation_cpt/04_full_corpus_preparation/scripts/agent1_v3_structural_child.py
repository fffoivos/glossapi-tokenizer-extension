#!/usr/bin/env python3
"""Fail-closed contract lane for Agent 1 v3's structural child run.

The normal Agent 1 v3 run deliberately stops at the immutable prestructural
corpus.  This utility is the *separate* three-stage child lane that can consume
that corpus only after Agent 2 has published a hash-pinned handoff and an
explicit structural materialization policy has been approved.

It intentionally does not invoke the legacy Phase-04 structural scripts.  The
legacy detector/finalizer are bound to the old mixed-cleaning Stage-50 schema,
which is incompatible with the required v3 ordering.  Instead, this utility
creates and validates the narrow receipt contract a future v3-compatible
detector/application backend must satisfy:

    75 structural detection + independent safety audit
    78 structural application + verification-only duplicate scan
    80 local private-release validation

There is no publication command here.  Every child contract and final receipt
sets ``publish_permitted`` to false.

The separate application-policy JSON is intentionally required even though the
frozen base policy defaults to ``no_op``.  Its minimum shape is::

  {
    "schema_version": "agent1_full_corpus_v3_structural_application_policy_v1",
    "status": "approved",
    "explicit_application_approved": true,
    "base_structural_policy_sha256": "<sha256 of agent1_v3_policy.json>",
    "materialization": {"toc": true, "bibliography": false},
    "source_profile_allowlist": ["academic_ocr"],
    "publish_permitted": false
  }

The approval document is an immutable input, not a switch inferred from a
model score.  It must remain outside source corpus data and is bound by hash in
all child-stage receipts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CHILD_RUN_SCHEMA = "agent1_full_corpus_v3_structural_child_run_contract_v1"
CHILD_STAGE_SCHEMA = "agent1_full_corpus_v3_structural_child_stage_contract_v1"
CHILD_RECEIPT_SCHEMA = "agent1_full_corpus_v3_structural_child_stage_receipt_v1"
CHILD_GATE_SCHEMA = "agent1_full_corpus_v3_structural_child_gate_v1"
PRESTRUCTURAL_SCHEMA = "agent1_full_corpus_v3_prestructural_manifest_v1"
AGENT2_HANDOFF_SCHEMA = "agent2_immutable_structural_model_handoff_v1"
APPLICATION_POLICY_SCHEMA = "agent1_full_corpus_v3_structural_application_policy_v1"
DETECTION_SCHEMA = "agent1_full_corpus_v3_structural_detection_manifest_v1"
AUDIT_SCHEMA = "agent1_full_corpus_v3_structural_audit_manifest_v1"
APPLY_SCHEMA = "agent1_full_corpus_v3_structural_apply_manifest_v1"
FINAL_VALIDATION_SCHEMA = "agent1_full_corpus_v3_final_validation_manifest_v1"
RELEASE_SCHEMA = "agent1_full_corpus_v3_release_manifest_v1"
RELEASE_VALIDATION_SCHEMA = "agent1_full_corpus_v3_release_validation_v1"
SITE_HANDOFF_SCHEMA = "agent1_full_corpus_v3_dataset_review_site_handoff_v1"

# Phase 10 is fail-closed: Stage 80 may receipt a private local release only
# after the compact Agent-3 evidence closure has all Stage 30/35/40 evidence
# plus the independently completed anonymization semantic review.
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

STAGES = (
    "75-structural-detection-audit",
    "78-structural-apply",
    "80-final-validation",
)
RUN_ID_RE = re.compile(r"^agent1-full-corpus-v3-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{7,40}$")
ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AGENT2_GATES = {
    "ready_for_corpus_application": True,
    "python_rust_probability_parity_passed": True,
    "python_rust_decoded_span_parity_passed": True,
    "source_balanced_safety_metrics_passed": True,
    "false_deletion_audit_passed": True,
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Mapping[str, Any], *, field: str) -> str:
    payload = dict(value)
    payload.pop("created_at", None)
    payload.pop(field, None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _binding(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _binding_with_expected_hash(path: Path, expected: str, *, label: str) -> dict[str, Any]:
    expected = _require_sha256(expected, label=f"{label} SHA-256")
    actual = _binding(path, label=label)
    if actual["sha256"] != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, actual {actual['sha256']}"
        )
    return actual


def _verify_binding(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding must be an object")
    path = Path(str(value.get("path", "")))
    expected_bytes = value.get("bytes")
    expected_sha = _require_sha256(value.get("sha256"), label=f"{label} binding SHA-256")
    actual = _binding(path, label=label)
    if actual["bytes"] != expected_bytes or actual["sha256"] != expected_sha:
        raise ValueError(f"{label} binding drift: {actual['path']}")
    return actual


def _require_exact_binding(value: object, expected: Mapping[str, Any], *, label: str) -> None:
    actual = _verify_binding(value, label=label)
    if actual != dict(expected):
        raise ValueError(f"{label} does not match the immutable child contract")


def _atomic_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_under(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not _is_relative_to(resolved, root.resolve()):
        raise ValueError(f"{label} must remain under {root}: {resolved}")
    return resolved


def _require_run_id(value: str) -> str:
    if not RUN_ID_RE.fullmatch(value):
        raise ValueError("child run id must use the Agent 1 v3 immutable run-id format")
    return value


def _require_attempt_id(value: str) -> str:
    if not ATTEMPT_ID_RE.fullmatch(value):
        raise ValueError("attempt id must use only letters, digits, '.', '_' or '-'")
    return value


def _unique_strings(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must be a non-empty list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    return list(value)


def _require_bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _require_zero_count(value: object, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise ValueError(f"{label} must be exactly zero")


def _parent_manifest(
    path: Path, expected_sha256: str, *, child_run_id: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    binding = _binding_with_expected_hash(path, expected_sha256, label="prestructural manifest")
    payload = _read_object(path, label="prestructural manifest")
    required = {
        "schema_version": PRESTRUCTURAL_SCHEMA,
        "status": "prestructural_frozen",
        "publish_permitted": False,
        "structural_state": "awaiting_agent2_handoff",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"prestructural manifest fails child gate: {key}")
    source_run_id = payload.get("run_id")
    if not isinstance(source_run_id, str) or not RUN_ID_RE.fullmatch(source_run_id):
        raise ValueError("prestructural manifest has no valid source run_id")
    if source_run_id == child_run_id:
        raise ValueError("structural work must use a distinct child run id")
    corpus_root_raw = payload.get("corpus_root")
    if not isinstance(corpus_root_raw, str) or not corpus_root_raw:
        raise ValueError("prestructural manifest lacks its immutable corpus_root")
    corpus_root = Path(corpus_root_raw).resolve()
    if not corpus_root.is_dir():
        raise ValueError(f"prestructural corpus root is unavailable: {corpus_root}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("prestructural manifest lacks protected input bindings")
    postmask = inputs.get("postmask_duplicate_report")
    if postmask is None:
        raise ValueError("prestructural manifest lacks post-mask duplicate verification")
    postmask_binding = _verify_binding(postmask, label="prestructural post-mask duplicate report")
    report = _read_object(Path(postmask_binding["path"]), label="prestructural post-mask duplicate report")
    count = report.get("material_new_duplicate_count", report.get("new_duplicate_count"))
    _require_zero_count(count, label="prestructural post-mask material duplicate count")
    return payload, binding, corpus_root


def _agent2_handoff(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    binding = _binding_with_expected_hash(path, expected_sha256, label="Agent 2 immutable handoff")
    payload = _read_object(path, label="Agent 2 immutable handoff")
    if payload.get("schema_version") != AGENT2_HANDOFF_SCHEMA:
        raise ValueError("Agent 2 handoff has an unsupported immutable schema")
    if payload.get("status") != "passed":
        raise ValueError("Agent 2 handoff is not passed")
    for key, expected in AGENT2_GATES.items():
        if payload.get(key) is not expected:
            raise ValueError(f"Agent 2 handoff does not pass required gate: {key}")
    artifact_hashes = payload.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError("Agent 2 handoff lacks exact model/config/code artifact hashes")
    normalized: dict[str, str] = {}
    for name in ("model", "config", "code"):
        normalized[name] = _require_sha256(
            artifact_hashes.get(name), label=f"Agent 2 {name} artifact SHA-256"
        )
    return payload, binding, normalized


def _base_policy(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    binding = _binding(path, label="frozen structural base policy")
    payload = _read_object(path, label="frozen structural base policy")
    if payload.get("schema_version") != "agent1_full_corpus_v3_policy_v1":
        raise ValueError("frozen structural base policy has an unsupported schema")
    structural = payload.get("structural")
    if not isinstance(structural, Mapping):
        raise ValueError("frozen policy lacks a structural section")
    if structural.get("default") != "no_op":
        raise ValueError("frozen structural base policy must default to no_op")
    if structural.get("requires_agent2_immutable_handoff") is not True:
        raise ValueError("frozen structural base policy must require Agent 2's immutable handoff")
    allowlist = _unique_strings(
        structural.get("allowlisted_profiles"), label="frozen structural source-profile allowlist"
    )
    allowed_profiles = {"academic_ocr", "academic_sectioned"}
    unexpected = sorted(set(allowlist) - allowed_profiles)
    if unexpected:
        raise ValueError(
            "frozen structural allowlist contains non-academic profiles: " + ", ".join(unexpected)
        )
    return payload, binding, sorted(allowlist)


def _application_policy(
    path: Path, *, base_binding: Mapping[str, Any], allowed_profiles: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool], list[str]]:
    binding = _binding(path, label="explicit structural application policy")
    payload = _read_object(path, label="explicit structural application policy")
    required = {
        "schema_version": APPLICATION_POLICY_SCHEMA,
        "status": "approved",
        "explicit_application_approved": True,
        "publish_permitted": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"explicit structural application policy fails required gate: {key}")
    if payload.get("base_structural_policy_sha256") != base_binding["sha256"]:
        raise ValueError("explicit structural application policy is not bound to the frozen base policy")
    materialization = payload.get("materialization")
    if not isinstance(materialization, Mapping):
        raise ValueError("explicit structural application policy lacks materialization choices")
    selected = {
        "toc": _require_bool(materialization.get("toc"), label="ToC materialization flag"),
        "bibliography": _require_bool(
            materialization.get("bibliography"), label="bibliography materialization flag"
        ),
    }
    if not any(selected.values()):
        raise ValueError("explicit structural application policy must enable ToC and/or bibliography")
    profiles = sorted(
        _unique_strings(payload.get("source_profile_allowlist"), label="application source-profile allowlist")
    )
    forbidden = sorted(set(profiles) - set(allowed_profiles))
    if forbidden:
        raise ValueError(
            "explicit structural application policy expands the frozen academic allowlist: "
            + ", ".join(forbidden)
        )
    return payload, binding, selected, profiles


def _validate_storage(run_root: Path, data_root: Path) -> tuple[Path, Path]:
    run = run_root.resolve()
    data = data_root.resolve()
    if run == data or _is_relative_to(run, data) or _is_relative_to(data, run):
        raise ValueError("child metadata and bulk-data roots must be distinct and non-overlapping")
    return run, data


def _child_contract_path(run_root: Path) -> Path:
    return run_root / "structural_child_run_contract.json"


def _stage_root(run_root: Path, stage: str) -> Path:
    return run_root / "stages" / stage


def _stage_contract_path(run_root: Path, stage: str) -> Path:
    return _stage_root(run_root, stage) / "stage_contract.json"


def _stage_receipt_path(run_root: Path, stage: str) -> Path:
    return _stage_root(run_root, stage) / "stage_receipt.json"


def _attempt_roots(contract: Mapping[str, Any], stage: str, attempt_id: str) -> tuple[Path, Path]:
    run_root = Path(str(contract["run_root"])).resolve()
    data_root = Path(str(contract["data_root"])).resolve()
    return (
        _stage_root(run_root, stage) / "attempts" / attempt_id,
        data_root / "stages" / stage / "attempts" / attempt_id,
    )


def _validate_child_contract(payload: Mapping[str, Any], *, expected_run_root: Path | None = None) -> dict[str, Any]:
    if payload.get("schema_version") != CHILD_RUN_SCHEMA or payload.get("status") != "ready":
        raise ValueError("unsupported or inactive structural child run contract")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str):
        raise ValueError("structural child contract lacks run_id")
    _require_run_id(run_id)
    run_root_raw = payload.get("run_root")
    data_root_raw = payload.get("data_root")
    if not isinstance(run_root_raw, str) or not isinstance(data_root_raw, str):
        raise ValueError("structural child contract lacks storage roots")
    run_root, data_root = _validate_storage(Path(run_root_raw), Path(data_root_raw))
    if expected_run_root is not None and run_root != expected_run_root.resolve():
        raise ValueError("structural child contract run_root drift")
    if payload.get("stage_graph") != list(STAGES):
        raise ValueError("structural child contract stage graph drift")
    if payload.get("publish_permitted") is not False:
        raise ValueError("structural child contract must never permit publication")
    if payload.get("contract_sha256") != _digest(payload, field="contract_sha256"):
        raise ValueError("structural child contract digest drift")

    parent = payload.get("parent")
    if not isinstance(parent, Mapping):
        raise ValueError("structural child contract lacks parent binding")
    parent_binding = _verify_binding(parent.get("prestructural_manifest"), label="child parent prestructural manifest")
    parent_payload, _, parent_root = _parent_manifest(
        Path(parent_binding["path"]), parent_binding["sha256"], child_run_id=run_id
    )
    if parent.get("source_run_id") != parent_payload.get("run_id"):
        raise ValueError("structural child contract parent source run drift")
    if parent.get("corpus_root") != str(parent_root):
        raise ValueError("structural child contract parent corpus root drift")

    handoff_binding = _verify_binding(payload.get("agent2_handoff"), label="child Agent 2 handoff")
    _, _, artifact_hashes = _agent2_handoff(Path(handoff_binding["path"]), handoff_binding["sha256"])
    if payload.get("agent2_artifact_hashes") != artifact_hashes:
        raise ValueError("structural child contract Agent 2 artifact-hash drift")

    base_binding = _verify_binding(payload.get("base_structural_policy"), label="child base structural policy")
    _, _, base_allowlist = _base_policy(Path(base_binding["path"]))
    application_binding = _verify_binding(payload.get("application_policy"), label="child application policy")
    _, _, materialization, profiles = _application_policy(
        Path(application_binding["path"]), base_binding=base_binding, allowed_profiles=base_allowlist
    )
    if payload.get("materialization") != materialization:
        raise ValueError("structural child contract materialization policy drift")
    if payload.get("source_profile_allowlist") != profiles:
        raise ValueError("structural child contract source-profile allowlist drift")
    return dict(payload)


def _load_child_contract(run_root: Path) -> dict[str, Any]:
    root = run_root.resolve()
    path = _child_contract_path(root)
    payload = _read_object(path, label="structural child run contract")
    contract = _validate_child_contract(payload, expected_run_root=root)
    gate_path = root / "structural_child_gate.json"
    gate = _read_object(gate_path, label="structural child gate")
    expected_gate = _stage_gate(contract)
    expected_gate["gate_sha256"] = _digest(expected_gate, field="gate_sha256")
    if gate != expected_gate:
        raise ValueError("structural child gate drift")
    return contract


def _stage_expected_inputs(contract: Mapping[str, Any], stage: str) -> dict[str, dict[str, Any]]:
    if stage == STAGES[0]:
        parent = contract["parent"]
        return {
            "parent_prestructural_manifest": dict(parent["prestructural_manifest"]),
            "agent2_immutable_handoff": dict(contract["agent2_handoff"]),
            "base_structural_policy": dict(contract["base_structural_policy"]),
            "explicit_application_policy": dict(contract["application_policy"]),
        }
    previous = STAGES[STAGES.index(stage) - 1]
    return {"upstream_stage_receipt": _binding(_stage_receipt_path(Path(contract["run_root"]), previous), label=f"{previous} receipt")}


def _validate_stage_contract(contract: Mapping[str, Any], stage: str) -> dict[str, Any]:
    path = _stage_contract_path(Path(contract["run_root"]), stage)
    payload = _read_object(path, label=f"{stage} child stage contract")
    required = {
        "schema_version": CHILD_STAGE_SCHEMA,
        "run_id": contract["run_id"],
        "stage": stage,
        "child_contract_sha256": contract["contract_sha256"],
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"{stage} child stage contract drift: {key}")
    if payload.get("contract_sha256") != _digest(payload, field="contract_sha256"):
        raise ValueError(f"{stage} child stage contract digest drift")
    attempt = payload.get("attempt_id")
    if not isinstance(attempt, str):
        raise ValueError(f"{stage} child stage contract lacks attempt id")
    _require_attempt_id(attempt)
    metadata, bulk = _attempt_roots(contract, stage, attempt)
    if payload.get("storage") != {
        "metadata_attempt_dir": str(metadata.resolve()),
        "data_attempt_dir": str(bulk.resolve()),
    }:
        raise ValueError(f"{stage} child stage storage boundary drift")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"{stage} child stage lacks inputs")
    expected_inputs = _stage_expected_inputs(contract, stage)
    if set(inputs) != set(expected_inputs):
        raise ValueError(f"{stage} child stage input names drift")
    for name, expected in expected_inputs.items():
        _require_exact_binding(inputs.get(name), expected, label=f"{stage} input {name}")
    return dict(payload)


def _load_stage_receipt(contract: Mapping[str, Any], stage: str) -> dict[str, Any]:
    stage_contract = _validate_stage_contract(contract, stage)
    path = _stage_receipt_path(Path(contract["run_root"]), stage)
    payload = _read_object(path, label=f"{stage} child stage receipt")
    required = {
        "schema_version": CHILD_RECEIPT_SCHEMA,
        "status": "passed",
        "run_id": contract["run_id"],
        "stage": stage,
        "child_contract_sha256": contract["contract_sha256"],
        "stage_contract_sha256": stage_contract["contract_sha256"],
        "attempt_id": stage_contract["attempt_id"],
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"{stage} child stage receipt drift: {key}")
    if payload.get("receipt_sha256") != _digest(payload, field="receipt_sha256"):
        raise ValueError(f"{stage} child stage receipt digest drift")
    marker = _stage_root(Path(contract["run_root"]), stage) / "COMPLETED"
    expected_marker = f"{_sha256_file(path)}  stage_receipt.json\n"
    if not marker.is_file() or marker.read_text(encoding="utf-8") != expected_marker:
        raise ValueError(f"{stage} child stage receipt COMPLETED marker drift")
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"{stage} child stage receipt has no outputs")
    metadata, bulk = _attempt_roots(contract, stage, stage_contract["attempt_id"])
    for output in outputs:
        actual = _verify_binding(output, label=f"{stage} output")
        output_path = Path(actual["path"])
        if not _is_relative_to(output_path, metadata) and not _is_relative_to(output_path, bulk):
            raise ValueError(f"{stage} child stage output escapes its attempt roots")
    return dict(payload)


def _progress(contract: Mapping[str, Any]) -> tuple[list[str], str | None, str | None]:
    completed: list[str] = []
    in_progress: str | None = None
    seen_gap = False
    for stage in STAGES:
        root = _stage_root(Path(contract["run_root"]), stage)
        receipt = _stage_receipt_path(Path(contract["run_root"]), stage)
        if receipt.exists():
            if seen_gap:
                raise ValueError(f"{stage} completed after an unfinished earlier child stage")
            _load_stage_receipt(contract, stage)
            completed.append(stage)
            continue
        seen_gap = True
        if root.exists():
            if in_progress is not None:
                raise ValueError(f"multiple unfinished child stages: {in_progress}, {stage}")
            _validate_stage_contract(contract, stage)
            in_progress = stage
    next_stage = next((stage for stage in STAGES if stage not in completed), None)
    if in_progress is not None and in_progress != next_stage:
        raise ValueError("unfinished child stage is not next in fixed order")
    return completed, in_progress, next_stage


def _assert_next(contract: Mapping[str, Any], stage: str, *, action: str) -> tuple[list[str], str | None]:
    if stage not in STAGES:
        raise ValueError(f"unsupported child structural stage: {stage}")
    completed, in_progress, next_stage = _progress(contract)
    if next_stage is None:
        raise ValueError("all structural child stages are already complete")
    if next_stage != stage:
        raise ValueError(f"cannot {action} {stage}; fixed child order requires {next_stage} next")
    return completed, in_progress


def _stage_gate(contract: Mapping[str, Any]) -> dict[str, Any]:
    parent = contract["parent"]
    return {
        "schema_version": CHILD_GATE_SCHEMA,
        "status": "passed",
        "child_run_id": contract["run_id"],
        "parent_run_id": parent["source_run_id"],
        "parent_prestructural_manifest": parent["prestructural_manifest"],
        "parent_corpus_root": parent["corpus_root"],
        "agent2_handoff": contract["agent2_handoff"],
        "agent2_artifact_hashes": contract["agent2_artifact_hashes"],
        "base_structural_policy": contract["base_structural_policy"],
        "application_policy": contract["application_policy"],
        "materialization": contract["materialization"],
        "source_profile_allowlist": contract["source_profile_allowlist"],
        "permissions": {
            "detection": True,
            "application": True,
            "local_private_release_validation": True,
            "publish_permitted": False,
        },
    }


def _validate_detection(
    path: Path, *, contract: Mapping[str, Any], data_attempt: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    binding = _binding(path, label="structural detection manifest")
    payload = _read_object(path, label="structural detection manifest")
    required = {
        "schema_version": DETECTION_SCHEMA,
        "status": "passed",
        "child_run_id": contract["run_id"],
        "input_root": contract["parent"]["corpus_root"],
        "non_allowlisted_mode": "explicit_no_op",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"structural detection manifest drift: {key}")
    _require_exact_binding(
        payload.get("parent_prestructural_manifest"),
        contract["parent"]["prestructural_manifest"],
        label="structural detection parent manifest",
    )
    _require_exact_binding(
        payload.get("agent2_handoff"), contract["agent2_handoff"], label="structural detection Agent 2 handoff"
    )
    _require_exact_binding(
        payload.get("application_policy"), contract["application_policy"], label="structural detection application policy"
    )
    if payload.get("source_profile_allowlist") != contract["source_profile_allowlist"]:
        raise ValueError("structural detection source-profile allowlist drift")
    output_root_raw = payload.get("output_root")
    if not isinstance(output_root_raw, str) or not output_root_raw:
        raise ValueError("structural detection manifest lacks output_root")
    output_root = _require_under(Path(output_root_raw), data_attempt, label="structural detection output_root")
    if not output_root.is_dir():
        raise ValueError("structural detection output_root is missing")
    span_ledger = _verify_binding(payload.get("detected_span_ledger"), label="structural detection span ledger")
    _require_under(Path(span_ledger["path"]), data_attempt, label="structural detection span ledger")
    return payload, binding, span_ledger


def _validate_audit(
    path: Path,
    *,
    contract: Mapping[str, Any],
    detection_binding: Mapping[str, Any],
    metadata_attempt: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    binding = _binding(path, label="structural safety-audit manifest")
    payload = _read_object(path, label="structural safety-audit manifest")
    required = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "child_run_id": contract["run_id"],
        "python_rust_probability_parity_passed": True,
        "python_rust_decoded_span_parity_passed": True,
        "source_balanced_safety_metrics_passed": True,
        "false_deletion_audit_passed": True,
        "independent_source_balanced_safety_metrics_passed": True,
        "targeted_false_deletion_audit_cases": 100,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"structural safety-audit manifest drift: {key}")
    _require_exact_binding(payload.get("detection_manifest"), detection_binding, label="structural audit detection manifest")
    _require_exact_binding(
        payload.get("agent2_handoff"), contract["agent2_handoff"], label="structural audit Agent 2 handoff"
    )
    _require_exact_binding(
        payload.get("application_policy"), contract["application_policy"], label="structural audit application policy"
    )
    if payload.get("source_profile_allowlist") != contract["source_profile_allowlist"]:
        raise ValueError("structural safety-audit source-profile allowlist drift")
    manual_receipt = _verify_binding(payload.get("manual_false_deletion_audit_receipt"), label="manual false-deletion audit receipt")
    _require_under(Path(manual_receipt["path"]), metadata_attempt, label="manual false-deletion audit receipt")
    return payload, binding, manual_receipt


def _validate_duplicate_report(
    path: Path, *, final_root: Path, data_attempt: Path
) -> dict[str, Any]:
    binding = _binding(path, label="post-structural duplicate verification report")
    payload = _read_object(path, label="post-structural duplicate verification report")
    if payload.get("verification_only") is not True:
        raise ValueError("post-structural duplicate report is not verification-only")
    if payload.get("dedup_applied") is not False:
        raise ValueError("post-structural duplicate report must not apply a second dedup pass")
    _require_zero_count(
        payload.get("material_new_duplicate_count", payload.get("new_duplicate_count")),
        label="post-structural material duplicate count",
    )
    if Path(str(payload.get("input_root", ""))).resolve() != final_root:
        raise ValueError("post-structural duplicate report input root drift")
    _require_under(Path(binding["path"]), data_attempt, label="post-structural duplicate report")
    return binding


def _validate_apply(
    path: Path,
    *,
    contract: Mapping[str, Any],
    audit_binding: Mapping[str, Any],
    data_attempt: Path,
    structural_ledger: Path,
    span_ledger: Path,
    nonallowlisted_noop_ledger: Path,
    duplicate_report: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    binding = _binding(path, label="structural application manifest")
    payload = _read_object(path, label="structural application manifest")
    required = {
        "schema_version": APPLY_SCHEMA,
        "status": "passed",
        "mode": "applied",
        "child_run_id": contract["run_id"],
        "input_root": contract["parent"]["corpus_root"],
        "ready_for_application": True,
        "python_rust_probability_parity_passed": True,
        "python_rust_decoded_span_parity_passed": True,
        "source_balanced_safety_metrics_passed": True,
        "false_deletion_audit_passed": True,
        "non_allowlisted_mode": "explicit_no_op",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"structural application manifest drift: {key}")
    for name, expected in (
        ("parent_prestructural_manifest", contract["parent"]["prestructural_manifest"]),
        ("agent2_handoff", contract["agent2_handoff"]),
        ("application_policy", contract["application_policy"]),
        ("audit_manifest", audit_binding),
        ("structural_ledger", _binding(structural_ledger, label="structural action ledger")),
        ("structural_span_ledger", _binding(span_ledger, label="structural span ledger")),
        (
            "nonallowlisted_noop_ledger",
            _binding(nonallowlisted_noop_ledger, label="non-allowlisted no-op ledger"),
        ),
    ):
        _require_exact_binding(payload.get(name), expected, label=f"structural application {name}")
    if payload.get("source_profile_allowlist") != contract["source_profile_allowlist"]:
        raise ValueError("structural application source-profile allowlist drift")
    final_root_raw = payload.get("output_root")
    if not isinstance(final_root_raw, str) or not final_root_raw:
        raise ValueError("structural application manifest lacks output_root")
    final_root = _require_under(Path(final_root_raw), data_attempt, label="structural application output_root")
    if not final_root.is_dir() or final_root == Path(contract["parent"]["corpus_root"]):
        raise ValueError("structural application output_root must be a new child data tree")
    report_binding = _validate_duplicate_report(duplicate_report, final_root=final_root, data_attempt=data_attempt)
    _require_exact_binding(
        payload.get("poststructural_duplicate_report"), report_binding, label="structural application duplicate report"
    )
    return payload, binding, final_root


def _compact_binding(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a compact binding object")
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"{label} has invalid bytes")
    return {"bytes": size, "sha256": _require_sha256(value.get("sha256"), label=f"{label} SHA-256")}


def _contains_not_included(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = "".join(character for character in str(key).casefold() if character.isalnum())
            if normalized == "notincluded" or _contains_not_included(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_not_included(item) for item in value)
    return False


def _validate_required_release_evidence(
    release: Mapping[str, Any], handoff: Mapping[str, Any]
) -> None:
    """Check Phase-10 evidence survived from private release to Agent 3.

    The release materializer has already opened and semantically validated the
    source artifacts.  The structural child independently verifies the
    persisted, path-free closure here so a manually forged Stage-80 manifest
    cannot bypass review/quality/admission/execution/waterfall requirements.
    """

    release_evidence = release.get("required_evidence")
    handoff_evidence = handoff.get("required_evidence")
    if not isinstance(release_evidence, Mapping) or not isinstance(handoff_evidence, Mapping):
        raise ValueError("local release or Agent 3 handoff lacks required Phase-10 evidence closure")
    if _contains_not_included(release_evidence) or _contains_not_included(handoff_evidence):
        raise ValueError("local release or Agent 3 handoff contains not_included evidence")
    for label, evidence in (("local release", release_evidence), ("Agent 3 handoff", handoff_evidence)):
        if evidence.get("status") != "passed":
            raise ValueError(f"{label} required evidence closure is not passed")
        if evidence.get("required_categories") != list(REQUIRED_RELEASE_EVIDENCE_NAMES):
            raise ValueError(f"{label} required evidence category inventory drift")
        bindings = evidence.get("source_bindings")
        if not isinstance(bindings, Mapping) or set(bindings) != set(REQUIRED_RELEASE_EVIDENCE_NAMES):
            raise ValueError(f"{label} required evidence bindings drift")
        normalized = {
            name: _compact_binding(bindings[name], label=f"{label} evidence {name}")
            for name in REQUIRED_RELEASE_EVIDENCE_NAMES
        }
        clearance = evidence.get("anonymization_semantic_false_positive_clearance")
        if not isinstance(clearance, Mapping) or clearance.get("status") != "cleared":
            raise ValueError(f"{label} anonymization semantic false-positive clearance is incomplete")
        if clearance.get("sha256") != normalized["anonymization_semantic_clearance"]["sha256"]:
            raise ValueError(f"{label} semantic clearance hash drift")
        if clearance.get("waterfall_sha256") != normalized["transformation_waterfall"]["sha256"]:
            raise ValueError(f"{label} semantic clearance waterfall binding drift")
    release_bindings = release_evidence["source_bindings"]
    handoff_bindings = handoff_evidence["source_bindings"]
    if release_bindings != handoff_bindings:
        raise ValueError("Agent 3 handoff evidence bindings differ from the private release")
    if (
        release_evidence["anonymization_semantic_false_positive_clearance"]
        != handoff_evidence["anonymization_semantic_false_positive_clearance"]
    ):
        raise ValueError("Agent 3 handoff semantic-clearance closure differs from the private release")
    compact = handoff_evidence.get("compact_files")
    if not isinstance(compact, Mapping) or set(compact) != set(REQUIRED_RELEASE_EVIDENCE_NAMES):
        raise ValueError("Agent 3 handoff lacks the complete compact evidence inventory")
    for name in REQUIRED_RELEASE_EVIDENCE_NAMES:
        if _compact_binding(compact[name], label=f"Agent 3 compact evidence {name}") != _compact_binding(
            release_bindings[name], label=f"local release evidence {name}"
        ):
            raise ValueError("Agent 3 compact evidence receipt differs from the private release")
    upstream = release.get("upstream_bindings")
    if not isinstance(upstream, Mapping):
        raise ValueError("local private release lacks upstream evidence bindings")
    for name in REQUIRED_RELEASE_EVIDENCE_NAMES:
        expected = _compact_binding(release_bindings[name], label=f"local release evidence {name}")
        actual = upstream.get(f"evidence_{name}")
        if not isinstance(actual, Mapping) or _compact_binding(
            actual, label=f"local release upstream evidence {name}"
        ) != expected:
            raise ValueError("local private release upstream evidence binding drift")


def _validate_final(
    path: Path,
    *,
    contract: Mapping[str, Any],
    apply_binding: Mapping[str, Any],
    release_manifest: Path,
    release_validation: Path,
    site_handoff: Path,
    metadata_attempt: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _binding(path, label="final structural child validation manifest")
    payload = _read_object(path, label="final structural child validation manifest")
    release_binding = _binding(release_manifest, label="local private release manifest")
    validation_binding = _binding(release_validation, label="local release validation")
    handoff_binding = _binding(site_handoff, label="Agent 3 compact handoff")
    required = {
        "schema_version": FINAL_VALIDATION_SCHEMA,
        "status": "passed",
        "child_run_id": contract["run_id"],
        "local_release_only": True,
        "publish_permitted": False,
        "no_hf_publish_without_user_confirmation": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"final structural child validation manifest drift: {key}")
    _require_exact_binding(payload.get("structural_apply_manifest"), apply_binding, label="final validation apply manifest")
    _require_exact_binding(payload.get("release_manifest"), release_binding, label="final validation release manifest")
    _require_exact_binding(payload.get("release_validation"), validation_binding, label="final validation release validation")
    _require_exact_binding(payload.get("site_handoff"), handoff_binding, label="final validation site handoff")

    release = _read_object(release_manifest, label="local private release manifest")
    if release.get("schema_version") != RELEASE_SCHEMA or release.get("status") != "passed":
        raise ValueError("local private release manifest is not passed Agent 1 v3 output")
    if release.get("publish_permitted") is not False or release.get("release_kind") != "local_private_training_release":
        raise ValueError("local release manifest must remain private and non-publishable")
    ordered = release.get("ordered_transform_contract")
    if not isinstance(ordered, Mapping) or any(
        ordered.get(key) is not True
        for key in ("dedup_before_greekmmlu", "greekmmlu_before_anonymization", "anonymization_before_structural")
    ):
        raise ValueError("local release manifest violates ordered transformation closure")
    if ordered.get("structural_mode") != "applied":
        raise ValueError("local release manifest does not bind an applied structural child output")

    validation = _read_object(release_validation, label="local release validation")
    if (
        validation.get("schema_version") != RELEASE_VALIDATION_SCHEMA
        or validation.get("status") != "passed"
        or validation.get("failed_checks") != []
        or validation.get("release_contract_sha256") != release.get("release_contract_sha256")
    ):
        raise ValueError("local release validation does not close the private release")
    handoff = _read_object(site_handoff, label="Agent 3 compact handoff")
    if (
        handoff.get("schema_version") != SITE_HANDOFF_SCHEMA
        or handoff.get("status") != "passed"
        or handoff.get("release_is_public_dataset") is not False
        or handoff.get("release_contract_sha256") != release.get("release_contract_sha256")
    ):
        raise ValueError("Agent 3 handoff does not bind the private local release")
    _validate_required_release_evidence(release, handoff)
    _require_under(Path(binding["path"]), metadata_attempt, label="final structural child validation manifest")
    return payload, binding


def cmd_init(args: argparse.Namespace) -> None:
    run_id = _require_run_id(args.run_id)
    run_root, data_root = _validate_storage(args.run_root, args.data_root)
    if run_root.exists() or data_root.exists():
        raise FileExistsError("structural child metadata/data roots must be new immutable paths")
    parent_payload, parent_binding, parent_root = _parent_manifest(
        args.prestructural_manifest, args.prestructural_manifest_sha256, child_run_id=run_id
    )
    _, handoff_binding, artifact_hashes = _agent2_handoff(args.agent2_handoff, args.agent2_handoff_sha256)
    _, base_binding, base_allowlist = _base_policy(args.base_structural_policy)
    _, application_binding, materialization, profiles = _application_policy(
        args.application_policy, base_binding=base_binding, allowed_profiles=base_allowlist
    )
    payload: dict[str, Any] = {
        "schema_version": CHILD_RUN_SCHEMA,
        "status": "ready",
        "run_id": run_id,
        "run_root": str(run_root),
        "data_root": str(data_root),
        "stage_graph": list(STAGES),
        "parent": {
            "prestructural_manifest": parent_binding,
            "source_run_id": parent_payload["run_id"],
            "corpus_root": str(parent_root),
        },
        "agent2_handoff": handoff_binding,
        "agent2_artifact_hashes": artifact_hashes,
        "base_structural_policy": base_binding,
        "application_policy": application_binding,
        "materialization": materialization,
        "source_profile_allowlist": profiles,
        "publish_permitted": False,
        "created_at": _utc_now(),
    }
    payload["contract_sha256"] = _digest(payload, field="contract_sha256")
    run_root.mkdir(parents=True, exist_ok=False)
    try:
        _atomic_json_no_replace(_child_contract_path(run_root), payload)
        gate = _stage_gate(payload)
        gate["gate_sha256"] = _digest(gate, field="gate_sha256")
        _atomic_json_no_replace(run_root / "structural_child_gate.json", gate)
    except BaseException:
        # Do not remove a partially initialized root: immutable artifacts are
        # forensic evidence and must be inspected rather than silently erased.
        raise
    print(
        json.dumps(
            {
                "ok": True,
                "child_contract": str(_child_contract_path(run_root)),
                "gate": str(run_root / "structural_child_gate.json"),
                "publish_permitted": False,
            },
            sort_keys=True,
        )
    )


def cmd_status(args: argparse.Namespace) -> None:
    contract = _load_child_contract(args.run_root)
    completed, in_progress, next_stage = _progress(contract)
    print(
        json.dumps(
            {
                "ok": True,
                "run_id": contract["run_id"],
                "completed_stages": completed,
                "in_progress_stage": in_progress,
                "next_stage": next_stage,
                "publish_permitted": False,
            },
            sort_keys=True,
        )
    )


def cmd_begin(args: argparse.Namespace) -> None:
    contract = _load_child_contract(args.run_root)
    stage = args.stage
    attempt = _require_attempt_id(args.attempt_id)
    _, in_progress = _assert_next(contract, stage, action="begin")
    if in_progress is not None:
        raise ValueError(f"cannot begin {stage}; its existing child attempt must be inspected")
    stage_root = _stage_root(Path(contract["run_root"]), stage)
    data_stage_root = Path(contract["data_root"]) / "stages" / stage
    if stage_root.exists() or data_stage_root.exists():
        raise FileExistsError(f"{stage} child stage roots already exist")
    metadata, bulk = _attempt_roots(contract, stage, attempt)
    stage_root.mkdir(parents=True, exist_ok=False)
    metadata.mkdir(parents=True, exist_ok=False)
    try:
        inputs = _stage_expected_inputs(contract, stage)
        payload: dict[str, Any] = {
            "schema_version": CHILD_STAGE_SCHEMA,
            "run_id": contract["run_id"],
            "stage": stage,
            "child_contract_sha256": contract["contract_sha256"],
            "attempt_id": attempt,
            "upstream_stages": [] if stage == STAGES[0] else [STAGES[STAGES.index(stage) - 1]],
            "inputs": inputs,
            "storage": {
                "metadata_attempt_dir": str(metadata.resolve()),
                "data_attempt_dir": str(bulk.resolve()),
            },
            "created_at": _utc_now(),
        }
        payload["contract_sha256"] = _digest(payload, field="contract_sha256")
        _atomic_json_no_replace(_stage_contract_path(Path(contract["run_root"]), stage), payload)
        bulk.mkdir(parents=True, exist_ok=False)
    except BaseException:
        raise
    print(
        json.dumps(
            {
                "ok": True,
                "stage": stage,
                "metadata_attempt_dir": str(metadata),
                "data_attempt_dir": str(bulk),
                "publish_permitted": False,
            },
            sort_keys=True,
        )
    )


def _finish_stage(contract: Mapping[str, Any], stage: str, outputs: Iterable[Path]) -> None:
    _, in_progress = _assert_next(contract, stage, action="finish")
    if in_progress != stage:
        raise ValueError(f"cannot finish {stage}; it has not been begun")
    stage_contract = _validate_stage_contract(contract, stage)
    metadata, bulk = _attempt_roots(contract, stage, stage_contract["attempt_id"])
    output_bindings: list[dict[str, Any]] = []
    observed: set[Path] = set()
    for output in outputs:
        binding = _binding(output, label=f"{stage} output")
        path = Path(binding["path"])
        if path in observed:
            raise ValueError(f"duplicate child-stage output: {path}")
        observed.add(path)
        if not _is_relative_to(path, metadata) and not _is_relative_to(path, bulk):
            raise ValueError(f"{stage} output escapes its job-unique child attempt directory: {path}")
        output_bindings.append(binding)
    if not output_bindings:
        raise ValueError("child stage finish requires outputs")
    receipt: dict[str, Any] = {
        "schema_version": CHILD_RECEIPT_SCHEMA,
        "status": "passed",
        "run_id": contract["run_id"],
        "stage": stage,
        "child_contract_sha256": contract["contract_sha256"],
        "stage_contract_sha256": stage_contract["contract_sha256"],
        "attempt_id": stage_contract["attempt_id"],
        "outputs": output_bindings,
        "completed_at": _utc_now(),
    }
    receipt["receipt_sha256"] = _digest(receipt, field="receipt_sha256")
    receipt_path = _stage_receipt_path(Path(contract["run_root"]), stage)
    _atomic_json_no_replace(receipt_path, receipt)
    marker = _stage_root(Path(contract["run_root"]), stage) / "COMPLETED"
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{_sha256_file(receipt_path)}  stage_receipt.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"ok": True, "stage": stage, "receipt": str(receipt_path)}, sort_keys=True))


def cmd_finish_audit(args: argparse.Namespace) -> None:
    contract = _load_child_contract(args.run_root)
    stage = STAGES[0]
    _, in_progress = _assert_next(contract, stage, action="finish")
    if in_progress != stage:
        raise ValueError("Stage 75 has not been begun")
    stage_contract = _validate_stage_contract(contract, stage)
    metadata, bulk = _attempt_roots(contract, stage, stage_contract["attempt_id"])
    _, detection_binding, span_ledger = _validate_detection(
        args.detection_manifest, contract=contract, data_attempt=bulk
    )
    _, audit_binding, manual_receipt = _validate_audit(
        args.audit_manifest,
        contract=contract,
        detection_binding=detection_binding,
        metadata_attempt=metadata,
    )
    _finish_stage(contract, stage, [args.detection_manifest, args.audit_manifest, Path(span_ledger["path"]), Path(manual_receipt["path"])])


def cmd_finish_apply(args: argparse.Namespace) -> None:
    contract = _load_child_contract(args.run_root)
    stage = STAGES[1]
    _, in_progress = _assert_next(contract, stage, action="finish")
    if in_progress != stage:
        raise ValueError("Stage 78 has not been begun")
    stage_contract = _validate_stage_contract(contract, stage)
    _, bulk = _attempt_roots(contract, stage, stage_contract["attempt_id"])
    audit_receipt = _load_stage_receipt(contract, STAGES[0])
    audit_candidates = [item for item in audit_receipt["outputs"] if Path(item["path"]).name == args.audit_manifest.name]
    if len(audit_candidates) != 1:
        raise ValueError("Stage 75 receipt does not bind the supplied safety-audit manifest")
    audit_binding = _binding(args.audit_manifest, label="structural safety-audit manifest")
    if audit_binding != audit_candidates[0]:
        raise ValueError("supplied safety-audit manifest drifted from Stage 75 receipt")
    _, apply_binding, _ = _validate_apply(
        args.apply_manifest,
        contract=contract,
        audit_binding=audit_binding,
        data_attempt=bulk,
        structural_ledger=args.structural_ledger,
        span_ledger=args.structural_span_ledger,
        nonallowlisted_noop_ledger=args.nonallowlisted_noop_ledger,
        duplicate_report=args.poststructural_duplicate_report,
    )
    _finish_stage(
        contract,
        stage,
        [
            args.apply_manifest,
            args.structural_ledger,
            args.structural_span_ledger,
            args.nonallowlisted_noop_ledger,
            args.poststructural_duplicate_report,
        ],
    )


def cmd_finish_final(args: argparse.Namespace) -> None:
    contract = _load_child_contract(args.run_root)
    stage = STAGES[2]
    _, in_progress = _assert_next(contract, stage, action="finish")
    if in_progress != stage:
        raise ValueError("Stage 80 has not been begun")
    stage_contract = _validate_stage_contract(contract, stage)
    metadata, _ = _attempt_roots(contract, stage, stage_contract["attempt_id"])
    apply_receipt = _load_stage_receipt(contract, STAGES[1])
    candidates = [item for item in apply_receipt["outputs"] if Path(item["path"]).name == args.apply_manifest.name]
    if len(candidates) != 1:
        raise ValueError("Stage 78 receipt does not bind the supplied structural application manifest")
    apply_binding = _binding(args.apply_manifest, label="structural application manifest")
    if apply_binding != candidates[0]:
        raise ValueError("supplied structural application manifest drifted from Stage 78 receipt")
    _validate_final(
        args.final_validation_manifest,
        contract=contract,
        apply_binding=apply_binding,
        release_manifest=args.release_manifest,
        release_validation=args.release_validation,
        site_handoff=args.site_handoff,
        metadata_attempt=metadata,
    )
    _finish_stage(
        contract,
        stage,
        [
            args.final_validation_manifest,
            args.release_manifest,
            args.release_validation,
            args.site_handoff,
        ],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a new, hash-pinned structural child contract")
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--data-root", type=Path, required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--prestructural-manifest", type=Path, required=True)
    init.add_argument("--prestructural-manifest-sha256", required=True)
    init.add_argument("--agent2-handoff", type=Path, required=True)
    init.add_argument("--agent2-handoff-sha256", required=True)
    init.add_argument("--base-structural-policy", type=Path, required=True)
    init.add_argument("--application-policy", type=Path, required=True)
    init.set_defaults(func=cmd_init)

    status = commands.add_parser("status", help="verify child-contract and receipt closure")
    status.add_argument("--run-root", type=Path, required=True)
    status.set_defaults(func=cmd_status)

    begin = commands.add_parser("begin", help="create the sole next child-stage attempt")
    begin.add_argument("--run-root", type=Path, required=True)
    begin.add_argument("--stage", choices=STAGES, required=True)
    begin.add_argument("--attempt-id", required=True)
    begin.set_defaults(func=cmd_begin)

    audit = commands.add_parser("finish-audit", help="validate and receipt Stage 75 artifacts")
    audit.add_argument("--run-root", type=Path, required=True)
    audit.add_argument("--detection-manifest", type=Path, required=True)
    audit.add_argument("--audit-manifest", type=Path, required=True)
    audit.set_defaults(func=cmd_finish_audit)

    apply = commands.add_parser("finish-apply", help="validate and receipt Stage 78 artifacts")
    apply.add_argument("--run-root", type=Path, required=True)
    apply.add_argument("--audit-manifest", type=Path, required=True)
    apply.add_argument("--apply-manifest", type=Path, required=True)
    apply.add_argument("--structural-ledger", type=Path, required=True)
    apply.add_argument("--structural-span-ledger", type=Path, required=True)
    apply.add_argument("--nonallowlisted-noop-ledger", type=Path, required=True)
    apply.add_argument("--poststructural-duplicate-report", type=Path, required=True)
    apply.set_defaults(func=cmd_finish_apply)

    final = commands.add_parser("finish-final", help="validate and receipt Stage 80 local-release closure")
    final.add_argument("--run-root", type=Path, required=True)
    final.add_argument("--apply-manifest", type=Path, required=True)
    final.add_argument("--final-validation-manifest", type=Path, required=True)
    final.add_argument("--release-manifest", type=Path, required=True)
    final.add_argument("--release-validation", type=Path, required=True)
    final.add_argument("--site-handoff", type=Path, required=True)
    final.set_defaults(func=cmd_finish_final)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

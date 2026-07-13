#!/usr/bin/env python3
"""Dry-run-safe orchestration for Agent 1's immutable full-corpus v3 stages.

This is deliberately a *contract orchestrator*, not a data-processing runner.
It establishes the only permitted stage order, validates the frozen run plus
all completed receipts, and delegates writes to ``agent1_v3_contract.py``.
Therefore an ordinary invocation is read-only; ``--execute`` is required to
create a stage attempt or publish a stage receipt.

The ordered graph is fixed here as a guard against accidental reuse of the v2
ordering:

``10 -> 20 -> 30 -> 35 -> 40 -> 50 -> 55 -> 60 -> 65 -> 70 -> 75 -> 78 -> 80``

Stage 78 is additionally restricted to a child run importing an immutable
prestructural manifest and a hash-pinned, passed Agent 2 handoff.  Stage 75 is
allowed to be an explicit no-op audit; no missing handoff may be treated as
permission to apply structural removal.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import agent1_v3_contract as contract


PIPELINE_PLAN_SCHEMA = "agent1_full_corpus_v3_pipeline_plan_v1"
PIPELINE_STATUS_SCHEMA = "agent1_full_corpus_v3_pipeline_status_v1"
PRESTRUCTURAL_SCHEMA = "agent1_full_corpus_v3_prestructural_manifest_v1"

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
ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STRUCTURAL_APPLY_STAGE = "78-structural-apply"
AGENT2_REQUIRED_GATES = {
    "ready_for_corpus_application": True,
    "python_rust_probability_parity_passed": True,
    "python_rust_decoded_span_parity_passed": True,
    "source_balanced_safety_metrics_passed": True,
    "false_deletion_audit_passed": True,
}


def _ensure_contract_stage_order() -> None:
    """Fail closed if the contract script's graph ever drifts from v3 order."""

    if tuple(contract.STAGES) != ORDERED_STAGES:
        raise RuntimeError(
            "agent1_v3_contract.STAGES drifted from the fixed v3 order; "
            "create a new lane rather than silently reordering stages"
        )
    if tuple(contract.PRESTRUCTURAL_STAGES) != ORDERED_STAGES[:10]:
        raise RuntimeError("prestructural stage boundary drifted from the fixed v3 order")


_ensure_contract_stage_order()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: str | None) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_attempt_id(value: str) -> str:
    if not ATTEMPT_ID_RE.fullmatch(value):
        raise ValueError("attempt id must use only letters, digits, '.', '_' or '-'")
    return value


def _parse_parameters(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--parameters-json is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--parameters-json must be a JSON object")
    return value


def _active_graph(run_contract: Mapping[str, Any]) -> tuple[str, ...]:
    return contract.PRESTRUCTURAL_STAGES if bool(run_contract["prestructural_only"]) else contract.STAGES


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return contract.read_object(path)


def _binding_with_expected_hash(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    _require_sha256(f"{label} SHA-256", expected_sha256)
    binding = contract.file_binding(path)
    if binding["sha256"] != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, actual {binding['sha256']}"
        )
    return binding


def _verify_completed_marker(run_root: Path, stage: str) -> None:
    """Require the immutable marker that the contract writes after its receipt."""

    receipt_path = contract.stage_receipt_path(run_root, stage)
    marker = contract.stage_root(run_root, stage) / "COMPLETED"
    if not marker.is_file():
        raise ValueError(f"{stage}: passed receipt lacks immutable COMPLETED marker")
    expected = f"{contract.sha256_file(receipt_path)}  stage_receipt.json\n"
    if marker.read_text(encoding="utf-8") != expected:
        raise ValueError(f"{stage}: COMPLETED marker hash drift")


def _verify_stage_contract(
    run_root: Path, stage: str, run_contract: Mapping[str, Any]
) -> dict[str, Any]:
    path = contract.stage_root(run_root, stage) / "stage_contract.json"
    stage_contract = _read_object(path, f"{stage} stage contract")
    expected_upstream = list(
        contract.expected_upstream(stage, prestructural_only=bool(run_contract["prestructural_only"]))
    )
    expected = {
        "schema_version": contract.STAGE_SCHEMA,
        "run_id": run_contract["run_id"],
        "stage": stage,
        "code_commit": run_contract["code_commit"],
        "run_contract_sha256": run_contract["contract_sha256"],
        "upstream_stages": expected_upstream,
    }
    for key, value in expected.items():
        if stage_contract.get(key) != value:
            raise ValueError(f"{path}: {key} drift")
    if stage_contract.get("contract_sha256") != contract.contract_digest(stage_contract):
        raise ValueError(f"{path}: stage contract hash mismatch")
    inputs = stage_contract.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"{path}: stage inputs must be an object")
    for name, binding in inputs.items():
        contract.verify_binding(f"{stage}:input:{name}", binding)
    return stage_contract


def _verify_stage_receipt(
    run_root: Path, stage: str, run_contract: Mapping[str, Any]
) -> dict[str, Any]:
    stage_contract = _verify_stage_contract(run_root, stage, run_contract)
    receipt = contract.load_stage_receipt(run_root, stage, dict(run_contract))
    if receipt.get("stage_contract_sha256") != stage_contract.get("contract_sha256"):
        raise ValueError(f"{stage}: receipt does not bind its stage contract")
    receipt_for_digest = dict(receipt)
    receipt_for_digest.pop("receipt_sha256", None)
    if receipt.get("receipt_sha256") != contract.contract_digest(receipt_for_digest):
        raise ValueError(f"{stage}: stage receipt hash mismatch")
    _verify_completed_marker(run_root, stage)
    return receipt


@dataclass(frozen=True)
class PipelineProgress:
    graph: tuple[str, ...]
    completed: tuple[str, ...]
    in_progress: str | None
    next_stage: str | None
    receipts: Mapping[str, Mapping[str, Any]]


def inspect_progress(run_root: Path, run_contract: Mapping[str, Any]) -> PipelineProgress:
    """Verify chain closure and find the sole stage eligible to advance."""

    graph = _active_graph(run_contract)
    completed: list[str] = []
    receipts: dict[str, Mapping[str, Any]] = {}
    in_progress: str | None = None
    seen_gap = False
    for stage in graph:
        directory = contract.stage_root(run_root, stage)
        receipt_path = contract.stage_receipt_path(run_root, stage)
        if receipt_path.exists():
            if seen_gap:
                raise ValueError(f"{stage}: completed receipt appears after an incomplete prior stage")
            receipt = _verify_stage_receipt(run_root, stage, run_contract)
            # The stage contract itself must declare exactly the prior receipts.
            stage_contract = _read_object(directory / "stage_contract.json", f"{stage} stage contract")
            expected_upstream = list(
                contract.expected_upstream(stage, prestructural_only=bool(run_contract["prestructural_only"]))
            )
            if stage_contract.get("upstream_stages") != expected_upstream:
                raise ValueError(f"{stage}: upstream stage declaration drift")
            for upstream in expected_upstream:
                if upstream not in receipts:
                    raise ValueError(f"{stage}: required upstream receipt was not completed first")
            completed.append(stage)
            receipts[stage] = receipt
            continue
        seen_gap = True
        if directory.exists():
            if in_progress is not None:
                raise ValueError(f"multiple incomplete stage directories: {in_progress}, {stage}")
            _verify_stage_contract(run_root, stage, run_contract)
            in_progress = stage
    next_stage = None
    for stage in graph:
        if stage not in completed:
            next_stage = stage
            break
    if in_progress is not None and next_stage != in_progress:
        raise ValueError("in-progress stage is not the first uncompleted ordered stage")
    return PipelineProgress(
        graph=tuple(graph),
        completed=tuple(completed),
        in_progress=in_progress,
        next_stage=next_stage,
        receipts=receipts,
    )


def _require_stage_is_next(progress: PipelineProgress, stage: str, *, action: str) -> None:
    if stage not in progress.graph:
        raise ValueError(f"{stage} is unavailable in this run mode")
    if progress.next_stage is None:
        raise ValueError("all stages in this immutable run are already completed")
    if stage != progress.next_stage:
        raise ValueError(
            f"cannot {action} {stage}; fixed v3 order requires {progress.next_stage} next"
        )


def _parse_input_pairs(values: Iterable[Sequence[str]]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if len(value) != 2:
            raise ValueError("each --input must have NAME PATH")
        name, raw_path = value
        if not name or name in names:
            raise ValueError(f"duplicate/empty stage input name: {name!r}")
        names.add(name)
        path = Path(raw_path)
        # Hash now, even for dry run, to ensure a proposed stage only names
        # immutable existing inputs.
        contract.file_binding(path)
        result.append((name, path))
    return result


def _prestructural_manifest_binding(
    path: Path, expected_sha256: str, current_run_id: str
) -> dict[str, Any]:
    binding = _binding_with_expected_hash(path, expected_sha256, "prestructural manifest")
    manifest = _read_object(path, "prestructural manifest")
    required = {
        "schema_version": PRESTRUCTURAL_SCHEMA,
        "status": "prestructural_frozen",
        "publish_permitted": False,
        "structural_state": "awaiting_agent2_handoff",
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"prestructural manifest does not satisfy child-run gate: {key}")
    source_run_id = manifest.get("run_id")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("prestructural manifest lacks source run_id")
    if source_run_id == current_run_id:
        raise ValueError("structural apply must import a prestructural manifest from a child-run parent")
    return binding


def _agent2_handoff_binding(path: Path, expected_sha256: str) -> dict[str, Any]:
    binding = _binding_with_expected_hash(path, expected_sha256, "Agent 2 immutable handoff")
    handoff = _read_object(path, "Agent 2 immutable handoff")
    for key, expected in AGENT2_REQUIRED_GATES.items():
        if handoff.get(key) is not expected:
            raise ValueError(f"Agent 2 handoff does not pass required gate: {key}")
    return binding


def validate_structural_apply_inputs(
    *,
    run_contract: Mapping[str, Any],
    agent2_handoff: Path | None,
    agent2_handoff_sha256: str | None,
    prestructural_manifest: Path | None,
    prestructural_manifest_sha256: str | None,
) -> list[tuple[str, Path]]:
    """Validate and name the mandatory Stage-78 imported immutable inputs."""

    if agent2_handoff is None or agent2_handoff_sha256 is None:
        raise ValueError("structural apply is blocked: immutable Agent 2 handoff and SHA-256 are required")
    if prestructural_manifest is None or prestructural_manifest_sha256 is None:
        raise ValueError(
            "structural apply is blocked: imported child-run prestructural manifest and SHA-256 are required"
        )
    _agent2_handoff_binding(agent2_handoff, agent2_handoff_sha256)
    _prestructural_manifest_binding(
        prestructural_manifest, prestructural_manifest_sha256, str(run_contract["run_id"])
    )
    return [
        ("agent2_immutable_handoff", agent2_handoff),
        ("imported_prestructural_manifest", prestructural_manifest),
    ]


def _stage_plan(
    *,
    run_root: Path,
    run_contract: Mapping[str, Any],
    progress: PipelineProgress,
    stage: str,
    action: str,
    attempt_id: str | None = None,
    inputs: Sequence[tuple[str, Path]] = (),
    parameters: Mapping[str, Any] | None = None,
    structural_inputs: Sequence[tuple[str, Path]] = (),
) -> dict[str, Any]:
    expected_upstream = list(
        contract.expected_upstream(stage, prestructural_only=bool(run_contract["prestructural_only"]))
    )
    all_inputs = [*inputs, *structural_inputs]
    bindings = {name: contract.file_binding(path) for name, path in all_inputs}
    if len(bindings) != len(all_inputs):
        raise ValueError("duplicate input names after stage-specific bindings")
    plan: dict[str, Any] = {
        "schema_version": PIPELINE_PLAN_SCHEMA,
        "action": action,
        "mode": "dry_run",
        "run_id": run_contract["run_id"],
        "run_root": str(run_root),
        "run_contract_sha256": run_contract["contract_sha256"],
        "stage": stage,
        "stage_index": progress.graph.index(stage),
        "ordered_stages": list(progress.graph),
        "expected_upstream": expected_upstream,
        "upstream_receipts": {
            upstream: progress.receipts[upstream].get("receipt_sha256") for upstream in expected_upstream
        },
        "completed_stages": list(progress.completed),
        "in_progress_stage": progress.in_progress,
        "next_stage": progress.next_stage,
        "inputs": bindings,
    }
    if attempt_id is not None:
        plan["attempt_id"] = attempt_id
        plan["attempt_directory"] = str(contract.stage_root(run_root, stage) / "attempts" / attempt_id)
    if parameters is not None:
        plan["parameters"] = dict(parameters)
        plan["parameters_sha256"] = _sha256_json(parameters)
    if stage == STRUCTURAL_APPLY_STAGE:
        plan["structural_apply_gate"] = {
            "agent2_handoff_bound": "agent2_immutable_handoff" in bindings,
            "imported_prestructural_manifest_bound": "imported_prestructural_manifest" in bindings,
            "child_run_required": True,
        }
    plan["plan_sha256"] = _sha256_json(plan)
    return plan


def plan_stage(
    *,
    run_root: Path,
    run_id: str,
    stage: str | None,
    agent2_handoff: Path | None = None,
    agent2_handoff_sha256: str | None = None,
    prestructural_manifest: Path | None = None,
    prestructural_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a read-only next-stage plan; structural absence is reported as blocked."""

    root = run_root.resolve()
    run_contract = contract.load_valid_contract(root, run_id=run_id)
    progress = inspect_progress(root, run_contract)
    target = stage or progress.next_stage
    if target is None:
        result = {
            "schema_version": PIPELINE_PLAN_SCHEMA,
            "action": "plan",
            "mode": "dry_run",
            "run_id": run_contract["run_id"],
            "run_contract_sha256": run_contract["contract_sha256"],
            "status": "complete",
            "completed_stages": list(progress.completed),
            "next_stage": None,
        }
        result["plan_sha256"] = _sha256_json(result)
        return result
    _require_stage_is_next(progress, target, action="plan")
    structural_inputs: list[tuple[str, Path]] = []
    blocked_reason: str | None = None
    if target == STRUCTURAL_APPLY_STAGE:
        try:
            structural_inputs = validate_structural_apply_inputs(
                run_contract=run_contract,
                agent2_handoff=agent2_handoff,
                agent2_handoff_sha256=agent2_handoff_sha256,
                prestructural_manifest=prestructural_manifest,
                prestructural_manifest_sha256=prestructural_manifest_sha256,
            )
        except (FileNotFoundError, ValueError) as exc:
            blocked_reason = str(exc)
    result = _stage_plan(
        run_root=root,
        run_contract=run_contract,
        progress=progress,
        stage=target,
        action="plan",
        structural_inputs=structural_inputs,
    )
    result["would_begin"] = blocked_reason is None
    if blocked_reason is not None:
        result["status"] = "blocked"
        result["blocked_reason"] = blocked_reason
    else:
        result["status"] = "ready"
    result["plan_sha256"] = _sha256_json({key: value for key, value in result.items() if key != "plan_sha256"})
    return result


def begin_stage(
    *,
    run_root: Path,
    run_id: str,
    stage: str,
    attempt_id: str,
    inputs: Iterable[Sequence[str]] = (),
    parameters_json: str = "{}",
    execute: bool = False,
    agent2_handoff: Path | None = None,
    agent2_handoff_sha256: str | None = None,
    prestructural_manifest: Path | None = None,
    prestructural_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Plan or explicitly create the sole next immutable stage attempt."""

    root = run_root.resolve()
    safe_attempt_id = _require_attempt_id(attempt_id)
    parameters = _parse_parameters(parameters_json)
    run_contract = contract.load_valid_contract(root, run_id=run_id)
    progress = inspect_progress(root, run_contract)
    _require_stage_is_next(progress, stage, action="begin")
    if contract.stage_root(root, stage).exists():
        raise FileExistsError(f"{stage}: stage directory already exists; it must be finished or inspected")
    ordinary_inputs = _parse_input_pairs(inputs)
    structural_inputs: list[tuple[str, Path]] = []
    if stage == STRUCTURAL_APPLY_STAGE:
        structural_inputs = validate_structural_apply_inputs(
            run_contract=run_contract,
            agent2_handoff=agent2_handoff,
            agent2_handoff_sha256=agent2_handoff_sha256,
            prestructural_manifest=prestructural_manifest,
            prestructural_manifest_sha256=prestructural_manifest_sha256,
        )
    elif any(
        value is not None
        for value in (
            agent2_handoff,
            agent2_handoff_sha256,
            prestructural_manifest,
            prestructural_manifest_sha256,
        )
    ):
        raise ValueError("Agent 2/prestructural arguments are only valid for 78-structural-apply")
    all_names = [name for name, _ in [*ordinary_inputs, *structural_inputs]]
    if len(all_names) != len(set(all_names)):
        raise ValueError("stage input name collides with a reserved structural binding")
    plan = _stage_plan(
        run_root=root,
        run_contract=run_contract,
        progress=progress,
        stage=stage,
        action="begin",
        attempt_id=safe_attempt_id,
        inputs=ordinary_inputs,
        parameters=parameters,
        structural_inputs=structural_inputs,
    )
    if not execute:
        return plan
    begin_args = argparse.Namespace(
        run_root=root,
        run_id=run_id,
        stage=stage,
        attempt_id=safe_attempt_id,
        input=[(name, str(path)) for name, path in [*ordinary_inputs, *structural_inputs]],
        parameters_json=_canonical_json(parameters),
    )
    # The contract remains the sole writer of stage_contract.json and the
    # attempt directory.  Suppress its CLI progress line so this CLI emits one
    # deterministic machine-readable result.
    with contextlib.redirect_stdout(io.StringIO()):
        contract.cmd_begin_stage(begin_args)
    plan["mode"] = "executed"
    plan["stage_contract_path"] = str(contract.stage_root(root, stage) / "stage_contract.json")
    plan["plan_sha256"] = _sha256_json({key: value for key, value in plan.items() if key != "plan_sha256"})
    return plan


def _verify_structural_finish_inputs(
    stage_contract: Mapping[str, Any], run_contract: Mapping[str, Any]
) -> None:
    inputs = stage_contract.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("structural stage contract lacks inputs")
    handoff_binding = inputs.get("agent2_immutable_handoff")
    prestructural_binding = inputs.get("imported_prestructural_manifest")
    if handoff_binding is None or prestructural_binding is None:
        raise ValueError("structural apply is blocked: stage contract lacks immutable Agent 2/prestructural inputs")
    contract.verify_binding("agent2_immutable_handoff", handoff_binding)
    contract.verify_binding("imported_prestructural_manifest", prestructural_binding)
    _agent2_handoff_binding(
        Path(str(handoff_binding["path"])), str(handoff_binding["sha256"])
    )
    _prestructural_manifest_binding(
        Path(str(prestructural_binding["path"])),
        str(prestructural_binding["sha256"]),
        str(run_contract["run_id"]),
    )


def finish_stage(
    *,
    run_root: Path,
    run_id: str,
    stage: str,
    attempt_id: str,
    outputs: Iterable[Path],
    execute: bool = False,
) -> dict[str, Any]:
    """Plan or explicitly publish a receipt for the sole in-progress stage."""

    root = run_root.resolve()
    safe_attempt_id = _require_attempt_id(attempt_id)
    run_contract = contract.load_valid_contract(root, run_id=run_id)
    progress = inspect_progress(root, run_contract)
    _require_stage_is_next(progress, stage, action="finish")
    if progress.in_progress != stage:
        raise ValueError(f"cannot finish {stage}: it has not been begun")
    stage_contract = _verify_stage_contract(root, stage, run_contract)
    if stage_contract.get("attempt_id") != safe_attempt_id:
        raise ValueError("attempt id differs from the immutable stage contract")
    if stage == STRUCTURAL_APPLY_STAGE:
        _verify_structural_finish_inputs(stage_contract, run_contract)
    attempt_root = (contract.stage_root(root, stage) / "attempts" / safe_attempt_id).resolve()
    output_bindings: list[dict[str, Any]] = []
    output_values = list(outputs)
    if not output_values:
        raise ValueError("finish requires at least one output")
    seen_outputs: set[Path] = set()
    for value in output_values:
        output = value.resolve()
        if output in seen_outputs:
            raise ValueError(f"duplicate stage output: {output}")
        seen_outputs.add(output)
        try:
            output.relative_to(attempt_root)
        except ValueError as exc:
            raise ValueError(f"stage output must remain in its job-unique attempt directory: {output}") from exc
        output_bindings.append(contract.file_binding(output))
    plan: dict[str, Any] = {
        "schema_version": PIPELINE_PLAN_SCHEMA,
        "action": "finish",
        "mode": "dry_run",
        "run_id": run_contract["run_id"],
        "run_root": str(root),
        "run_contract_sha256": run_contract["contract_sha256"],
        "stage": stage,
        "attempt_id": safe_attempt_id,
        "stage_contract_sha256": stage_contract["contract_sha256"],
        "outputs": output_bindings,
        "completed_stages": list(progress.completed),
        "in_progress_stage": progress.in_progress,
        "next_stage": progress.next_stage,
    }
    plan["plan_sha256"] = _sha256_json(plan)
    if not execute:
        return plan
    finish_args = argparse.Namespace(
        run_root=root,
        run_id=run_id,
        stage=stage,
        attempt_id=safe_attempt_id,
        output=[str(path) for path in output_values],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        contract.cmd_finish_stage(finish_args)
    receipt_path = contract.stage_receipt_path(root, stage)
    receipt = _verify_stage_receipt(root, stage, run_contract)
    plan["mode"] = "executed"
    plan["stage_receipt_path"] = str(receipt_path)
    plan["stage_receipt_sha256"] = receipt["receipt_sha256"]
    plan["plan_sha256"] = _sha256_json({key: value for key, value in plan.items() if key != "plan_sha256"})
    return plan


def pipeline_status(*, run_root: Path, run_id: str) -> dict[str, Any]:
    """Read-only receipt-chain status; any drift fails instead of being hidden."""

    root = run_root.resolve()
    run_contract = contract.load_valid_contract(root, run_id=run_id)
    progress = inspect_progress(root, run_contract)
    result: dict[str, Any] = {
        "schema_version": PIPELINE_STATUS_SCHEMA,
        "run_id": run_contract["run_id"],
        "run_root": str(root),
        "run_contract_sha256": run_contract["contract_sha256"],
        "prestructural_only": bool(run_contract["prestructural_only"]),
        "ordered_stages": list(progress.graph),
        "completed_stages": list(progress.completed),
        "completed_receipt_sha256": {
            stage: progress.receipts[stage].get("receipt_sha256") for stage in progress.completed
        },
        "in_progress_stage": progress.in_progress,
        "next_stage": progress.next_stage,
        "status": "complete" if progress.next_stage is None else "ready_or_in_progress",
    }
    result["status_sha256"] = _sha256_json(result)
    return result


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)


def _add_structural_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent2-handoff", type=Path)
    parser.add_argument("--agent2-handoff-sha256")
    parser.add_argument("--prestructural-manifest", type=Path)
    parser.add_argument("--prestructural-manifest-sha256")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="verify receipts and show the only next ordered stage")
    _add_common_run_args(status)
    status.set_defaults(func=_cmd_status)

    plan = sub.add_parser("plan", help="read-only plan for the next fixed-order stage")
    _add_common_run_args(plan)
    plan.add_argument("--stage", choices=ORDERED_STAGES)
    _add_structural_args(plan)
    plan.set_defaults(func=_cmd_plan)

    begin = sub.add_parser("begin", help="dry-run by default; use --execute to create a stage attempt")
    _add_common_run_args(begin)
    begin.add_argument("--stage", choices=ORDERED_STAGES, required=True)
    begin.add_argument("--attempt-id", required=True)
    begin.add_argument("--input", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    begin.add_argument("--parameters-json", default="{}")
    begin.add_argument("--execute", action="store_true")
    _add_structural_args(begin)
    begin.set_defaults(func=_cmd_begin)

    finish = sub.add_parser("finish", help="dry-run by default; use --execute to publish a receipt")
    _add_common_run_args(finish)
    finish.add_argument("--stage", choices=ORDERED_STAGES, required=True)
    finish.add_argument("--attempt-id", required=True)
    finish.add_argument("--output", type=Path, action="append", required=True)
    finish.add_argument("--execute", action="store_true")
    finish.set_defaults(func=_cmd_finish)
    return result


def _cmd_status(args: argparse.Namespace) -> int:
    print(json.dumps(pipeline_status(run_root=args.run_root, run_id=args.run_id), ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            plan_stage(
                run_root=args.run_root,
                run_id=args.run_id,
                stage=args.stage,
                agent2_handoff=args.agent2_handoff,
                agent2_handoff_sha256=args.agent2_handoff_sha256,
                prestructural_manifest=args.prestructural_manifest,
                prestructural_manifest_sha256=args.prestructural_manifest_sha256,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _cmd_begin(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            begin_stage(
                run_root=args.run_root,
                run_id=args.run_id,
                stage=args.stage,
                attempt_id=args.attempt_id,
                inputs=args.input,
                parameters_json=args.parameters_json,
                execute=bool(args.execute),
                agent2_handoff=args.agent2_handoff,
                agent2_handoff_sha256=args.agent2_handoff_sha256,
                prestructural_manifest=args.prestructural_manifest,
                prestructural_manifest_sha256=args.prestructural_manifest_sha256,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _cmd_finish(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            finish_stage(
                run_root=args.run_root,
                run_id=args.run_id,
                stage=args.stage,
                attempt_id=args.attempt_id,
                outputs=args.output,
                execute=bool(args.execute),
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    return int(arguments.func(arguments))


if __name__ == "__main__":
    raise SystemExit(main())

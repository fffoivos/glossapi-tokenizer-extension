#!/usr/bin/env python3
"""Command-line orchestration for controlled bibliography evolution."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bibliography_evolution_contract import (
    CandidateStore,
    ContractError,
    REGISTRY_SCHEMA,
    atomic_write_bytes_exact_resume,
    atomic_write_bytes_exclusive,
    build_registry,
    canonical_json_bytes,
    expand_template,
    load_json,
    paired_work_bootstrap,
    sha256_directory,
    sha256_file,
    verify_finalized_receipt,
    validate_candidate_spec,
    verify_parent_lineage,
    verify_g0,
    with_candidate_id,
    write_json_exclusive,
)


SEALED_MANIFEST_SCHEMA = "bibliography-evolution-frozen-pareto-manifest-v1"
SEALED_REQUEST_SCHEMA = "bibliography-evolution-sealed-batch-request-v1"
SEALED_FREEZE_SCHEMA = "bibliography-sealed-freeze-v1"
SEALED_RESULTS_SCHEMA = "bibliography-evolution-sealed-results-v1"
SEALED_RESULTS_RECEIPT_SCHEMA = "bibliography-evolution-sealed-results-receipt-v1"
SEALED_FRONTIER_LOCK_SCHEMA = "bibliography-evolution-sealed-frontier-lock-v1"


def _regular_file(path: Path | str, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ContractError(f"{label} is missing or a symlink")
    return raw.resolve()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

RUNNER_BY_COMPONENT = {
    "baseline.replay": "sequence_models.bibliography_evolution_g0_replay",
    "decoder.anchor_and_expansion_policy": "sequence_models.bibliography_evolution_core_decode",
    "headers.role_controller": "sequence_models.bibliography_evolution_postprocess",
    "decoder.fringe_trim": "sequence_models.bibliography_evolution_postprocess",
    "decoder.gap_connector": "sequence_models.bibliography_evolution_postprocess",
    "decoder.component_veto": "sequence_models.bibliography_evolution_postprocess",
    "decoder.outward_edge": "sequence_models.bibliography_evolution_postprocess",
    "decoder.weak_unseeded": "sequence_models.bibliography_evolution_postprocess",
    "signal.architecture": "sequence_models.bibliography_evolution_signal_pipeline",
    "signal.features": "sequence_models.bibliography_evolution_signal_pipeline",
    "signal.training": "sequence_models.bibliography_evolution_signal_pipeline",
    "composition.pairwise": "sequence_models.bibliography_evolution_composition",
}

RUNNER_FLAGS = {
    "sequence_models.bibliography_evolution_g0_replay": {
        "--lock", "--authoritative-root", "--validation-table-dir",
        "--validation-signal-probability", "--validation-line-probability",
        "--validation-scope-mask", "--qualified-documents", "--output-dir",
        "--code-commit", "--slurm-job-id",
    },
    "sequence_models.bibliography_evolution_core_decode": {
        "--table-dir", "--signal-probability", "--line-probability", "--scope-mask",
        "--qualified-documents", "--anchor-probability", "--anchors-required",
        "--anchor-window", "--maximum-bridge-gap", "--inside-probability",
        "--adjacent-expansion", "--header-window", "--output-dir", "--code-commit",
        "--slurm-job-id",
    },
    "sequence_models.bibliography_evolution_postprocess": {
        "--table-dir", "--baseline-prediction", "--signal-probability", "--scope-mask",
        "--barrier-artifact", "--header-roles", "--qualified-documents", "--operation",
        "--threshold", "--max-lines", "--output-dir", "--code-commit", "--slurm-job-id",
    },
    "sequence_models.bibliography_evolution_signal_pipeline": {
        "--input", "--train-table-dir", "--line-oof-dir", "--block-oof-dir",
        "--deterministic-roles-dir", "--train-quality-decisions", "--validation-table-dir",
        "--validation-line-probability", "--train-recall-block-dir",
        "--validation-quality-decisions", "--validation-policy", "--output-dir",
        "--hidden-dim", "--dilations", "--dropout", "--epochs", "--seed", "--workers",
        "--cpus", "--code-commit", "--slurm-job-id",
    },
    "sequence_models.bibliography_evolution_composition": {
        "--table-dir", "--left-prediction", "--right-prediction", "--qualified-documents",
        "--left-barrier-artifact", "--right-barrier-artifact", "--operation", "--output-dir",
        "--code-commit", "--slurm-job-id",
    },
}


def _validate_runner_schema(spec: Mapping[str, Any]) -> None:
    runner = spec["runner"]
    module = str(runner["module"])
    expected = RUNNER_BY_COMPONENT.get(str(spec["changed_component"]))
    if module != expected or module not in RUNNER_FLAGS:
        raise ContractError(
            f"runner {module!r} is not pinned for {spec['changed_component']!r}"
        )
    argv = list(runner["argv"])
    seen: set[str] = set()
    index = 0
    while index < len(argv):
        flag = argv[index]
        if not isinstance(flag, str) or not flag.startswith("--"):
            raise ContractError(f"runner has an unflagged/positional argument: {flag!r}")
        if flag not in RUNNER_FLAGS[module]:
            raise ContractError(f"runner has an unapproved flag: {flag}")
        if flag in seen:
            raise ContractError(f"runner repeats a flag: {flag}")
        seen.add(flag)
        index += 1
        if flag == "--dilations":
            start = index
            while index < len(argv) and not str(argv[index]).startswith("--"):
                index += 1
            if index == start:
                raise ContractError("--dilations requires at least one value")
        else:
            if index >= len(argv) or str(argv[index]).startswith("--"):
                raise ContractError(f"runner flag requires exactly one value: {flag}")
            index += 1
    required = RUNNER_FLAGS[module] - ({"--header-roles"} if module.endswith("postprocess") else set())
    if not required.issubset(seen):
        raise ContractError(f"runner misses required flags: {sorted(required - seen)}")
    if module.endswith("postprocess"):
        operation = argv[argv.index("--operation") + 1]
        has_headers = "--header-roles" in seen
        if (operation == "header_controller") != has_headers:
            raise ContractError("header_controller alone requires --header-roles")


def _artifact_row(path: Path, candidate_dir: Path) -> dict[str, Any]:
    path = path.resolve()
    path.relative_to(candidate_dir.resolve())
    return {
        "path": path.relative_to(candidate_dir.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _prepare_result(args: argparse.Namespace) -> None:
    candidate_dir = args.candidate_dir.resolve()
    spec = load_json(candidate_dir / "spec.json")
    execution = load_json(candidate_dir / "execution.json")
    metrics = load_json(args.metrics_report)
    tests = load_json(args.tests)
    paired = load_json(args.paired_deltas)
    backend_receipt = load_json(args.backend_receipt)
    reasons = list(args.reject_reason or ())
    if execution.get("returncode") != 0:
        reasons.append("backend_returncode_nonzero")
    if metrics.get("status") != "passed" or int(metrics.get("document_count", -1)) != 268:
        reasons.append("work_metrics_not_passed_268")
    if tests.get("status") != "passed":
        reasons.append("tests_not_passed")
    acceptance, gate_reasons = _evaluate_acceptance(
        spec,
        backend_receipt,
        metrics,
        paired,
        tests,
        load_json(candidate_dir / "invariants.json"),
    )
    reasons.extend(gate_reasons)
    backend_root = args.backend_receipt.resolve().parent
    artifacts = {}
    for path in sorted(backend_root.rglob("*")):
        if path.is_file() and path.resolve() != args.prediction.resolve():
            artifacts[path.relative_to(backend_root).as_posix()] = _artifact_row(path, candidate_dir)
    result = {
        "schema_version": "bibliography-evolution-result-v1",
        "status": "passed" if not reasons else "rejected",
        "all_rows": _artifact_row(args.all_rows, candidate_dir),
        "artifacts": artifacts,
        "predictions": {"main": _artifact_row(args.prediction, candidate_dir)},
        "metrics": metrics["metrics"],
        "metrics_by_source": metrics["metrics_by_source"],
        "paired_deltas": paired,
        "runtime": {
            "wall_seconds": execution.get("wall_seconds"),
            "backend_schema_version": backend_receipt.get("schema_version"),
        },
        "job": {
            "slurm_job_id": execution.get("slurm_job_id"),
            "slurm_array_task_id": execution.get("slurm_array_task_id"),
            "queue_sha256": execution.get("queue_sha256"),
            "queue_index": execution.get("queue_index"),
        },
        "tests": tests,
        "selection": {
            "eligible_for_pareto": not reasons,
            "rule": "passed backend, fixed 268-document metrics, tests, leakage, and explicit rejection gates",
            "acceptance": acceptance,
        },
        "rejection": {"reasons": sorted(set(reasons))},
    }
    write_json_exclusive(args.output, result)


def _evaluate_acceptance(
    spec: Mapping[str, Any],
    backend: Mapping[str, Any],
    metrics: Mapping[str, Any],
    paired: Mapping[str, Any],
    tests: Mapping[str, Any],
    invariants: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    rules = spec["acceptance_rule"]
    checks: dict[str, bool] = {}
    checks["paired_bootstrap_complete"] = (
        paired.get("schema_version") == "bibliography-evolution-paired-work-bootstrap-v1"
        and int(paired.get("work_count", 0)) > 0
        and set(paired.get("deltas_candidate_minus_baseline", ()))
        == {"token_fp", "token_fn", "spurious_blocks_per_zero_block_document", "mean_boundary_error_emitted_lines"}
        and isinstance(paired.get("candidate_rows_sha256"), str)
        and isinstance(paired.get("baseline_rows_sha256"), str)
    )
    checks["tests_bound_to_commit"] = (
        tests.get("status") == "passed" and tests.get("code_commit") == spec["code_commit"]
    )
    trace = backend.get("module_trace", ())
    trace_names = [row.get("module") for row in trace if isinstance(row, Mapping)]
    enabled_trace = [
        row for row in trace
        if isinstance(row, Mapping) and row.get("status") == "enabled_changed_family"
    ]
    canonical_g3 = [
        "internal_gap_connection",
        "boundary_trim",
        "outward_edge_optional",
        "weak_unseeded_optional",
        "whole_component_veto",
    ]
    expected_operation = {
        "decoder.gap_connector": "internal_gap_connection",
        "decoder.fringe_trim": "boundary_trim",
        "decoder.outward_edge": "outward_edge",
        "decoder.weak_unseeded": "weak_unseeded",
        "decoder.component_veto": "whole_component_veto",
    }.get(spec["changed_component"])
    if spec["generation"] == "G3":
        from .bibliography_evolution_postprocess import REFERENCE_PARAMETERS

        fixed_rows_valid = all(
            row.get("status") != "executed_fixed_reference"
            or (
                row.get("operation") in REFERENCE_PARAMETERS
                and float(row.get("threshold"))
                == float(REFERENCE_PARAMETERS[row["operation"]]["threshold"])
                and int(row.get("max_lines"))
                == int(REFERENCE_PARAMETERS[row["operation"]]["max_lines"])
            )
            for row in trace
            if isinstance(row, Mapping)
        )
        checks["canonical_g3_execution_trace"] = (
            trace_names == canonical_g3
            and len(enabled_trace) == 1
            and enabled_trace[0].get("operation") == expected_operation
            and all(
                row.get("status")
                in {"enabled_changed_family", "executed_fixed_reference"}
                for row in trace
                if isinstance(row, Mapping)
            )
            and fixed_rows_valid
        )
    for name, expected in rules.items():
        if name in {"pareto_eligible", "gpu_allowed_only_for_this_signal_training"}:
            checks[name] = bool(expected)
        elif name == "headline_documents":
            checks[name] = int(metrics.get("document_count", -1)) == int(expected)
        elif name in {"no_scope_or_physical_wall_crossing", "no_wall_crossing"}:
            checks[name] = (
                invariants.get("hard_wall_prediction_count") == 0
                and invariants.get("directional_wall_crossing_count") == 0
                and tests.get("invariants", {}).get("physical_gap_walls") is True
            )
        elif name == "headers_are_non_seeds":
            checks[name] = tests.get("invariants", {}).get("header_roles_non_seed") is True
        elif name == "non_bib_headers_never_included":
            checks[name] = invariants.get("non_bib_header_prediction_count") == 0
        elif name == "byte_identical_prediction":
            checks[name] = backend.get("status") == "passed_executed_byte_identical_replay"
        elif name == "receipt_hashes_match":
            verification = backend.get("verification", {})
            checks[name] = verification.get("status") == "passed_byte_identical"
        elif name == "runs_before_boundary_trim":
            checks[name] = (
                trace_names == canonical_g3
                and trace_names.index("internal_gap_connection") < trace_names.index("boundary_trim")
                and expected_operation == "internal_gap_connection"
            )
        elif name == "runs_after_internal_gap_connection":
            checks[name] = (
                trace_names == canonical_g3
                and trace_names.index("internal_gap_connection") < trace_names.index("boundary_trim")
                and expected_operation == "boundary_trim"
            )
        elif name == "runs_after_boundary_trim":
            checks[name] = (
                trace_names == canonical_g3
                and trace_names.index("boundary_trim")
                < trace_names.index(enabled_trace[0]["module"])
                if len(enabled_trace) == 1 and enabled_trace[0].get("module") in trace_names
                else False
            )
        elif name == "whole_component_only":
            checks[name] = (
                backend.get("operation") == "whole_component_veto"
                and len(enabled_trace) == 1
                and enabled_trace[0].get("operation") == "whole_component_veto"
            )
        elif name == "work_grouped_oof":
            checks[name] = "train_oof" in str(backend.get("status", ""))
        elif name == "no_refitting":
            checks[name] = spec["runner"]["module"] == "sequence_models.bibliography_evolution_composition"
        elif name == "parents_must_both_be_pareto":
            try:
                lineage = verify_parent_lineage(spec)
            except ContractError:
                checks[name] = False
            else:
                checks[name] = (
                    len(spec["parent_candidate_ids"]) == 2
                    and lineage.get("pareto_registry") is not None
                )
        else:
            checks[name] = False
    reasons = [f"acceptance_gate_failed:{name}" for name, passed in checks.items() if not passed]
    return {"rules": rules, "checks": checks, "passed": not reasons}, reasons


def _compute_invariants(
    spec: Mapping[str, Any], candidate_dir: Path, prediction_path: Path
) -> dict[str, Any]:
    import numpy as np

    prediction = np.load(prediction_path, allow_pickle=False).astype(bool)
    barrier_path = candidate_dir / "backend" / "combined_barriers.npz"
    if not barrier_path.is_file():
        raise ContractError("backend did not preserve combined_barriers.npz")
    barrier = np.load(barrier_path, allow_pickle=False)
    hard = barrier["hard_wall"].astype(bool)
    upward = barrier["upward_stop"].astype(bool)
    downward = barrier["downward_stop"].astype(bool)
    if not (prediction.shape == hard.shape == upward.shape == downward.shape):
        raise ContractError("prediction/barrier invariant arrays do not align")
    crossings = int(np.count_nonzero(prediction[:-1] & prediction[1:] & (upward[1:] | downward[:-1])))
    non_bib_count = 0
    runner_argv = list(spec["runner"]["argv"])
    header_inputs: list[Path] = []
    if "--header-roles" in runner_argv:
        header_inputs.append(Path(runner_argv[runner_argv.index("--header-roles") + 1]))
    if header_inputs:
        roles = np.load(header_inputs[0], allow_pickle=False)
        if roles.shape != prediction.shape:
            raise ContractError("header roles do not align to prediction")
        non_bib_count = int(np.count_nonzero(prediction & (roles == 3)))
    return {
        "schema_version": "bibliography-evolution-invariants-v1",
        "hard_wall_prediction_count": int(np.count_nonzero(prediction & hard)),
        "directional_wall_crossing_count": crossings,
        "non_bib_header_prediction_count": non_bib_count,
        "barrier_sha256": sha256_file(barrier_path),
        "prediction_sha256": sha256_file(prediction_path),
    }


def _one_input_by_class(spec: Mapping[str, Any], data_class: str) -> Path:
    matches = [
        Path(str(row["path"])).resolve()
        for row in spec["input_receipts"].values()
        if row.get("data_class") == data_class
    ]
    if len(matches) != 1:
        raise ContractError(f"candidate requires exactly one {data_class!r} input")
    return matches[0]


def _auto_finalize_candidate(
    store: CandidateStore, candidate_dir: Path, spec: Mapping[str, Any]
) -> Path:
    """Complete metrics, paired bootstrap, test attestation, and receipt."""

    from .bibliography_evolution_metrics import parse_args as metrics_args
    from .bibliography_evolution_metrics import run as run_metrics

    backend = candidate_dir / "backend"
    prediction = backend / "prediction.npy"
    backend_receipt = backend / "receipt.json"
    if not prediction.is_file() or not backend_receipt.is_file():
        raise ContractError("backend did not emit prediction.npy and receipt.json")

    table = _one_input_by_class(spec, "development_table")
    qualified = _one_input_by_class(spec, "qualified_development_inventory")
    work_rows = candidate_dir / "work_objectives.jsonl"
    metrics_report = candidate_dir / "work_metrics.json"
    run_metrics(
        metrics_args(
            [
                "--table-dir", str(table),
                "--prediction", str(prediction),
                "--qualified-documents", str(qualified),
                "--output-rows", str(work_rows),
                "--output-report", str(metrics_report),
            ]
        )
    )
    baseline_rows = _one_input_by_class(spec, "baseline_work_objectives")
    paired = paired_work_bootstrap(
        _jsonl(work_rows), _jsonl(baseline_rows), iterations=2000, seed=20260718
    )
    paired["candidate_rows_sha256"] = sha256_file(work_rows)
    paired["baseline_rows_sha256"] = sha256_file(baseline_rows)
    paired_path = candidate_dir / "paired_bootstrap.json"
    write_json_exclusive(paired_path, paired)
    tests_path = _one_input_by_class(spec, "code_test_receipt")
    tests = load_json(tests_path)
    if tests.get("status") != "passed" or tests.get("code_commit") != spec["code_commit"]:
        raise ContractError("code test receipt is not passed for this exact commit")
    pinned_tests = candidate_dir / "tests.json"
    write_json_exclusive(pinned_tests, tests)
    invariants_path = candidate_dir / "invariants.json"
    write_json_exclusive(
        invariants_path, _compute_invariants(spec, candidate_dir, prediction)
    )
    result_path = candidate_dir / "result.json"
    _prepare_result(
        argparse.Namespace(
            candidate_dir=candidate_dir,
            backend_receipt=backend_receipt,
            prediction=prediction,
            all_rows=work_rows,
            metrics_report=metrics_report,
            paired_deltas=paired_path,
            tests=pinned_tests,
            reject_reason=[],
            output=result_path,
        )
    )
    return store.finalize(candidate_dir, load_json(result_path))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ContractError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"{label} is not UTF-8") from error
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ContractError(f"{label}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ContractError(f"{label}:{line_number}: expected an object")
        rows.append(row)
    return rows


def _read_exact_bytes(path: Path | str, expected_sha256: str, label: str) -> bytes:
    source = _regular_file(path, label)
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ContractError(f"{label} bytes changed")
    return payload


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_bytes_exclusive(
        path, b"".join(canonical_json_bytes(row) for row in rows)
    )


def _write_jsonl_exact_resume(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    atomic_write_bytes_exact_resume(path, payload)


def _validate_runner(
    spec: Mapping[str, Any], *, candidate_dir: Path | None = None
) -> list[str]:
    validate_candidate_spec(spec)
    runner = spec["runner"]
    module = str(runner["module"])
    if not module.startswith("sequence_models.bibliography_"):
        raise ContractError("candidate runner must be a bibliography module")
    values = list(runner["argv"])
    if any("@CANDIDATE_DIR@" in value for value in values) and candidate_dir is None:
        raise ContractError("runner requires an initialized candidate directory")
    if candidate_dir is not None:
        values = [
            value.replace("@CANDIDATE_DIR@", str(candidate_dir.resolve()))
            for value in values
        ]
    if any("@SLURM_JOB_ID@" in value for value in values):
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if candidate_dir is not None and not slurm_job_id:
            raise ContractError("SLURM_JOB_ID is required at execution time")
        replacement = slurm_job_id or "unresolved-runtime-job"
        values = [value.replace("@SLURM_JOB_ID@", replacement) for value in values]
    argv = [sys.executable, "-m", module, *values]
    if any("${" in value for value in argv):
        raise ContractError("candidate runner contains an unresolved binding")
    return argv


def _attest_git_checkout(cwd: Path, expected_commit: str) -> dict[str, Any]:
    def git(*values: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(cwd.resolve()), *values],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode:
            raise ContractError(f"git checkout attestation failed: {completed.stderr.strip()}")
        return completed.stdout.strip()

    head = git("rev-parse", "HEAD")
    if head != expected_commit:
        raise ContractError(f"code commit drift: expected {expected_commit}, found {head}")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ContractError("code checkout is not clean")
    return {"status": "passed", "head": head, "clean": True}


def _verify_sealed_freeze(
    documents_path: Path,
    labels_path: Path,
    consensus_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    # The prediction lane may read the sealed documents and terminal FROZEN
    # receipt, but it must not touch label or consensus bytes. Their expected
    # hashes come from the annotation seal and are checked only after the
    # candidate replay preflight has durably committed the final fuse.
    documents_path = _regular_file(documents_path, "sealed documents")
    receipt_path = _regular_file(receipt_path, "annotation FROZEN receipt")
    labels_path = _regular_file(labels_path, "sealed labels")
    consensus_path = _regular_file(consensus_path, "sealed consensus receipt")
    documents = _jsonl(documents_path)
    document_ids = [str(row.get("document_id", "")) for row in documents]
    source_counts: dict[str, int] = {}
    for row in documents:
        source = str(row.get("source", ""))
        source_counts[source] = source_counts.get(source, 0) + 1
    if (
        len(documents) != 150
        or len(set(document_ids)) != 150
        or any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in document_ids)
        or source_counts != {"greek_phd": 50, "kallipos": 50, "openarchives": 50}
    ):
        raise ContractError("sealed documents must contain 150 unique SHA-256 IDs and exactly 50/source")
    receipt = load_json(receipt_path)
    hashes = receipt.get("sealed_hashes", {})
    if (
        receipt.get("schema_version") != SEALED_FREEZE_SCHEMA
        or receipt.get("status") != "frozen_prediction_blind_test_set"
        or int(receipt.get("document_count", -1)) != 150
        or receipt.get("source_document_counts") != source_counts
        or hashes.get("documents_sha256") != sha256_file(documents_path)
        or not isinstance(hashes.get("labels_sha256"), str)
        or len(hashes["labels_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in hashes["labels_sha256"])
        or not isinstance(hashes.get("consensus_receipt_sha256"), str)
        or len(hashes["consensus_receipt_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in hashes["consensus_receipt_sha256"]
        )
    ):
        raise ContractError("annotation-lane FROZEN receipt does not bind sealed bytes")
    return {
        "status": "passed",
        "document_ids": document_ids,
        "documents": {"path": str(documents_path), "sha256": sha256_file(documents_path)},
        "labels": {"path": str(labels_path), "sha256": hashes["labels_sha256"]},
        "consensus_receipt": {
            "path": str(consensus_path),
            "sha256": hashes["consensus_receipt_sha256"],
        },
        "freeze_receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
        "source_document_counts": source_counts,
    }


def _sealed_frontier_paths(freeze_receipt_path: Path) -> tuple[Path, Path]:
    """Return the one output-independent lock and evaluation root for a seal."""

    parent = _regular_file(
        freeze_receipt_path, "annotation FROZEN receipt"
    ).parent
    return (
        parent / "BIBLIOGRAPHY_EVOLUTION_FINAL_FRONTIER.lock.json",
        parent / "BIBLIOGRAPHY_EVOLUTION_FINAL_EVALUATION",
    )


def _frontier_lock_payload(
    *,
    registry_path: Path,
    registry_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
    sealed: Mapping[str, Any],
    runtime_code: Mapping[str, Any],
    canonical_batch_root: Path,
) -> dict[str, Any]:
    core = {
        "schema_version": SEALED_FRONTIER_LOCK_SCHEMA,
        "status": "reserved_exact_frontier_before_sealed_evaluation",
        "canonical_batch_root": str(canonical_batch_root.resolve()),
        "runtime_code": runtime_code,
        "registry": {
            "path": str(registry_path.resolve()),
            "sha256": registry_sha256,
        },
        "terminal_seal": sealed["freeze_receipt"],
        "sealed_documents": sealed["documents"],
        "sealed_labels": sealed["labels"],
        "sealed_consensus_receipt": sealed["consensus_receipt"],
        "source_document_counts": sealed["source_document_counts"],
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "generation": row["generation"],
                "code_commit": row["code_commit"],
                "receipt_sha256": row["receipt_sha256"],
                "spec_sha256": row["spec_sha256"],
            }
            for row in candidates
        ],
    }
    return {
        **core,
        "frontier_id": hashlib.sha256(canonical_json_bytes(core)).hexdigest(),
    }


def _reserve_exact_frontier(lock_path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically reserve the terminal seal for one immutable frontier."""

    atomic_write_bytes_exact_resume(
        lock_path,
        canonical_json_bytes(payload),
        mode=0o440,
        label="sealed frontier lock",
    )
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ContractError("sealed frontier lock is not a regular file")


def _freeze_manifest(
    registry_path: Path,
    output: Path,
    sealed_documents_path: Path,
    sealed_labels_path: Path,
    sealed_consensus_receipt_path: Path,
    sealed_freeze_receipt_path: Path,
) -> None:
    from .bibliography_evolution_sealed_inference import runtime_code_inventory

    raw_output = Path(output).expanduser()
    if raw_output.is_symlink():
        raise ContractError("frozen Pareto manifest output cannot be a symlink")
    output = raw_output.resolve()
    registry_path = _regular_file(registry_path, "development registry")
    registry = load_json(registry_path)
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ContractError("unsupported evolution registry")
    receipt_paths = [Path(row["receipt_path"]) for row in registry.get("candidates", ())]
    rebuilt = build_registry(receipt_paths)
    if rebuilt != registry:
        raise ContractError("stored development registry differs from a fresh frontier rebuild")
    pareto_ids = list(registry.get("pareto_candidate_ids", ()))
    if not pareto_ids:
        raise ContractError("cannot freeze an empty Pareto set")
    indexed = {row["candidate_id"]: row for row in registry["candidates"]}
    if set(pareto_ids) - set(indexed):
        raise ContractError("registry Pareto inventory is inconsistent")
    runtime_code = runtime_code_inventory()
    runtime_commit = str(runtime_code["git_commit"])
    candidates = []
    for candidate_id in pareto_ids:
        row = indexed[candidate_id]
        if not row.get("pareto") or not row.get("eligible"):
            raise ContractError("frozen candidate is not eligible Pareto")
        receipt_path = _regular_file(row["receipt_path"], "candidate receipt")
        if sha256_file(receipt_path) != row["receipt_sha256"]:
            raise ContractError(f"candidate receipt changed: {candidate_id}")
        verification = verify_finalized_receipt(receipt_path)
        receipt = load_json(receipt_path)
        spec = load_json(receipt_path.parent / "spec.json")
        if verification["candidate_id"] != candidate_id:
            raise ContractError("candidate verification identity differs from registry")
        if spec.get("code_commit") != runtime_commit:
            raise ContractError(
                f"candidate code commit is incompatible with sealed runtime: {candidate_id}"
            )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "generation": spec["generation"],
                "code_commit": spec["code_commit"],
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": row["receipt_sha256"],
                "spec_sha256": receipt["spec_sha256"],
                "artifacts": receipt.get("artifacts", {}),
                "predictions": receipt.get("predictions", {}),
                "all_rows": receipt.get("all_rows"),
                "verified_inventory": verification,
            }
        )
    sealed = _verify_sealed_freeze(
        sealed_documents_path,
        sealed_labels_path,
        sealed_consensus_receipt_path,
        sealed_freeze_receipt_path,
    )
    frontier_lock_path, canonical_batch_root = _sealed_frontier_paths(
        Path(sealed["freeze_receipt"]["path"])
    )
    frontier_lock = _frontier_lock_payload(
        registry_path=registry_path,
        registry_sha256=sha256_file(registry_path),
        candidates=candidates,
        sealed=sealed,
        runtime_code=runtime_code,
        canonical_batch_root=canonical_batch_root,
    )
    _reserve_exact_frontier(frontier_lock_path, frontier_lock)
    core = {
        "schema_version": SEALED_MANIFEST_SCHEMA,
        "status": "frozen_before_model_evaluation",
        "evaluation_mode": "one_simultaneous_batch_all_pareto_candidates",
        "incremental_evaluation_allowed": False,
        # Bind the only permitted final-evaluation root into the signed
        # manifest itself. Deriving it from the manifest's current path would
        # allow a copied manifest to mint a second fuse elsewhere.
        "canonical_batch_root": str(canonical_batch_root),
        "sealed_frontier_lock": {
            "path": str(frontier_lock_path),
            "sha256": sha256_file(frontier_lock_path),
            "frontier_id": frontier_lock["frontier_id"],
        },
        "sealed_inference_runtime_code": runtime_code,
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path),
        "sealed_documents": sealed["documents"],
        "sealed_labels": sealed["labels"],
        "sealed_consensus_receipt": sealed["consensus_receipt"],
        "sealed_freeze_receipt": sealed["freeze_receipt"],
        "sealed_source_document_counts": sealed["source_document_counts"],
        "candidate_ids": pareto_ids,
        "candidates": candidates,
    }
    core["frozen_manifest_id"] = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    atomic_write_bytes_exact_resume(output, canonical_json_bytes(core), mode=0o440)


def _verify_frozen_manifest_fresh(manifest_path: Path) -> dict[str, Any]:
    """Rebuild the development frontier and validate the annotation seal.

    This verifier intentionally does not open the label or consensus files. It
    is safe to call from the prediction-only bridge.
    """

    from .bibliography_evolution_sealed_inference import runtime_code_inventory

    manifest_path = _regular_file(manifest_path, "frozen Pareto manifest")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SEALED_MANIFEST_SCHEMA:
        raise ContractError("unsupported frozen Pareto manifest")
    if manifest.get("sealed_inference_runtime_code") != runtime_code_inventory():
        raise ContractError("sealed inference runtime source changed after frontier freeze")
    canonical_batch_root = manifest.get("canonical_batch_root")
    if (
        not isinstance(canonical_batch_root, str)
        or not canonical_batch_root
        or not Path(canonical_batch_root).is_absolute()
    ):
        raise ContractError("frozen manifest lacks an absolute canonical batch root")
    frozen_id = manifest.get("frozen_manifest_id")
    core = {key: value for key, value in manifest.items() if key != "frozen_manifest_id"}
    if frozen_id != hashlib.sha256(canonical_json_bytes(core)).hexdigest():
        raise ContractError("frozen manifest ID does not match its current bytes")
    registry_path = _regular_file(
        str(manifest.get("registry_path", "")), "frozen development registry"
    )
    if (
        sha256_file(registry_path) != manifest.get("registry_sha256")
    ):
        raise ContractError("frozen development registry changed")
    registry = load_json(registry_path)
    receipt_paths = [Path(str(row["receipt_path"])) for row in registry.get("candidates", ())]
    rebuilt = build_registry(receipt_paths)
    if rebuilt != registry:
        raise ContractError("development frontier differs from a fresh registry rebuild")
    pareto_ids = list(registry.get("pareto_candidate_ids", ()))
    candidates = manifest.get("candidates")
    if (
        not pareto_ids
        or manifest.get("candidate_ids") != pareto_ids
        or not isinstance(candidates, list)
        or len(candidates) != len(pareto_ids)
        or [row.get("candidate_id") for row in candidates] != pareto_ids
        or len(set(pareto_ids)) != len(pareto_ids)
    ):
        raise ContractError("frozen candidate inventory/cardinality differs from fresh Pareto")
    registry_rows = {row["candidate_id"]: row for row in registry["candidates"]}
    runtime_commit = str(manifest["sealed_inference_runtime_code"].get("git_commit", ""))
    for row in candidates:
        candidate_id = row["candidate_id"]
        current = registry_rows.get(candidate_id, {})
        receipt_path = _regular_file(
            str(row.get("receipt_path", "")), "frozen Pareto candidate receipt"
        )
        receipt = load_json(receipt_path)
        spec_path = _regular_file(
            receipt_path.parent / "spec.json", "frozen Pareto candidate spec"
        )
        spec = load_json(spec_path)
        if (
            not current.get("pareto")
            or not current.get("eligible")
            or receipt_path != Path(str(current.get("receipt_path", ""))).resolve()
            or sha256_file(receipt_path) != row.get("receipt_sha256")
            or row.get("receipt_sha256") != current.get("receipt_sha256")
            or row.get("spec_sha256") != sha256_file(spec_path)
            or receipt.get("spec_sha256") != row.get("spec_sha256")
            or row.get("generation") != spec.get("generation")
            or row.get("code_commit") != runtime_commit
            or spec.get("code_commit") != runtime_commit
        ):
            raise ContractError(f"frozen Pareto candidate changed: {candidate_id}")
    if manifest.get("sealed_source_document_counts") != {
        "greek_phd": 50, "kallipos": 50, "openarchives": 50
    }:
        raise ContractError("frozen sealed source balance is not exactly 50/source")
    freeze_row = manifest.get("sealed_freeze_receipt", {})
    freeze_path = _regular_file(
        str(freeze_row.get("path", "")), "annotation FROZEN receipt"
    )
    if (
        sha256_file(freeze_path) != freeze_row.get("sha256")
    ):
        raise ContractError("annotation FROZEN receipt changed")
    freeze = load_json(freeze_path)
    hashes = freeze.get("sealed_hashes", {})
    if (
        freeze.get("schema_version") != SEALED_FREEZE_SCHEMA
        or freeze.get("status") != "frozen_prediction_blind_test_set"
        or int(freeze.get("document_count", -1)) != 150
        or freeze.get("source_document_counts") != manifest["sealed_source_document_counts"]
        or hashes.get("documents_sha256") != manifest.get("sealed_documents", {}).get("sha256")
        or hashes.get("labels_sha256") != manifest.get("sealed_labels", {}).get("sha256")
        or hashes.get("consensus_receipt_sha256")
        != manifest.get("sealed_consensus_receipt", {}).get("sha256")
    ):
        raise ContractError("annotation FROZEN receipt differs from the manifest")
    expected_lock_path, expected_batch_root = _sealed_frontier_paths(freeze_path)
    if (
        Path(str(manifest.get("canonical_batch_root", ""))) != expected_batch_root
    ):
        raise ContractError("canonical batch root is not tied to the terminal seal")
    sealed_view = {
        "documents": manifest["sealed_documents"],
        "labels": manifest["sealed_labels"],
        "consensus_receipt": manifest["sealed_consensus_receipt"],
        "freeze_receipt": manifest["sealed_freeze_receipt"],
        "source_document_counts": manifest["sealed_source_document_counts"],
    }
    expected_lock = _frontier_lock_payload(
        registry_path=registry_path,
        registry_sha256=sha256_file(registry_path),
        candidates=candidates,
        sealed=sealed_view,
        runtime_code=manifest["sealed_inference_runtime_code"],
        canonical_batch_root=expected_batch_root,
    )
    lock_row = manifest.get("sealed_frontier_lock", {})
    lock_path = _regular_file(
        str(lock_row.get("path", "")), "sealed frontier lock"
    )
    if (
        lock_path != expected_lock_path
        or sha256_file(lock_path) != lock_row.get("sha256")
        or lock_row.get("frontier_id") != expected_lock["frontier_id"]
        or load_json(lock_path) != expected_lock
    ):
        raise ContractError("sealed frontier lock differs from the exact frozen frontier")
    return manifest


def _sealed_request(
    manifest: Mapping[str, Any], request_path: Path
) -> tuple[dict[str, Any], Path]:
    request_path = _regular_file(request_path, "sealed batch request")
    request = load_json(request_path)
    required = {
        "schema_version", "evaluation_mode", "frozen_manifest_id",
        "candidate_ids", "sealed_inference_receipt", "bootstrap",
    }
    if set(request) != required:
        raise ContractError("sealed request has missing or unapproved fields")
    if (
        request.get("schema_version") != SEALED_REQUEST_SCHEMA
        or request.get("evaluation_mode") != "one_simultaneous_batch_all_pareto_candidates"
        or request.get("frozen_manifest_id") != manifest["frozen_manifest_id"]
        or request.get("candidate_ids") != manifest["candidate_ids"]
    ):
        raise ContractError("sealed request does not match the frozen frontier")
    inference_row = request.get("sealed_inference_receipt")
    if not isinstance(inference_row, Mapping) or set(inference_row) != {"path", "sha256"}:
        raise ContractError("sealed request lacks one exact inference receipt binding")
    inference_path = _regular_file(
        str(inference_row["path"]), "sealed inference receipt"
    )
    if (
        sha256_file(inference_path) != inference_row["sha256"]
    ):
        raise ContractError("sealed inference receipt changed")
    bootstrap = request.get("bootstrap")
    if (
        not isinstance(bootstrap, Mapping)
        or set(bootstrap) != {"method", "iterations", "seed"}
        or bootstrap.get("method")
        != "source_stratified_work_bootstrap_bonferroni_simultaneous"
        or not isinstance(bootstrap.get("iterations"), int)
        or isinstance(bootstrap.get("iterations"), bool)
        or int(bootstrap["iterations"]) <= 0
        or not isinstance(bootstrap.get("seed"), int)
        or isinstance(bootstrap.get("seed"), bool)
    ):
        raise ContractError("sealed request has an invalid frozen bootstrap contract")
    return request, inference_path


def _preflight_sealed_batch(
    manifest_path: Path, request_path: Path, batch_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from .bibliography_evolution_sealed_inference import verify_inference_receipt

    manifest = _verify_frozen_manifest_fresh(manifest_path)
    request, inference_path = _sealed_request(manifest, request_path)
    inference = verify_inference_receipt(manifest, inference_path)
    requested_root = Path(batch_root).expanduser()
    declared_root = Path(manifest["canonical_batch_root"])
    if requested_root.is_symlink() or declared_root.is_symlink():
        raise ContractError("canonical sealed batch root cannot be a symlink")
    batch_root = requested_root.resolve()
    canonical_root = declared_root.resolve()
    if batch_root != canonical_root:
        raise ContractError(f"sealed batch root is not canonical: {canonical_root}")
    if batch_root.exists() and not batch_root.is_dir():
        raise ContractError("canonical sealed batch root is not a real directory")
    return manifest, request, inference


def _commit_sealed_fuse(
    manifest_path: Path,
    request_path: Path,
    batch_root: Path,
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
    inference: Mapping[str, Any],
) -> Path:
    raw_root = Path(batch_root).expanduser()
    if raw_root.is_symlink():
        raise ContractError("canonical final-evaluation root is a symlink")
    batch_root = raw_root.resolve()
    fuse = batch_root / "FINAL_EVALUATION_FUSE.json"
    payload = {
        "schema_version": "bibliography-evolution-sealed-evaluation-fuse-v1",
        "status": "committed_after_all_prediction_shape_table_hash_preflight",
        "labels_parsed": False,
        "frozen_manifest_id": manifest["frozen_manifest_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "request_sha256": sha256_file(request_path),
        "inference_receipt_sha256": inference["receipt_sha256"],
        "candidate_ids": request["candidate_ids"],
        "line_count": inference["line_count"],
        "bootstrap": request["bootstrap"],
    }
    if batch_root.exists() or batch_root.is_symlink():
        if not fuse.is_file() or fuse.is_symlink() or load_json(fuse) != payload:
            raise ContractError("canonical final-evaluation fuse differs or is incomplete")
        return fuse
    parent = batch_root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ContractError("canonical final-evaluation parent is not a real directory")
    stage = Path(
        tempfile.mkdtemp(prefix=f".{batch_root.name}.atomic-", dir=parent)
    )
    stage_fuse = stage / fuse.name
    try:
        atomic_write_bytes_exclusive(
            stage_fuse, canonical_json_bytes(payload), mode=0o440
        )
        _fsync_directory(stage)
        try:
            os.rename(stage, batch_root)
            _fsync_directory(parent)
        except OSError:
            if (
                not batch_root.is_symlink()
                and (batch_root / fuse.name).is_file()
                and not (batch_root / fuse.name).is_symlink()
                and load_json(batch_root / fuse.name) == payload
            ):
                stage_fuse.unlink(missing_ok=True)
                stage.rmdir()
            else:
                raise
    finally:
        if stage.exists():
            stage_fuse.unlink(missing_ok=True)
            try:
                stage.rmdir()
            except OSError:
                pass
    fuse = batch_root / fuse.name
    if fuse.is_symlink() or not fuse.is_file() or load_json(fuse) != payload:
        raise ContractError("atomic final-evaluation fuse publication failed")
    return fuse


def _begin_sealed_batch(manifest_path: Path, request_path: Path, batch_root: Path) -> Path:
    manifest, request, inference = _preflight_sealed_batch(
        manifest_path, request_path, batch_root
    )
    return _commit_sealed_fuse(
        manifest_path,
        request_path,
        batch_root,
        manifest,
        request,
        inference,
    )


def _sealed_work_rows(table: Any, prediction: Any, selected: set[int]) -> list[dict[str, Any]]:
    from .bibliography_evolution_metrics import _document_objectives

    indexed: dict[str, dict[str, Any]] = {}
    for document_index in sorted(selected):
        row = _document_objectives(table, prediction, document_index)
        work_id = row["work_id"]
        if work_id not in indexed:
            indexed[work_id] = row
            continue
        target = indexed[work_id]
        if target["source"] != row["source"]:
            raise ContractError("one sealed work identity crosses source strata")
        for field in (
            "token_fp", "token_fn", "spurious_zero_blocks", "zero_doc_count",
            "boundary_error_sum", "boundary_match_count", "document_count",
        ):
            target[field] += row[field]
    return [indexed[key] for key in sorted(indexed)]


def _simultaneous_sealed_intervals(
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    import random

    from .bibliography_evolution_contract import OBJECTIVES, _aggregate_work_rows

    if iterations <= 0 or not rows_by_candidate:
        raise ContractError("sealed bootstrap requires candidates and positive iterations")
    indexed = {
        candidate_id: {str(row["work_id"]): row for row in rows}
        for candidate_id, rows in rows_by_candidate.items()
    }
    first = next(iter(indexed.values()))
    if any(set(rows) != set(first) for rows in indexed.values()):
        raise ContractError("sealed candidate work inventories differ")
    by_source: dict[str, list[str]] = {}
    for work_id, row in first.items():
        by_source.setdefault(str(row["source"]), []).append(work_id)
    rng = random.Random(seed)
    samples: dict[tuple[str, str], list[float]] = {
        (candidate_id, objective): []
        for candidate_id in sorted(indexed)
        for objective in OBJECTIVES
    }
    for _ in range(iterations):
        sampled: list[str] = []
        for source in sorted(by_source):
            group = sorted(by_source[source])
            sampled.extend(rng.choice(group) for _ in range(len(group)))
        for candidate_id in sorted(indexed):
            vector = _aggregate_work_rows([indexed[candidate_id][work_id] for work_id in sampled])
            for objective, value in zip(OBJECTIVES, vector, strict=True):
                samples[(candidate_id, objective)].append(float(value))

    family_size = len(samples)
    tail = 0.05 / (2 * family_size)

    def percentile(values: Sequence[float], probability: float) -> float:
        ordered = sorted(values)
        position = probability * (len(ordered) - 1)
        lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    intervals: dict[str, dict[str, Any]] = {}
    for candidate_id in sorted(indexed):
        point = _aggregate_work_rows(list(indexed[candidate_id].values()))
        intervals[candidate_id] = {
            objective: {
                "point": float(value),
                "simultaneous_ci95": [
                    percentile(samples[(candidate_id, objective)], tail),
                    percentile(samples[(candidate_id, objective)], 1 - tail),
                ],
            }
            for objective, value in zip(OBJECTIVES, point, strict=True)
        }
    return {
        "method": "source_stratified_work_bootstrap_bonferroni_simultaneous",
        "familywise_alpha": 0.05,
        "family_size": family_size,
        "per_tail_probability": tail,
        "iterations": iterations,
        "seed": seed,
        "intervals": intervals,
    }


def _evaluate_sealed_batch(
    manifest_path: Path,
    request_path: Path,
    batch_root: Path,
    *,
    iterations: int,
    seed: int,
) -> Path:
    import numpy as np

    from .bibliography_entry_blocks import evaluate_prediction
    from .bibliography_entry_dataset import LABEL_TO_ID, TARGET_ENTRY, TARGET_NEGATIVE
    from .bibliography_entry_models import load_table
    # Bind these CLI values before the fuse. A crash/restart must not be able
    # to silently change the simultaneous-interval procedure.
    raw_request = load_json(request_path)
    if raw_request.get("bootstrap") != {
        "method": "source_stratified_work_bootstrap_bonferroni_simultaneous",
        "iterations": iterations,
        "seed": seed,
    }:
        raise ContractError("CLI bootstrap values differ from the sealed request")
    manifest, request, inference = _preflight_sealed_batch(
        manifest_path, request_path, batch_root
    )
    fuse = _commit_sealed_fuse(
        manifest_path,
        request_path,
        batch_root,
        manifest,
        request,
        inference,
    )
    result_path = batch_root.resolve() / "sealed_results.json"
    completion_path = batch_root.resolve() / "sealed_results.receipt.json"
    if result_path.exists() or result_path.is_symlink():
        if result_path.is_symlink() or not result_path.is_file():
            raise ContractError("existing sealed result is not a regular file")
        existing = load_json(result_path)
        existing_identity_matches = (
            existing.get("schema_version") == SEALED_RESULTS_SCHEMA
            and existing.get("frozen_manifest_id")
            == load_json(manifest_path).get("frozen_manifest_id")
            and existing.get("inputs", {}).get("manifest_sha256")
            == sha256_file(manifest_path)
            and existing.get("inputs", {}).get("request_sha256")
            == sha256_file(request_path)
        )
        if not existing_identity_matches:
            raise ContractError("existing sealed result differs from this exact frozen run")
        if completion_path.exists() or completion_path.is_symlink():
            if completion_path.is_symlink() or not completion_path.is_file():
                raise ContractError("sealed result receipt is not a regular file")
            completion = load_json(completion_path)
            if completion != {
                "schema_version": SEALED_RESULTS_RECEIPT_SCHEMA,
                "status": "passed_complete_exact_sealed_result",
                "frozen_manifest_id": manifest["frozen_manifest_id"],
                "candidate_ids": manifest["candidate_ids"],
                "result": {
                    "path": str(result_path),
                    "sha256": sha256_file(result_path),
                },
                "fuse_sha256": sha256_file(fuse),
            }:
                raise ContractError("sealed result completion receipt differs")
            return result_path
    elif completion_path.exists() or completion_path.is_symlink():
        raise ContractError("sealed result receipt exists without its result")
    # The fuse now exists. Only this side of the durable boundary may touch
    # label or consensus bytes. Every consumed artifact is loaded from the
    # exact byte snapshot whose hash is checked here, rather than reopened by
    # path after a time-of-check/time-of-use gap.
    labels_payload = _read_exact_bytes(
        manifest["sealed_labels"]["path"],
        manifest["sealed_labels"]["sha256"],
        "sealed labels",
    )
    consensus_payload = _read_exact_bytes(
        manifest["sealed_consensus_receipt"]["path"],
        manifest["sealed_consensus_receipt"]["sha256"],
        "sealed consensus receipt",
    )
    try:
        consensus = json.loads(consensus_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("sealed consensus receipt is invalid JSON") from error
    if not isinstance(consensus, Mapping):
        raise ContractError("sealed consensus receipt is not an object")
    labels = _jsonl_bytes(labels_payload, "sealed labels")

    table_root = Path(inference["feature_table"])
    expected_table_sha256 = str(inference["feature_table_sha256"])
    if (
        table_root.is_symlink()
        or not table_root.is_dir()
        or sha256_directory(table_root) != expected_table_sha256
    ):
        raise ContractError("sealed feature table changed after the fuse")
    mapped_table = load_table(table_root, expected_split="sealed_unlabelled")
    line_rows = _jsonl(table_root / "lines.jsonl")
    table = dataclasses.replace(
        mapped_table,
        counts=np.array(mapped_table.counts, copy=True),
        targets=np.array(mapped_table.targets, copy=True),
        original_labels=np.array(mapped_table.original_labels, copy=True),
        header_kinds=np.array(mapped_table.header_kinds, copy=True),
        abs_indices=np.array(mapped_table.abs_indices, copy=True),
        token_counts=np.array(mapped_table.token_counts, copy=True),
        char_lengths=np.array(mapped_table.char_lengths, copy=True),
        block_indices=np.array(mapped_table.block_indices, copy=True),
        document_indices=np.array(mapped_table.document_indices, copy=True),
        folds=np.array(mapped_table.folds, copy=True),
    )
    if sha256_directory(table_root) != expected_table_sha256:
        raise ContractError("sealed feature table drifted while being consumed")
    expected = {
        (str(row["document_id"]), str(row["line_id"])): (index, int(row["abs_idx"]))
        for index, row in enumerate(line_rows)
    }
    if len(expected) != len(line_rows) or len(labels) != len(line_rows):
        raise ContractError("sealed labels do not have exact feature-table cardinality")
    original = np.full(len(line_rows), LABEL_TO_ID["UNKNOWN"], dtype=np.uint8)
    targets = np.full(len(line_rows), TARGET_NEGATIVE, dtype=np.int8)
    seen: set[tuple[str, str]] = set()
    unresolved = 0
    for row in labels:
        key = (str(row.get("document_id", "")), str(row.get("line_id", "")))
        if key not in expected or key in seen:
            raise ContractError("sealed labels invent or duplicate a feature-table line")
        index, abs_idx = expected[key]
        if int(row.get("abs_idx", -1)) != abs_idx:
            raise ContractError("sealed label coordinate differs from feature table")
        binary = row.get("binary_label")
        if binary == "BIB":
            original[index] = LABEL_TO_ID["BIB"]
            targets[index] = TARGET_ENTRY
        elif binary == "NON_BIB":
            original[index] = LABEL_TO_ID["O"]
        elif binary is None and row.get("role") == "UNKNOWN":
            unresolved += 1
        else:
            raise ContractError("sealed label has invalid binary/role semantics")
        seen.add(key)
    if seen != set(expected):
        raise ContractError("sealed labels omit feature-table lines")
    labelled_table = dataclasses.replace(table, original_labels=original, targets=targets)
    selected = set(range(len(labelled_table.documents)))
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, Any] = {}
    consumed_prediction_sha256: dict[str, str] = {}
    batch_root = batch_root.resolve()
    for candidate_id in manifest["candidate_ids"]:
        expected_prediction_sha256 = inference["prediction_sha256"][candidate_id]
        prediction_payload = _read_exact_bytes(
            inference["prediction_paths"][candidate_id],
            expected_prediction_sha256,
            f"sealed prediction {candidate_id}",
        )
        prediction = np.load(io.BytesIO(prediction_payload), allow_pickle=False)
        if prediction.shape != (len(labelled_table.targets),) or prediction.dtype != np.bool_:
            raise ContractError("sealed prediction changed after the fuse")
        consumed_prediction_sha256[candidate_id] = hashlib.sha256(
            prediction_payload
        ).hexdigest()
        rows = _sealed_work_rows(labelled_table, prediction, selected)
        rows_by_candidate[candidate_id] = rows
        rows_path = batch_root / f"{candidate_id}.work_objectives.jsonl"
        _write_jsonl_exact_resume(rows_path, rows)
        by_source = {}
        for source in sorted(manifest["sealed_source_document_counts"]):
            subset = {
                index
                for index, document in enumerate(labelled_table.documents)
                if str(document["source"]) == source
            }
            by_source[source] = evaluate_prediction(
                labelled_table, prediction, document_subset=subset
            )
        metrics[candidate_id] = {
            "overall": evaluate_prediction(
                labelled_table, prediction, document_subset=selected
            ),
            "by_source": by_source,
            "work_objectives": {
                "path": str(rows_path),
                "sha256": sha256_file(rows_path),
                "work_count": len(rows),
            },
        }
    intervals = _simultaneous_sealed_intervals(
        rows_by_candidate, iterations=iterations, seed=seed
    )
    result = {
        "schema_version": SEALED_RESULTS_SCHEMA,
        "status": "passed_one_simultaneous_batch_all_pareto_candidates",
        "evaluation_fuse": {"path": str(fuse), "sha256": sha256_file(fuse)},
        "frozen_manifest_id": manifest["frozen_manifest_id"],
        "candidate_ids": list(manifest["candidate_ids"]),
        "candidate_count": len(manifest["candidate_ids"]),
        "document_count": 150,
        "source_document_counts": manifest["sealed_source_document_counts"],
        "line_count": len(line_rows),
        "unresolved_line_count": unresolved,
        "unresolved_metric_policy": "UNKNOWN is conservatively treated as NON_BIB",
        "metrics": metrics,
        "simultaneous_intervals": intervals,
        "inputs": {
            "manifest_sha256": sha256_file(manifest_path),
            "request_sha256": sha256_file(request_path),
            "inference_receipt_sha256": inference["receipt_sha256"],
            "feature_table_sha256": expected_table_sha256,
            "prediction_sha256": consumed_prediction_sha256,
            "labels_sha256": hashlib.sha256(labels_payload).hexdigest(),
            "consensus_receipt_sha256": hashlib.sha256(
                consensus_payload
            ).hexdigest(),
        },
    }
    result_path = batch_root / "sealed_results.json"
    if result_path.exists():
        if load_json(result_path) != result:
            raise ContractError("partial sealed result differs on exact resume")
    else:
        write_json_exclusive(result_path, result)
    completion = {
        "schema_version": SEALED_RESULTS_RECEIPT_SCHEMA,
        "status": "passed_complete_exact_sealed_result",
        "frozen_manifest_id": manifest["frozen_manifest_id"],
        "candidate_ids": manifest["candidate_ids"],
        "result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "fuse_sha256": sha256_file(fuse),
    }
    if completion_path.exists():
        if load_json(completion_path) != completion:
            raise ContractError("sealed result completion receipt differs on exact resume")
    else:
        write_json_exclusive(completion_path, completion)
    return result_path


def _make_input_row(
    path: Path,
    *,
    data_class: str,
    split: str,
    document_scope: str,
    contains_labels: bool,
    candidate_id: str | None = None,
    parent_candidate_id: str | None = None,
) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_dir():
        digest = sha256_directory(path)
        digest_kind = "recursive_tree_sha256_v1"
    elif path.is_file() and not path.is_symlink():
        digest = sha256_file(path)
        digest_kind = "file_sha256"
    else:
        raise ContractError(f"input is missing or a symlink: {path}")
    row: dict[str, Any] = {
        "path": str(path),
        "sha256": digest,
        "digest_kind": digest_kind,
        "data_class": data_class,
        "split": split,
        "document_scope": document_scope,
        "contains_labels": contains_labels,
    }
    if candidate_id is not None:
        row["candidate_id"] = candidate_id
    if parent_candidate_id is not None:
        row["parent_candidate_id"] = parent_candidate_id
    return row


def _make_parent_inputs(
    receipt_path: Path, *, prefix: str, receipt_only: bool
) -> dict[str, Any]:
    verification = verify_finalized_receipt(receipt_path)
    receipt_path = Path(receipt_path).resolve()
    receipt = load_json(receipt_path)
    candidate_id = verification["candidate_id"]
    result = {
        f"{prefix}_receipt": _make_input_row(
            receipt_path,
            data_class="parent_candidate_receipt",
            split="development",
            document_scope="aggregate_no_rows",
            contains_labels=False,
            candidate_id=candidate_id,
        )
    }
    if receipt_only:
        return result
    prediction = receipt.get("predictions", {}).get("main")
    if not isinstance(prediction, Mapping):
        raise ContractError("parent receipt has no main prediction")
    prediction_path = Path(str(prediction["path"]))
    if not prediction_path.is_absolute():
        prediction_path = receipt_path.parent / prediction_path
    barriers = [
        row for name, row in receipt.get("artifacts", {}).items()
        if str(name).endswith("combined_barriers.npz") and isinstance(row, Mapping)
    ]
    if len(barriers) != 1:
        raise ContractError("parent receipt must own exactly one combined_barriers.npz")
    barrier_path = Path(str(barriers[0]["path"]))
    if not barrier_path.is_absolute():
        barrier_path = receipt_path.parent / barrier_path
    result[f"{prefix}_prediction"] = _make_input_row(
        prediction_path,
        data_class="parent_prediction",
        split="development",
        document_scope="prediction_blind_extraction_qualified_268",
        contains_labels=False,
        parent_candidate_id=candidate_id,
    )
    result[f"{prefix}_barriers"] = _make_input_row(
        barrier_path,
        data_class="parent_barrier_artifact",
        split="development",
        document_scope="prediction_blind_extraction_qualified_268",
        contains_labels=False,
        parent_candidate_id=candidate_id,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-spec")
    validate.add_argument("--spec", type=Path, required=True)

    identify = sub.add_parser("candidate-id")
    identify.add_argument("--spec", type=Path, required=True)

    render = sub.add_parser("render-queue")
    render.add_argument("--templates", type=Path, required=True)
    render.add_argument("--generation", choices=[f"G{x}" for x in range(6)], required=True)
    render.add_argument("--bindings", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)

    initialize = sub.add_parser("init-candidate")
    initialize.add_argument("--spec", type=Path, required=True)
    initialize.add_argument("--leakage-policy", type=Path, required=True)
    initialize.add_argument("--candidate-root", type=Path, required=True)

    run = sub.add_parser("run-command")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--cwd", type=Path, required=True)

    execute = sub.add_parser("execute-queue-index")
    execute.add_argument("--queue", type=Path, required=True)
    execute.add_argument("--index", type=int, required=True)
    execute.add_argument("--leakage-policy", type=Path, required=True)
    execute.add_argument("--candidate-root", type=Path, required=True)
    execute.add_argument("--cwd", type=Path, required=True)

    finalize = sub.add_parser("finalize-candidate")
    finalize.add_argument("--candidate-dir", type=Path, required=True)
    finalize.add_argument("--candidate-root", type=Path, required=True)
    finalize.add_argument("--result", type=Path, required=True)

    prepare = sub.add_parser("prepare-result")
    prepare.add_argument("--candidate-dir", type=Path, required=True)
    prepare.add_argument("--backend-receipt", type=Path, required=True)
    prepare.add_argument("--prediction", type=Path, required=True)
    prepare.add_argument("--all-rows", type=Path, required=True)
    prepare.add_argument("--metrics-report", type=Path, required=True)
    prepare.add_argument("--paired-deltas", type=Path, required=True)
    prepare.add_argument("--tests", type=Path, required=True)
    prepare.add_argument("--reject-reason", action="append", default=[])
    prepare.add_argument("--output", type=Path, required=True)

    registry = sub.add_parser("build-registry")
    registry.add_argument("--candidate-root", type=Path, required=True)
    registry.add_argument("--output", type=Path, required=True)

    baseline = sub.add_parser("verify-g0")
    baseline.add_argument("--lock", type=Path, required=True)
    baseline.add_argument("--root", type=Path, required=True)
    baseline.add_argument("--replay-prediction", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)

    bootstrap = sub.add_parser("paired-bootstrap")
    bootstrap.add_argument("--candidate-rows", type=Path, required=True)
    bootstrap.add_argument("--baseline-rows", type=Path, required=True)
    bootstrap.add_argument("--iterations", type=int, default=2000)
    bootstrap.add_argument("--seed", type=int, default=20260718)
    bootstrap.add_argument("--output", type=Path, required=True)

    freeze = sub.add_parser("freeze-pareto")
    freeze.add_argument("--registry", type=Path, required=True)
    freeze.add_argument("--sealed-documents", type=Path, required=True)
    freeze.add_argument("--sealed-labels", type=Path, required=True)
    freeze.add_argument("--sealed-consensus-receipt", type=Path, required=True)
    freeze.add_argument("--sealed-freeze-receipt", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    infer = sub.add_parser(
        "prepare-sealed-inference",
        help="derive all frozen Pareto predictions before opening sealed labels",
    )
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--sealed-documents", type=Path, required=True)
    infer.add_argument("--sealed-freeze-receipt", type=Path, required=True)
    infer.add_argument("--output-root", type=Path, required=True)

    begin = sub.add_parser("begin-sealed-batch")
    begin.add_argument("--manifest", type=Path, required=True)
    begin.add_argument("--request", type=Path, required=True)
    begin.add_argument("--batch-root", type=Path, required=True)

    sealed = sub.add_parser("evaluate-sealed-batch")
    sealed.add_argument("--manifest", type=Path, required=True)
    sealed.add_argument("--request", type=Path, required=True)
    sealed.add_argument("--batch-root", type=Path, required=True)
    sealed.add_argument("--iterations", type=int, default=10000)
    sealed.add_argument("--seed", type=int, default=20260718)

    digest = sub.add_parser("hash-input")
    digest.add_argument("--path", type=Path, required=True)

    policy_digest = sub.add_parser("hash-policy")
    policy_digest.add_argument("--path", type=Path, required=True)

    input_row = sub.add_parser("make-input-row")
    input_row.add_argument("--path", type=Path, required=True)
    input_row.add_argument("--data-class", required=True)
    input_row.add_argument("--split", required=True)
    input_row.add_argument("--document-scope", required=True)
    input_row.add_argument("--contains-labels", action="store_true")
    input_row.add_argument("--candidate-id")
    input_row.add_argument("--parent-candidate-id")
    input_row.add_argument("--output", type=Path, required=True)

    parent_rows = sub.add_parser("make-parent-inputs")
    parent_rows.add_argument("--receipt", type=Path, required=True)
    parent_rows.add_argument("--prefix", required=True)
    parent_rows.add_argument("--receipt-only", action="store_true")
    parent_rows.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-spec":
        validate_candidate_spec(load_json(args.spec))
        return 0
    if args.command == "candidate-id":
        print(with_candidate_id(load_json(args.spec))["candidate_id"])
        return 0
    if args.command == "render-queue":
        packet = load_json(args.templates)
        bindings = load_json(args.bindings)
        templates = [row for row in packet["templates"] if row["generation"] == args.generation]
        if not templates:
            raise ContractError("no template exists for the requested generation")
        rows = []
        for template in templates:
            rows.extend(expand_template(template, bindings))
        _write_jsonl_exclusive(args.output, rows)
        return 0
    if args.command == "init-candidate":
        spec = load_json(args.spec)
        path = CandidateStore(args.candidate_root).create(spec, load_json(args.leakage_policy))
        print(path)
        return 0
    if args.command == "run-command":
        spec = load_json(args.spec)
        command = _validate_runner(spec)
        environment = os.environ.copy()
        environment["BIB_EVOLUTION_CANDIDATE_ID"] = spec["candidate_id"]
        completed = subprocess.run(command, cwd=args.cwd, env=environment, check=False)
        return int(completed.returncode)
    if args.command == "execute-queue-index":
        rows = _jsonl(args.queue)
        if args.index < 0 or args.index >= len(rows):
            raise ContractError("queue index is out of range")
        spec = rows[args.index]
        git_attestation = _attest_git_checkout(args.cwd, spec["code_commit"])
        store = CandidateStore(args.candidate_root)
        candidate_dir = store.create(spec, load_json(args.leakage_policy))
        command = _validate_runner(spec, candidate_dir=candidate_dir)
        environment = os.environ.copy()
        environment["BIB_EVOLUTION_CANDIDATE_ID"] = spec["candidate_id"]
        environment["BIB_EVOLUTION_CANDIDATE_DIR"] = str(candidate_dir)
        started = time.monotonic()
        with (candidate_dir / "stdout.log").open("x", encoding="utf-8") as stdout, (
            candidate_dir / "stderr.log"
        ).open("x", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=args.cwd,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        write_json_exclusive(
            candidate_dir / "execution.json",
            {
                "schema_version": "bibliography-evolution-execution-v1",
                "candidate_id": spec["candidate_id"],
                "queue_sha256": sha256_file(args.queue),
                "queue_index": args.index,
                "argv": command,
                "returncode": completed.returncode,
                "wall_seconds": time.monotonic() - started,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "git_attestation": git_attestation,
            },
        )
        if completed.returncode:
            print(candidate_dir)
            return int(completed.returncode)
        receipt = _auto_finalize_candidate(store, candidate_dir, spec)
        print(receipt)
        return 0
    if args.command == "finalize-candidate":
        path = CandidateStore(args.candidate_root).finalize(args.candidate_dir, load_json(args.result))
        print(path)
        return 0
    if args.command == "prepare-result":
        _prepare_result(args)
        return 0
    if args.command == "build-registry":
        receipts = list(args.candidate_root.glob("*/receipt.json"))
        write_json_exclusive(args.output, build_registry(receipts))
        return 0
    if args.command == "verify-g0":
        write_json_exclusive(
            args.output,
            verify_g0(load_json(args.lock), root=args.root, replay_prediction=args.replay_prediction),
        )
        return 0
    if args.command == "paired-bootstrap":
        write_json_exclusive(
            args.output,
            paired_work_bootstrap(
                _jsonl(args.candidate_rows),
                _jsonl(args.baseline_rows),
                iterations=args.iterations,
                seed=args.seed,
            ),
        )
        return 0
    if args.command == "freeze-pareto":
        _freeze_manifest(
            args.registry,
            args.output,
            args.sealed_documents,
            args.sealed_labels,
            args.sealed_consensus_receipt,
            args.sealed_freeze_receipt,
        )
        return 0
    if args.command == "prepare-sealed-inference":
        from .bibliography_evolution_sealed_inference import prepare_inference

        print(
            prepare_inference(
                args.manifest,
                args.sealed_documents,
                args.sealed_freeze_receipt,
                args.output_root,
            )
        )
        return 0
    if args.command == "begin-sealed-batch":
        _begin_sealed_batch(args.manifest, args.request, args.batch_root)
        return 0
    if args.command == "evaluate-sealed-batch":
        _evaluate_sealed_batch(
            args.manifest,
            args.request,
            args.batch_root,
            iterations=args.iterations,
            seed=args.seed,
        )
        return 0
    if args.command == "hash-input":
        path = args.path
        digest = sha256_directory(path) if path.is_dir() else sha256_file(path)
        print(
            json.dumps(
                {
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "digest_kind": "recursive_tree_sha256_v1" if path.is_dir() else "file_sha256",
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "hash-policy":
        policy = load_json(args.path)
        if policy.get("schema_version") != "bibliography-evolution-leakage-policy-v1":
            raise ContractError("unsupported leakage policy")
        print(hashlib.sha256(canonical_json_bytes(policy)).hexdigest())
        return 0
    if args.command == "make-input-row":
        write_json_exclusive(
            args.output,
            _make_input_row(
                args.path,
                data_class=args.data_class,
                split=args.split,
                document_scope=args.document_scope,
                contains_labels=args.contains_labels,
                candidate_id=args.candidate_id,
                parent_candidate_id=args.parent_candidate_id,
            ),
        )
        return 0
    if args.command == "make-parent-inputs":
        write_json_exclusive(
            args.output,
            _make_parent_inputs(
                args.receipt, prefix=args.prefix, receipt_only=args.receipt_only
            ),
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error

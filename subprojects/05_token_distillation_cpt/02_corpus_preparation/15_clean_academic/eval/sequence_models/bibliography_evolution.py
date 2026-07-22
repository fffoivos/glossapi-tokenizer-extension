#!/usr/bin/env python3
"""Command-line orchestration for controlled bibliography evolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bibliography_evolution_contract import (
    CandidateStore,
    ContractError,
    REGISTRY_SCHEMA,
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


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row).decode("utf-8"))


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
    paths = [Path(value).resolve() for value in (documents_path, labels_path, consensus_path, receipt_path)]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ContractError("sealed annotation artifact is missing or a symlink")
    documents_path, labels_path, consensus_path, receipt_path = paths
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
        or hashes.get("labels_sha256") != sha256_file(labels_path)
        or hashes.get("consensus_receipt_sha256") != sha256_file(consensus_path)
    ):
        raise ContractError("annotation-lane FROZEN receipt does not bind sealed bytes")
    return {
        "status": "passed",
        "document_ids": document_ids,
        "documents": {"path": str(documents_path), "sha256": sha256_file(documents_path)},
        "labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
        "consensus_receipt": {"path": str(consensus_path), "sha256": sha256_file(consensus_path)},
        "freeze_receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
        "source_document_counts": source_counts,
    }


def _freeze_manifest(
    registry_path: Path,
    output: Path,
    sealed_documents_path: Path,
    sealed_labels_path: Path,
    sealed_consensus_receipt_path: Path,
    sealed_freeze_receipt_path: Path,
) -> None:
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
    candidates = []
    for candidate_id in pareto_ids:
        row = indexed[candidate_id]
        if not row.get("pareto") or not row.get("eligible"):
            raise ContractError("frozen candidate is not eligible Pareto")
        receipt_path = Path(row["receipt_path"]).resolve()
        if sha256_file(receipt_path) != row["receipt_sha256"]:
            raise ContractError(f"candidate receipt changed: {candidate_id}")
        verification = verify_finalized_receipt(receipt_path)
        receipt = load_json(receipt_path)
        spec = load_json(receipt_path.parent / "spec.json")
        if verification["candidate_id"] != candidate_id:
            raise ContractError("candidate verification identity differs from registry")
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
    core = {
        "schema_version": SEALED_MANIFEST_SCHEMA,
        "status": "frozen_before_model_evaluation",
        "evaluation_mode": "one_simultaneous_batch_all_pareto_candidates",
        "incremental_evaluation_allowed": False,
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
    write_json_exclusive(output, core)


def _begin_sealed_batch(manifest_path: Path, request_path: Path, batch_root: Path) -> Path:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SEALED_MANIFEST_SCHEMA:
        raise ContractError("unsupported frozen Pareto manifest")
    frozen_id = manifest.get("frozen_manifest_id")
    core = {key: value for key, value in manifest.items() if key != "frozen_manifest_id"}
    recomputed = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    if frozen_id != recomputed:
        raise ContractError("frozen manifest ID does not match its current bytes")
    # Fail before creating a fuse or opening labels.  Development candidates
    # currently own validation-sized predictions, not a candidate-specific
    # inference recipe for the sealed feature table.  Accepting arbitrary
    # prediction files here would make the final comparison unauditable.
    raise ContractError(
        "sealed evaluation blocked: candidate-specific sealed inference bridge is not implemented"
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
    # Parameters remain in the CLI so a future implementation can be added
    # without silently changing the frozen interface.  The call always raises
    # before opening labels or creating a batch directory.
    _ = (iterations, seed)
    return _begin_sealed_batch(manifest_path, request_path, batch_root)


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

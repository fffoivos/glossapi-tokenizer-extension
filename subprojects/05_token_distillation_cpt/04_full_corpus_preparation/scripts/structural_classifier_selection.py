#!/usr/bin/env python3
"""Bind a joint ladder decision to the existing Rust C0 deployment artifacts.

The comparison ladder is research-only and never chooses a winner itself. This
bridge records an explicit operator choice after a completed joint run. Only C0
can pass today because it is the frozen LR+hysteresis implementation already
available in Rust. C1/C2/N1 require a separate reviewed Rust port, export and
runtime-parity package before this bridge may be extended to accept them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cleaning_runtime import (
    canonical_json_sha256,
    file_receipt,
    valid_sha256,
    verify_file_receipt,
    write_json_atomic,
)
from full_corpus_io import read_json_object, sha256_file


SCHEMA = "academic_structural_classifier_selection_v1"
RUN_SCHEMA = "academic-structure-bib-ladder-run-v1"
SOURCE_SCHEMA = "academic-structure-silver-contract-receipt-v1"
C0 = "c0-rust-lr-hysteresis"
RESEARCH_ONLY = {
    "c1-feature-bioes-crf",
    "c2-char-ngram-feature-bioes-crf",
    "n1-bytecnn-tcn-masked-crf",
}
EXPECTED_ARCHITECTURES = [C0, *sorted(RESEARCH_ONLY)]
EXPECTED_ARTIFACT_ROLES = {
    "selection_silver",
    "selection_manifest",
    "validation_silver",
    "selection_receipt",
    "c0_validation_predictions",
    "c0_receipt",
    "c1_model",
    "c1_validation_predictions",
    "c1_receipt",
    "c2_model",
    "c2_validation_predictions",
    "c2_receipt",
    "n1_profile_receipt",
    "n1_model",
    "n1_train_receipt",
    "n1_validation_predictions",
    "validation_report",
}


def _load(path: Path) -> dict[str, Any]:
    value = read_json_object(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _artifact_paths(run_root: Path, run: Mapping[str, Any]) -> dict[str, Path]:
    rows = run.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("joint ladder receipt has no artifact inventory")
    result: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("joint ladder artifact row is malformed")
        role = row.get("role")
        relative = row.get("relative_path")
        if (
            not isinstance(role, str)
            or not role
            or role in result
            or not isinstance(relative, str)
            or not relative
        ):
            raise ValueError("joint ladder artifact role/path is malformed or duplicated")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("joint ladder artifact escapes its run root")
        path = (run_root / relative_path).resolve()
        if run_root not in path.parents or path in seen_paths:
            raise ValueError("joint ladder artifact path is outside/duplicated")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"joint ladder artifact is absent or a symlink: {role}")
        if (
            row.get("sha256") != sha256_file(path)
            or row.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"joint ladder artifact differs from its receipt: {role}")
        result[role] = path
        seen_paths.add(path)
    discovered = {
        path.resolve()
        for path in run_root.rglob("*")
        if path.is_file() and path.name != "run.receipt.json"
    }
    if any(path.is_symlink() for path in run_root.rglob("*")) or discovered != seen_paths:
        raise ValueError("joint ladder run has unreceipted, missing or symlinked files")
    if (
        set(result) != EXPECTED_ARTIFACT_ROLES
        or run.get("artifact_inventory_sha256") != canonical_json_sha256(rows)
    ):
        raise ValueError("joint ladder artifact role/inventory contract drift")
    return result


def _validate_joint_source(
    source_path: Path,
    split_path: Path,
    sequence_config_path: Path,
) -> dict[str, Any]:
    source = _load(source_path)
    split = _load(split_path)
    config = _load(sequence_config_path)
    expected_train = config.get("historical_partition_usage", {}).get(
        "historical_train_document_count"
    )
    expected_test = config.get("historical_partition_usage", {}).get(
        "historical_test_document_count"
    )
    exclusion = source.get("historical_partition_exclusion")
    scopes = source.get("task_scope_counts")
    assignments = split.get("assignments")
    assignment_counts = (
        {
            name: sum(value == name for value in assignments.values())
            for name in ("train", "validation")
        }
        if isinstance(assignments, Mapping)
        else None
    )
    materialized = source.get("materialized_artifacts")
    if (
        config.get("schema_version") != "academic-structure-sequence-eval-v1"
        or config.get("active_classes") != ["BIB", "TOC"]
        or config.get("split", {}).get("test_fraction") != 0.0
        or source.get("schema_version") != SOURCE_SCHEMA
        or source.get("status") != "pass"
        or source.get("evidence_tier") != "LLM_silver"
        or source.get("production_eligible") is not False
        or source.get("sequence_fit_eligible") is not True
        or source.get("config_sha256") != sha256_file(sequence_config_path)
        or not valid_sha256(source.get("silver_sha256"))
        or source.get("split_manifest_sha256") != sha256_file(split_path)
        or not isinstance(scopes, Mapping)
        or int(scopes.get("bibliography_toc_windows", 0)) <= 0
        or not isinstance(exclusion, Mapping)
        or exclusion.get("status") != "passed_before_materialization"
        or exclusion.get("eligible_historical_train_documents") != expected_train
        or exclusion.get("historical_test_documents_excluded") != expected_test
        or exclusion.get("historical_test_rows_emitted") != 0
        or exclusion.get("historical_test_predictions_permitted") is not False
        or source.get("document_count") != expected_train
        or split.get("schema_version") != "academic-structure-split-v1"
        or split.get("inventory_sha256") != source.get("inventory_sha256")
        or not isinstance(assignments, Mapping)
        or len(assignments) != expected_train
        or set(assignments.values()) != {"train", "validation"}
        or source.get("split_counts") != assignment_counts
        or not isinstance(materialized, Mapping)
        or materialized.get("split_manifest_filename") != split_path.name
        or source_path.resolve().parent != split_path.resolve().parent
    ):
        raise ValueError(
            "joint source does not prove the receipt-bound 1,392/608 historical split boundary"
        )
    return source


def _validate_ladder(
    run_root: Path,
    *,
    source_path: Path,
    split_path: Path,
    sequence_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    raw_root = Path(run_root)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("joint ladder root must be a real directory")
    root = raw_root.resolve()
    receipt_path = root / "run.receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("joint ladder run receipt is absent")
    run = _load(receipt_path)
    source = _validate_joint_source(source_path, split_path, sequence_config_path)
    expected_source = {
        "silver_sha256": source["silver_sha256"],
        "split_manifest_sha256": source["split_manifest_sha256"],
        "rehydration_receipt_sha256": sha256_file(source_path),
    }
    if (
        run.get("schema_version") != RUN_SCHEMA
        or run.get("status") != "passed_cpu_sealed_retrospective_comparison"
        or run.get("target") != "BIB+TOC"
        or run.get("active_classes") != ["BIB", "TOC"]
        or run.get("architecture_ids") != EXPECTED_ARCHITECTURES
        or run.get("config_sha256") != sha256_file(sequence_config_path)
        or run.get("production_eligible") is not False
        or run.get("decision")
        != "LLM_silver_replay_only_no_automatic_selection_no_production_change"
        or any(run.get("source", {}).get(key) != value for key, value in expected_source.items())
        or run.get("historically_named_test_partition", {}).get(
            "documents_loaded_by_model_or_validation_processes"
        )
        != 0
        or run.get("historically_named_test_partition", {}).get("predictions_written") != 0
    ):
        raise ValueError("joint ladder receipt does not prove the required comparison boundary")
    return run, _artifact_paths(root, run)


def _expected_models(bib_model: Path, toc_model: Path, smoother: Path) -> dict[str, str]:
    return {
        bib_model.name: sha256_file(bib_model),
        toc_model.name: sha256_file(toc_model),
        smoother.name: sha256_file(smoother),
    }


def build_selection(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite classifier selection: {args.output}")
    if args.selected_architecture in RESEARCH_ONLY:
        raise ValueError(
            f"{args.selected_architecture} is a Python research arm, not a Rust deployment; "
            "a separate reviewed port/export plus exact Rust parity package is required"
        )
    if args.selected_architecture != C0:
        raise ValueError(f"unsupported classifier architecture: {args.selected_architecture}")
    if not args.selection_rationale.strip():
        raise ValueError("operator selection rationale is required")
    run, artifacts = _validate_ladder(
        args.ladder_run_root,
        source_path=args.source_receipt,
        split_path=args.source_split_manifest,
        sequence_config_path=args.sequence_config,
    )
    c0_path = artifacts.get("c0_receipt")
    if c0_path is None:
        raise ValueError("joint ladder lacks the C0 arm receipt")
    c0 = _load(c0_path)
    expected_models = _expected_models(args.bib_model, args.toc_model, args.smoother)
    if (
        c0.get("schema_version") != "academic-structure-c0-reference-v2"
        or c0.get("status") != "passed_descriptive_reference_prediction"
        or c0.get("architecture_id") != C0
        or c0.get("target") != "BIB+TOC"
        or c0.get("active_classes") != ["BIB", "TOC"]
        or c0.get("comparison_role") != "historical_reference_only"
        or c0.get("production_eligible") is not False
        or c0.get("model_artifacts") != expected_models
    ):
        raise ValueError("joint ladder C0 receipt does not bind the deployed model artifacts")
    source = _load(args.source_receipt)
    receipt = {
        "schema_version": SCHEMA,
        "status": "passed_c0_runtime_compatibility_selection",
        "selected_architecture": C0,
        "selection_method": "explicit_operator_choice_after_receipt_bound_joint_ladder",
        "selection_rationale": args.selection_rationale.strip(),
        "automatic_winner_selection": False,
        "evidence_tier": "LLM_silver",
        "joint_ladder": {
            "run_receipt": file_receipt(args.ladder_run_root / "run.receipt.json"),
            "run_receipt_sha256": sha256_file(args.ladder_run_root / "run.receipt.json"),
            "target": run["target"],
            "active_classes": run["active_classes"],
            "c0_arm_receipt": file_receipt(c0_path),
        },
        "joint_source": {
            "receipt": file_receipt(args.source_receipt),
            "split_manifest": file_receipt(args.source_split_manifest),
            "inventory_sha256": source["inventory_sha256"],
            "historical_train_documents": source["document_count"],
            "historical_test_documents_excluded": source[
                "historical_partition_exclusion"
            ]["historical_test_documents_excluded"],
        },
        "deployment_artifacts": {
            "sequence_config": file_receipt(args.sequence_config),
            "bibliography_line_model": file_receipt(args.bib_model),
            "toc_line_model": file_receipt(args.toc_model),
            "structural_smoother": file_receipt(args.smoother),
        },
        "runtime_compatibility": {
            "status": "existing_rust_c0_implementation_requires_fresh_exact_parity",
            "rust_deployable": True,
            "port_package_required": False,
            "parity_receipt_created": False,
        },
        "stage52_candidate_eligible": True,
        "stage54_model_selection_eligible": True,
        "structural_application_eligible": False,
        "remaining_gates": [
            "fresh_exact_rust_parity_on_imported_joint_source",
            "stage52_predictions_for_this_exact_artifact_bundle",
            "manual_100_case_false_deletion_audit",
            "approved_policy_frozen_before_stage10",
        ],
    }
    write_json_atomic(args.output, receipt)
    return validate_selection(
        args.output,
        source_receipt=args.source_receipt,
        source_split_manifest=args.source_split_manifest,
        sequence_config=args.sequence_config,
        bib_model=args.bib_model,
        toc_model=args.toc_model,
        smoother=args.smoother,
    )


def validate_selection(
    path: Path,
    *,
    source_receipt: Path,
    source_split_manifest: Path,
    sequence_config: Path,
    bib_model: Path,
    toc_model: Path,
    smoother: Path,
) -> dict[str, Any]:
    receipt = _load(path)
    if (
        receipt.get("schema_version") != SCHEMA
        or receipt.get("status") != "passed_c0_runtime_compatibility_selection"
        or receipt.get("selected_architecture") != C0
        or receipt.get("selection_method")
        != "explicit_operator_choice_after_receipt_bound_joint_ladder"
        or receipt.get("automatic_winner_selection") is not False
        or receipt.get("evidence_tier") != "LLM_silver"
        or receipt.get("stage52_candidate_eligible") is not True
        or receipt.get("stage54_model_selection_eligible") is not True
        or receipt.get("structural_application_eligible") is not False
        or not isinstance(receipt.get("selection_rationale"), str)
        or not receipt["selection_rationale"]
    ):
        raise ValueError("classifier selection receipt status/decision boundary is invalid")
    joint = receipt.get("joint_ladder")
    source = receipt.get("joint_source")
    deployment = receipt.get("deployment_artifacts")
    runtime = receipt.get("runtime_compatibility")
    if not all(isinstance(value, Mapping) for value in (joint, source, deployment, runtime)):
        raise ValueError("classifier selection receipt sections are malformed")
    run_path = verify_file_receipt(joint["run_receipt"])
    run_root = run_path.parent
    run, artifacts = _validate_ladder(
        run_root,
        source_path=source_receipt,
        split_path=source_split_manifest,
        sequence_config_path=sequence_config,
    )
    if (
        run_path != (run_root / "run.receipt.json").resolve()
        or joint.get("run_receipt_sha256") != sha256_file(run_path)
        or joint.get("target") != run["target"]
        or joint.get("active_classes") != run["active_classes"]
        or verify_file_receipt(joint["c0_arm_receipt"]) != artifacts.get("c0_receipt")
    ):
        raise ValueError("classifier selection ladder binding drift")
    expected_source = _validate_joint_source(
        source_receipt, source_split_manifest, sequence_config
    )
    if (
        verify_file_receipt(source["receipt"]) != source_receipt.resolve()
        or verify_file_receipt(source["split_manifest"])
        != source_split_manifest.resolve()
        or source.get("inventory_sha256") != expected_source["inventory_sha256"]
        or source.get("historical_train_documents") != expected_source["document_count"]
        or source.get("historical_test_documents_excluded")
        != expected_source["historical_partition_exclusion"][
            "historical_test_documents_excluded"
        ]
    ):
        raise ValueError("classifier selection joint-source binding drift")
    expected_paths = {
        "sequence_config": sequence_config,
        "bibliography_line_model": bib_model,
        "toc_line_model": toc_model,
        "structural_smoother": smoother,
    }
    if any(
        verify_file_receipt(deployment[name]) != expected.resolve()
        for name, expected in expected_paths.items()
    ):
        raise ValueError("classifier selection deployment artifact binding drift")
    c0 = _load(artifacts["c0_receipt"])
    if c0.get("model_artifacts") != _expected_models(bib_model, toc_model, smoother):
        raise ValueError("classifier selection C0 model hash drift")
    if runtime != {
        "status": "existing_rust_c0_implementation_requires_fresh_exact_parity",
        "rust_deployable": True,
        "port_package_required": False,
        "parity_receipt_created": False,
    }:
        raise ValueError("classifier selection runtime-compatibility claim drift")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--ladder-run-root", type=Path, required=True)
    build.add_argument("--source-receipt", type=Path, required=True)
    build.add_argument("--source-split-manifest", type=Path, required=True)
    build.add_argument("--sequence-config", type=Path, required=True)
    build.add_argument("--bib-model", type=Path, required=True)
    build.add_argument("--toc-model", type=Path, required=True)
    build.add_argument("--smoother", type=Path, required=True)
    build.add_argument("--selected-architecture", required=True)
    build.add_argument("--selection-rationale", required=True)
    build.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    for command in (validate,):
        command.add_argument("--source-receipt", type=Path, required=True)
        command.add_argument("--source-split-manifest", type=Path, required=True)
        command.add_argument("--sequence-config", type=Path, required=True)
        command.add_argument("--bib-model", type=Path, required=True)
        command.add_argument("--toc-model", type=Path, required=True)
        command.add_argument("--smoother", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        receipt = build_selection(args)
    else:
        receipt = validate_selection(
            args.receipt,
            source_receipt=args.source_receipt,
            source_split_manifest=args.source_split_manifest,
            sequence_config=args.sequence_config,
            bib_model=args.bib_model,
            toc_model=args.toc_model,
            smoother=args.smoother,
        )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

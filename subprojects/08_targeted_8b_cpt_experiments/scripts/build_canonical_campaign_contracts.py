#!/usr/bin/env python3
"""Compile the matched hard-H-to-G study into canonical campaign-v3 inputs.

This is a scientific adapter only.  Submission, retries, holders, checkpoint
permits, evaluation dispatch and finalization remain owned by the canonical
``apertus-cscs-efficiency`` runner.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

from contract_utils import file_binding, read_json, require, write_json_atomic
from freeze_online_validation_binaries import NAMES as ONLINE_PANEL_NAMES
from freeze_online_validation_binaries import NEW_GREEK as ONLINE_NEW_GREEK
from run_canonical_train_segment import LATE_BOUND_PHASE_CACHE

BOUNDARIES = (0, 952, 1904, 2261, 3218, 3456, 3694)
MILESTONES = (
    0,
    238,
    476,
    714,
    952,
    1190,
    1428,
    1666,
    1904,
    2142,
    2261,
    2380,
    2618,
    2856,
    3094,
    3218,
    3456,
    3694,
)
NATIVE_MILESTONES = (0, 2261, 2618, 3218, 3456, 3694)
GLOBAL_BATCH_TOKEN_SLOTS = 4_194_304
UENV_IMAGE = "pytorch/v2.9.1:v2"
UENV_DIGEST = "c05f143d5fbf092714650697c821ee9fcf3d00c3580d56dabeacb2370a824b0f"
PROFILE_CHECKS = (
    "fixed_batch_loss_parity",
    "fixed_batch_gradient_parity",
    "restart_next_step_parity",
    "phase2_entry_and_restart_parity",
    "sample_and_mask_cursor_continuity",
    "zero_skipped_or_nonfinite_updates",
    "tokens_per_gpu_hour_measured",
)


def immutable_input(item_id: str, path: Path) -> dict[str, Any]:
    return {"id": item_id, **file_binding(path), "verify_at_submit": True}


def portable_binding(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            "training-data manifest must be inside the campaign root"
        ) from exc
    require(
        ".." not in relative.parts and not relative.is_absolute(),
        "unsafe manifest binding",
    )
    value = file_binding(resolved)
    value["path"] = str(relative)
    return value


def phase_for_start(start: int) -> int:
    if 0 <= start < 2261:
        return 1
    if 2261 <= start < 3218:
        return 2
    if 3218 <= start < 3694:
        return 3
    raise ValueError(f"invalid segment start: {start}")


def read_phase_asset(receipt_path: Path, phase: int) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    require(
        receipt.get("schema_version") == "apertus_hard_h_to_g_phase_blend_cache_v1"
        and receipt.get("status") == "frozen"
        and int(receipt.get("phase", -1)) == phase,
        f"Phase-{phase} cache receipt drift",
    )
    spec_binding = receipt.get("data_path_spec")
    require(isinstance(spec_binding, dict), f"Phase-{phase} data-path binding missing")
    spec = Path(str(spec_binding.get("path", ""))).resolve()
    require(
        spec.is_file() and spec_binding == file_binding(spec),
        f"Phase-{phase} data-path spec drift",
    )
    cache_root = Path(str(receipt.get("cache_root", ""))).resolve()
    require(cache_root.is_dir(), f"Phase-{phase} cache root missing")
    tokens = receipt.get("data_path_tokens")
    require(
        isinstance(tokens, list)
        and bool(tokens)
        and all(isinstance(token, str) and token for token in tokens),
        f"Phase-{phase} data-path tokens missing",
    )
    cache_sha256 = str(receipt.get("cache_tree_sha256", ""))
    require(len(cache_sha256) == 64, f"Phase-{phase} cache hash missing")
    return {
        "phase": phase,
        "spec": spec,
        "receipt": receipt_path.resolve(),
        "root": cache_root,
        "data_path": shlex.join(tokens),
        "cache_sha256": cache_sha256,
    }


def late_phase3_asset(
    *, spec: Path, receipt: Path, root: Path, stage_root: Path
) -> dict[str, Any]:
    resolved_stage = stage_root.resolve()
    result = {
        "phase": 3,
        "spec": spec.resolve(),
        "receipt": receipt.resolve(),
        "root": root.resolve(),
        "data_path": LATE_BOUND_PHASE_CACHE,
        "cache_sha256": LATE_BOUND_PHASE_CACHE,
    }
    for key in ("spec", "receipt", "root"):
        try:
            result[key].relative_to(resolved_stage)
        except ValueError as exc:
            raise ValueError(
                f"future Phase-3 {key} path escapes the stage root"
            ) from exc
    return result


def build_runtime(scale: str, run_permit: dict[str, Any]) -> dict[str, Any]:
    profile = run_permit["profile"]
    nodes = int(profile["nodes"])
    return {
        "schema_version": "apertus_runtime_profile_v2",
        "profile_id": f"hard_h2g_{scale}_{profile['profile_id']}",
        "profile_kind": "production",
        "status": "candidate",
        "slurm": {
            "resource_class": "normal",
            "account": "a0140",
            "partition": "normal",
            "nodes": nodes,
            "tasks_per_node": 4,
            "gpus_per_node": 4,
            "cpus_per_task": 72,
            "memory": "450G",
            "time_limit_seconds": 43_200,
        },
        "parallelism": {
            "tensor": int(profile["tensor_parallel"]),
            "pipeline": int(profile["pipeline_parallel"]),
            "context": int(profile["context_parallel"]),
            "data": int(profile["data_parallel"]),
        },
        "uenv": {
            "image": UENV_IMAGE,
            "view": "default",
            "image_digest": UENV_DIGEST,
        },
        "signal_lead_seconds": 600,
        "evidence_mirror_root": "/users/fffoivos/apertus-campaign-evidence",
        "submission_policy": {
            "policy_id": f"hard_h2g_{scale}_bounded_campaign",
            "allowed_partitions": ["normal", "debug"],
            "max_nodes": max(nodes, 4),
            "max_running_jobs": 3,
            "max_submitted_jobs": 4,
            "pending_escalation_seconds": 86_400,
            "forbid_normal": False,
        },
    }


def build_qualification_gates(
    *, code_receipt: Path, phase1_cache_receipt: Path
) -> list[dict[str, Any]]:
    """Bind the exact-profile receipt's predeclared scientific checks.

    The profile finalizer evaluates the experiment's frozen numerical
    tolerances from machine-written logs.  The canonical gate layer verifies
    those immutable boolean results; it must not reinterpret the same evidence
    as bitwise equality after the profile has run.
    """

    gates: list[dict[str, Any]] = [
        {
            "id": "scientific_code_binding",
            "type": "file_binding",
            "qualified": True,
            "binding": file_binding(code_receipt),
        },
        {
            "id": "phase1_cache_binding",
            "type": "file_binding",
            "qualified": True,
            "binding": file_binding(phase1_cache_receipt),
        },
        {
            "id": "profile_receipt_passed",
            "type": "receipt_status",
            "qualified": True,
            "value": {"evidence_id": "profile_measurement", "field": "status"},
            "accepted": ["passed"],
        },
    ]
    gates.extend(
        {
            "id": check,
            "type": "json_field_equal",
            "qualified": True,
            "left": {
                "evidence_id": "profile_measurement",
                "field": f"checks.{check}",
            },
            "right": {"literal": True},
        }
        for check in PROFILE_CHECKS
    )
    gates.append(
        {
            "id": "exact_slurm_profile",
            "type": "slurm_job_binding",
            "qualified": True,
            "evidence_id": "slurm_profile",
            "checks_field": "checks",
        }
    )
    return gates


def evaluation_profile(
    *, nodes: int, cpus_per_task: int, memory: str
) -> dict[str, Any]:
    return {
        "resource_class": "debug",
        "account": "a0140",
        "partition": "debug",
        "nodes": nodes,
        "tasks_per_node": 4 if nodes > 1 else 1,
        "gpus_per_node": 4,
        "cpus_per_task": cpus_per_task,
        "memory": memory,
        "time_limit_seconds": 5400,
    }


def evaluator_row(
    evaluator_id: str,
    argv: list[str],
    milestones: tuple[int, ...],
    profile_id: str,
    schema: str,
    release_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": evaluator_id,
        "argv": argv,
        "milestone_iterations": list(milestones),
        "resource_profile_id": profile_id,
        "max_attempts": 2,
        "expected_receipt_schema": schema,
    }
    if release_gate is not None:
        row["release_gate"] = release_gate
    return row


def build_evaluation_plan(scale: str, paths: dict[str, Path]) -> dict[str, Any]:
    common = [
        "--scale",
        scale,
        "--iteration",
        "{iteration}",
        "--run-root",
        "{run_root}",
        "--output",
        "{evaluation_root}",
        "--code-root",
        str(paths["code_root"]),
        "--code-receipt",
        str(paths["code_receipt"]),
        "--campaign-id",
        "{campaign_id}",
        "--evaluator-id",
        "{evaluator_id}",
        "--attempt",
        "{attempt_id}",
        "--contract-digest",
        "{contract_digest}",
    ]
    python = "/usr/bin/python3.11"
    evaluation_root = (
        paths["code_root"] / "subprojects/08_targeted_8b_cpt_experiments/evaluation"
    )
    rows = [
        evaluator_row(
            "checkpoint_export",
            [
                python,
                str(evaluation_root / "run_checkpoint_export_evaluator.py"),
                *common,
                "--initial-checkpoint-root",
                str(paths["initial_checkpoint_root"]),
                "--megatron-root",
                str(paths["megatron_root"]),
                "--tokenizer-dir",
                str(paths["tokenizer_dir"]),
                "--model-contract",
                str(paths["experiment"]),
                "--eval-python",
                str(paths["eval_venv"] / "bin/python"),
            ],
            MILESTONES,
            "debug_export_one_node",
            "apertus_hard_h_to_g_checkpoint_export_evaluation_v1",
        ),
        evaluator_row(
            "offline_panels",
            [
                python,
                str(evaluation_root / "run_offline_panels_evaluator.py"),
                *common,
                "--validation-manifest",
                str(paths["validation_manifest"]),
                "--tokenizer-dir",
                str(paths["tokenizer_dir"]),
            ],
            MILESTONES,
            "debug_four_node",
            "apertus_hard_h_to_g_offline_panels_evaluation_v1",
        ),
        evaluator_row(
            "greekmmlu",
            [
                python,
                str(evaluation_root / "run_greekmmlu_evaluator.py"),
                *common,
                "--clean-examples",
                str(paths["clean_examples"]),
                "--sentinel-manifest",
                str(paths["sentinel_manifest"]),
                "--eval-venv",
                str(paths["eval_venv"]),
            ],
            MILESTONES,
            "debug_four_node",
            "apertus_hard_h_to_g_greekmmlu_evaluation_v1",
        ),
        evaluator_row(
            "native_greek_suite",
            [
                python,
                str(evaluation_root / "run_native_suite_evaluator.py"),
                *common,
                "--eval-code-root",
                str(paths["native_eval_code_root"]),
                "--eval-code-receipt",
                str(paths["native_eval_code_receipt"]),
                "--source-contract",
                str(paths["native_source_contract"]),
                "--source-manifest",
                str(paths["native_source_manifest"]),
                "--source-execution-gate",
                str(paths["native_source_gate"]),
                "--eval-venv",
                str(paths["eval_venv"]),
                "--exclusions",
                str(paths["native_exclusions"]),
            ],
            NATIVE_MILESTONES,
            "debug_four_node",
            "apertus_hard_h_to_g_native_suite_evaluation_v1",
        ),
    ]
    if scale == "8b":
        rows.append(
            evaluator_row(
                "legacy_public_greekmmlu",
                [
                    python,
                    str(evaluation_root / "run_legacy_public_greekmmlu_evaluator.py"),
                    *common,
                    "--legacy-contract",
                    str(paths["legacy_contract"]),
                    "--eval-venv",
                    str(paths["eval_venv"]),
                ],
                (3218,),
                "debug_one_node",
                "apertus_hard_h_to_g_legacy_public_evaluation_v1",
            )
        )
    rows.append(
        evaluator_row(
            "greekmmlu_calibration",
            [
                python,
                str(evaluation_root / "run_greekmmlu_calibration_evaluator.py"),
                *common,
                "--sentinel-manifest",
                str(paths["sentinel_manifest"]),
                "--eval-venv",
                str(paths["eval_venv"]),
            ],
            (3218,),
            "debug_one_node",
            "apertus_hard_h_to_g_greekmmlu_calibration_evaluation_v1",
        )
    )
    joint_calibration_release = {
        "path": str(
            paths["stage_root"]
            / "receipts/greekmmlu_sentinel_calibration_authority.json"
        ),
        "expected_schema": "apertus_greekmmlu_sentinel_calibration_authority_v1",
        "accepted_statuses": ["passed"],
        "field_equals": {"scope": "both_scales"},
    }
    for target in (952, 1190, 1428, 1666, 1904, 2142, 2261, 2380, 3456):
        rows.append(
            evaluator_row(
                f"greekmmlu_full_fallback_i{target:07d}",
                [
                    python,
                    str(evaluation_root / "run_greekmmlu_fallback_evaluator.py"),
                    *common,
                    "--target-iteration",
                    str(target),
                    "--clean-examples",
                    str(paths["clean_examples"]),
                    "--sentinel-manifest",
                    str(paths["sentinel_manifest"]),
                    "--eval-venv",
                    str(paths["eval_venv"]),
                    "--joint-calibration-authority",
                    str(
                        paths["stage_root"]
                        / "receipts/greekmmlu_sentinel_calibration_authority.json"
                    ),
                ],
                (max(3218, target),),
                "debug_four_node",
                "apertus_hard_h_to_g_greekmmlu_fallback_evaluation_v1",
                joint_calibration_release,
            )
        )
    rows.append(
        evaluator_row(
            "greekmmlu_plateau_confirmation",
            [
                python,
                str(evaluation_root / "run_greekmmlu_plateau_evaluator.py"),
                *common,
                "--clean-examples",
                str(paths["clean_examples"]),
                "--sentinel-manifest",
                str(paths["sentinel_manifest"]),
                "--eval-venv",
                str(paths["eval_venv"]),
                "--joint-calibration-authority",
                str(
                    paths["stage_root"]
                    / "receipts/greekmmlu_sentinel_calibration_authority.json"
                ),
            ],
            (3694,),
            "debug_four_node",
            "apertus_hard_h_to_g_greekmmlu_plateau_evaluation_v1",
        )
    )
    return {
        "schema_version": "apertus_evaluation_plan_v2",
        "plan_id": f"hard_h2g_{scale}_benchmark_clean",
        "resource_profiles": {
            "debug_export_one_node": evaluation_profile(
                nodes=1, cpus_per_task=72, memory="220G"
            ),
            "debug_one_node": evaluation_profile(
                nodes=1, cpus_per_task=72, memory="220G"
            ),
            "debug_four_node": evaluation_profile(
                nodes=4, cpus_per_task=54, memory="640G"
            ),
        },
        "evaluators": rows,
    }


def build_segments(
    *,
    scale: str,
    initialization_root: Path,
    initialization_permit: Path,
    phase_assets: dict[int, dict[str, Any]],
    stage_root: Path,
    timing: dict[str, Any],
    schedule: dict[str, Any],
    pre_main_launch_gate: Path,
) -> list[dict[str, Any]]:
    scale_timing = timing["scales"][scale]
    timing_rows = {int(row["segment_id"]): row for row in scale_timing["segments"]}
    successor_rows = {
        int(row["target_segment_id"]): row
        for row in schedule["successors"]
        if row["scale"] == scale
    }
    require(set(timing_rows) == set(range(6)), "timing segment inventory drift")
    require(
        set(successor_rows) == set(range(1, 6)), "allocation successor inventory drift"
    )
    result = []
    for segment_id, (start, end) in enumerate(pairwise(BOUNDARIES)):
        phase = phase_for_start(start)
        asset = phase_assets[phase]
        gate = (
            pre_main_launch_gate
            if phase < 3
            else stage_root / "receipts/launch_gate_pre_extension.json"
            if start < 3456
            else stage_root / "receipts/launch_gate_pre_second_extension.json"
        )
        overrides = [
            "--phase",
            str(phase),
            "--phase-data-path-spec",
            str(asset["spec"]),
            "--phase-data-path",
            str(asset["data_path"]),
            "--phase-cache-receipt",
            str(asset["receipt"]),
            "--phase-cache-root",
            str(asset["root"]),
            "--phase-cache-tree-sha256",
            str(asset["cache_sha256"]),
            "--authorization-gate",
            str(gate),
        ]
        if segment_id == 0:
            overrides.extend(("--initialization-permit", str(initialization_permit)))
        timing_row = timing_rows[segment_id]
        segment: dict[str, Any] = {
            "id": f"s{segment_id}",
            "start_iteration": start,
            "end_iteration": end,
            "load_checkpoint": (
                str(initialization_root) if segment_id == 0 else "{previous_checkpoint}"
            ),
            "minimum_train_seconds": int(timing_row["minimum_train_seconds"]),
            "max_attempts": 3,
            "completion": {
                "receipt_path": "{attempt_root}/completion.json",
                "expected_schema": "apertus_hard_h_to_g_training_completion_v1",
                "checkpoint_required": True,
            },
            "argv_overrides": overrides,
        }
        if phase == 3:
            segment["release_gate"] = {
                "path": str(gate),
                "expected_schema": "apertus_hard_h_to_g_frozen_contract_v2",
                "accepted_statuses": ["launch_ready"],
                "field_equals": {
                    "gate_stage": (
                        "pre_extension" if start < 3456 else "pre_second_extension"
                    )
                },
            }
        if segment_id:
            successor = successor_rows[segment_id]
            source_timing = timing_rows[segment_id - 1]
            require(
                int(successor["target_runtime_seconds"])
                == int(timing_row["conservative_wall_seconds"]),
                f"segment {segment_id}: target timing drift",
            )
            segment["handoff"] = {
                "allocation_seconds": int(schedule["allocation_seconds"]),
                "reserve_seconds": int(successor["reserve_seconds"]),
                "maximum_hold_seconds": int(successor["maximum_hold_seconds"]),
                "conservative_target_runtime_seconds": int(
                    successor["target_runtime_seconds"]
                ),
                "conservative_source_runtime_seconds": int(
                    source_timing["conservative_wall_seconds"]
                ),
                "source_trigger_seconds": int(successor["source_trigger_seconds"]),
            }
        result.append(segment)
    return result


def build_campaign(
    *,
    scale: str,
    paths: dict[str, Path],
    experiment: dict[str, Any],
    run_permit: dict[str, Any],
    data_manifest: dict[str, Any],
    runtime: dict[str, Any],
    evaluation: dict[str, Any],
    phase_assets: dict[int, dict[str, Any]],
    timing: dict[str, Any],
    schedule: dict[str, Any],
) -> dict[str, Any]:
    model = experiment["models"][scale]
    tokenizer = experiment["tokenizer"]
    megatron = read_json(paths["megatron_receipt"])
    initialization = file_binding(paths["initialization_permit"])
    immutable_paths = {
        "scientific_code_receipt": paths["code_receipt"],
        "experiment_contract": paths["experiment"],
        "allocation_contract": paths["allocation"],
        "initialization_permit": paths["initialization_permit"],
        "training_run_permit": paths["run_permit"],
        "profile_promotion": paths["profile_promotion"],
        "production_timing": paths["timing"],
        "allocation_schedule": paths["schedule"],
        "phase1_data_spec": phase_assets[1]["spec"],
        "phase1_cache_receipt": phase_assets[1]["receipt"],
        "phase2_data_spec": phase_assets[2]["spec"],
        "phase2_cache_receipt": phase_assets[2]["receipt"],
        "megatron_receipt": paths["megatron_receipt"],
        "online_validation_receipt": paths["validation_receipt"],
        "tokenizer_receipt": paths["tokenizer_receipt"],
        "validation_manifest": paths["validation_manifest"],
        "clean_greekmmlu_examples": paths["clean_examples"],
        "greekmmlu_sentinel_manifest": paths["sentinel_manifest"],
        "statistical_decision_contract": paths["statistical_contract"],
        "qualification_contract": paths["qualification_contract"],
        "producer_compatibility": paths["producer_compatibility"],
        "native_eval_code_receipt": paths["native_eval_code_receipt"],
        "native_source_contract": paths["native_source_contract"],
        "native_source_manifest": paths["native_source_manifest"],
        "native_source_gate": paths["native_source_gate"],
        "native_exclusions": paths["native_exclusions"],
    }
    if scale == "8b":
        immutable_paths["legacy_public_contract"] = paths["legacy_contract"]
    common_train_argv = [
        "/usr/bin/python3.11",
        str(
            paths["code_root"]
            / "subprojects/08_targeted_8b_cpt_experiments/scripts/run_canonical_train_segment.py"
        ),
        "--scale",
        scale,
        "--start-update",
        "{start_iteration}",
        "--end-update",
        "{end_iteration}",
        "--load-checkpoint",
        "{load_checkpoint}",
        "--attempt-root",
        "{attempt_root}",
        "--completion-receipt",
        "{completion_receipt}",
        "--run-root",
        "{run_root}",
        "--campaign-id",
        "{campaign_id}",
        "--segment-id",
        "{segment_id}",
        "--attempt",
        "{attempt_id}",
        "--contract-digest",
        "{contract_digest}",
        "--code-root",
        str(paths["code_root"]),
        "--code-receipt",
        str(paths["code_receipt"]),
        "--stage-root",
        str(paths["stage_root"]),
        "--megatron-root",
        str(paths["megatron_root"]),
        "--megatron-receipt",
        str(paths["megatron_receipt"]),
        "--validation-root",
        str(paths["validation_root"]),
        "--validation-receipt",
        str(paths["validation_receipt"]),
        "--extra-valid-sets",
        " ".join(ONLINE_PANEL_NAMES),
        "--new-greek-valid-sets",
        " ".join(sorted(ONLINE_NEW_GREEK)),
        "--training-run-permit",
        str(paths["run_permit"]),
        "--qualification-contract",
        str(paths["qualification_contract"]),
        "--peak-lr",
        str(run_permit["learning_rate"]["peak"]),
        "--floor-lr",
        str(run_permit["learning_rate"]["floor"]),
        "--microbatch",
        str(run_permit["profile"]["microbatch"]),
        "--tensor-parallel",
        str(run_permit["profile"]["tensor_parallel"]),
    ]
    immutable = [
        immutable_input(item_id, path) for item_id, path in immutable_paths.items()
    ]
    return {
        "schema_version": "apertus_campaign_v3",
        "campaign_id": f"hard-h2g-{scale}-matched-r2",
        "runtime_profile_id": runtime["profile_id"],
        "evaluation_plan_id": evaluation["plan_id"],
        "science": {
            "model_revision": f"{model['repo_id']}@{model['revision']}",
            "tokenizer_revision": f"{tokenizer['repo_id']}@{tokenizer['revision']}",
            "initialization_revision": initialization["sha256"],
            "data_manifest_revision": data_manifest["identity_sha256"],
            "training_data_manifest": portable_binding(
                paths["canonical_data_root"], paths["training_data_manifest"]
            ),
            "training_token_horizon": BOUNDARIES[-1] * GLOBAL_BATCH_TOKEN_SLOTS,
            "megatron_revision": str(megatron["upstream_commit"]),
            "train_argv": common_train_argv,
            "immutable_inputs": immutable,
        },
        "segments": build_segments(
            scale=scale,
            initialization_root=paths["initial_checkpoint_root"],
            initialization_permit=paths["initialization_permit"],
            phase_assets=phase_assets,
            stage_root=paths["stage_root"],
            timing=timing,
            schedule=schedule,
            pre_main_launch_gate=paths["pre_main_launch_gate"],
        ),
        "gates": build_qualification_gates(
            code_receipt=paths["code_receipt"],
            phase1_cache_receipt=phase_assets[1]["receipt"],
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--canonical-runner-root", type=Path, required=True)
    parser.add_argument("--canonical-data-root", type=Path, required=True)
    parser.add_argument("--initialization-permit", type=Path, required=True)
    parser.add_argument("--megatron-receipt", type=Path, required=True)
    parser.add_argument("--qualification-contract", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--training-run-permit", type=Path, required=True)
    parser.add_argument("--profile-promotion", type=Path, required=True)
    parser.add_argument("--production-timing", type=Path, required=True)
    parser.add_argument("--allocation-schedule", type=Path, required=True)
    parser.add_argument("--pre-main-launch-gate", type=Path, required=True)
    parser.add_argument("--phase1-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase2-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase3-data-path-spec", type=Path, required=True)
    parser.add_argument("--phase3-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase3-cache-root", type=Path, required=True)
    parser.add_argument("--eval-venv", type=Path, required=True)
    parser.add_argument("--native-eval-code-root", type=Path, required=True)
    parser.add_argument("--native-eval-code-receipt", type=Path, required=True)
    parser.add_argument("--native-source-contract", type=Path, required=True)
    parser.add_argument("--native-source-manifest", type=Path, required=True)
    parser.add_argument("--native-source-gate", type=Path, required=True)
    parser.add_argument("--native-exclusions", type=Path, required=True)
    parser.add_argument("--clean-examples", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_root = args.stage_root.resolve()
    code_root = args.code_root.resolve()
    canonical_data_root = args.canonical_data_root.resolve()
    require(
        stage_root.is_dir() and code_root.is_dir(),
        "stage or scientific code root missing",
    )
    require(canonical_data_root.is_dir(), "canonical data root missing")
    subproject = code_root / "subprojects/08_targeted_8b_cpt_experiments"
    receipts = stage_root / "receipts"
    experiment_path = subproject / "configs/hard_h_to_g_replication_v1.json"
    allocation_path = subproject / "configs/hard_h_to_g_allocation_v1.json"
    run_permit_path = args.training_run_permit.resolve()
    profile_promotion_path = args.profile_promotion.resolve()
    timing_path = args.production_timing.resolve()
    schedule_path = args.allocation_schedule.resolve()
    pre_main_launch_gate = args.pre_main_launch_gate.resolve()
    try:
        pre_main_launch_gate.relative_to(stage_root / "receipts")
    except ValueError as exc:
        raise ValueError(
            "pre-main launch-gate path escapes the stage receipts root"
        ) from exc
    megatron_receipt_path = args.megatron_receipt.resolve()
    try:
        megatron_receipt_path.relative_to(receipts)
    except ValueError as exc:
        raise ValueError("training Megatron receipt escapes the stage receipts root") from exc
    validation_receipt_path = receipts / "historical_online_validation_binaries.json"
    tokenizer_receipt_path = receipts / "tokenizer_148480.json"
    statistical_path = receipts / "statistical_decision_contract.json"
    legacy_path = receipts / "legacy_public_evaluator_contract.json"
    sentinel_manifest = (
        stage_root / "evaluation/greekmmlu_sentinels/sentinel_manifest.json"
    )
    clean_examples = (
        args.clean_examples.resolve()
        if args.clean_examples
        else stage_root / "evaluation/greekmmlu_sentinels/clean_examples.json"
    )
    required_files = [
        args.code_receipt,
        experiment_path,
        allocation_path,
        args.initialization_permit,
        args.qualification_contract,
        args.producer_compatibility,
        run_permit_path,
        profile_promotion_path,
        timing_path,
        schedule_path,
        megatron_receipt_path,
        validation_receipt_path,
        tokenizer_receipt_path,
        statistical_path,
        sentinel_manifest,
        clean_examples,
        args.native_eval_code_receipt,
        args.native_source_contract,
        args.native_source_manifest,
        args.native_source_gate,
        args.native_exclusions,
    ]
    if args.scale == "8b":
        required_files.append(legacy_path)
    require(
        all(path.resolve().is_file() for path in required_files),
        "one or more canonical campaign inputs are missing",
    )
    require(args.eval_venv.resolve().is_dir(), "evaluation venv missing")
    require(
        (args.eval_venv.resolve() / "bin/python").is_file(), "evaluation Python missing"
    )
    require(
        args.native_eval_code_root.resolve().is_dir(),
        "native evaluation code root missing",
    )

    experiment = read_json(experiment_path)
    run_permit = read_json(run_permit_path)
    require(
        run_permit.get("schema_version") == "apertus_hard_h_to_g_training_run_permit_v1"
        and run_permit.get("status") == "passed"
        and run_permit.get("scale") == args.scale,
        "training run permit drift",
    )
    timing = read_json(timing_path)
    schedule = read_json(schedule_path)
    require(
        timing.get("schema_version") == "apertus_hard_h_to_g_production_timing_v1"
        and timing.get("status") == "passed"
        and timing.get("scale") == args.scale,
        "production timing drift",
    )
    require(
        schedule.get("schema_version") == "apertus_hard_h_to_g_allocation_schedule_v1"
        and schedule.get("status") == "passed"
        and schedule.get("scale") == args.scale,
        "allocation schedule drift",
    )
    init_receipt = read_json(args.initialization_permit)
    require(
        init_receipt.get("schema_version") == "apertus_targeted_init_roundtrip_v1"
        and init_receipt.get("status") == "passed"
        and init_receipt.get("scale") == args.scale,
        "initialization permit drift",
    )
    initial_checkpoint_root = Path(str(init_receipt.get("megatron_root", ""))).resolve()
    require(initial_checkpoint_root.is_dir(), "initial checkpoint root missing")
    megatron = read_json(megatron_receipt_path)
    megatron_root = Path(str(megatron.get("output_root", ""))).resolve()
    require(megatron_root.is_dir(), "training Megatron root missing")
    validation = read_json(validation_receipt_path)
    validation_root = Path(str(validation.get("root", ""))).resolve()
    require(validation_root.is_dir(), "online validation root missing")
    tokenizer_receipt = read_json(tokenizer_receipt_path)
    tokenizer_dir = Path(str(tokenizer_receipt.get("output_root", ""))).resolve()
    require(tokenizer_dir.is_dir(), "tokenizer root missing")
    validation_manifest = (
        args.validation_manifest.resolve()
        if args.validation_manifest
        else Path(
            str(
                read_json(receipts / "reused_validation_panels.json")[
                    "parent_manifest"
                ]["path"]
            )
        ).resolve()
    )
    require(validation_manifest.is_file(), "validation manifest missing")
    training_data_manifest = canonical_data_root / "training_data_manifest.json"
    require(
        training_data_manifest.is_file(), "canonical training-data manifest missing"
    )
    data_manifest = read_json(training_data_manifest)
    require(
        data_manifest.get("schema_version") == "apertus_training_data_manifest_v1"
        and data_manifest.get("status") == "completed",
        "canonical training-data manifest drift",
    )
    phase_assets = {
        1: read_phase_asset(args.phase1_cache_receipt.resolve(), 1),
        2: read_phase_asset(args.phase2_cache_receipt.resolve(), 2),
        3: late_phase3_asset(
            spec=args.phase3_data_path_spec,
            receipt=args.phase3_cache_receipt,
            root=args.phase3_cache_root,
            stage_root=stage_root,
        ),
    }
    paths = {
        "stage_root": stage_root,
        "code_root": code_root,
        "code_receipt": args.code_receipt.resolve(),
        "canonical_data_root": canonical_data_root,
        "training_data_manifest": training_data_manifest,
        "experiment": experiment_path,
        "allocation": allocation_path,
        "initialization_permit": args.initialization_permit.resolve(),
        "initial_checkpoint_root": initial_checkpoint_root,
        "qualification_contract": args.qualification_contract.resolve(),
        "producer_compatibility": args.producer_compatibility.resolve(),
        "run_permit": run_permit_path,
        "profile_promotion": profile_promotion_path,
        "timing": timing_path,
        "schedule": schedule_path,
        "pre_main_launch_gate": pre_main_launch_gate,
        "megatron_receipt": megatron_receipt_path,
        "megatron_root": megatron_root,
        "validation_receipt": validation_receipt_path,
        "validation_root": validation_root,
        "tokenizer_receipt": tokenizer_receipt_path,
        "tokenizer_dir": tokenizer_dir,
        "validation_manifest": validation_manifest,
        "clean_examples": clean_examples.resolve(),
        "sentinel_manifest": sentinel_manifest.resolve(),
        "statistical_contract": statistical_path,
        "eval_venv": args.eval_venv.resolve(),
        "native_eval_code_root": args.native_eval_code_root.resolve(),
        "native_eval_code_receipt": args.native_eval_code_receipt.resolve(),
        "native_source_contract": args.native_source_contract.resolve(),
        "native_source_manifest": args.native_source_manifest.resolve(),
        "native_source_gate": args.native_source_gate.resolve(),
        "native_exclusions": args.native_exclusions.resolve(),
        "legacy_contract": legacy_path,
    }
    runtime = build_runtime(args.scale, run_permit)
    evaluation = build_evaluation_plan(args.scale, paths)
    campaign = build_campaign(
        scale=args.scale,
        paths=paths,
        experiment=experiment,
        run_permit=run_permit,
        data_manifest=data_manifest,
        runtime=runtime,
        evaluation=evaluation,
        phase_assets=phase_assets,
        timing=timing,
        schedule=schedule,
    )

    canonical_src = args.canonical_runner_root.resolve() / "src"
    require(canonical_src.is_dir(), "canonical runner source root missing")
    sys.path.insert(0, str(canonical_src))
    from apertus_cscs_campaign.contracts import (  # pylint: disable=import-outside-toplevel
        compile_contracts,
    )

    final_paths = {
        "campaign": canonical_data_root / f"campaign_{args.scale}_candidate.json",
        "runtime": canonical_data_root / f"runtime_{args.scale}_candidate.json",
        "evaluation": canonical_data_root / f"evaluation_{args.scale}.json",
        "compiled": canonical_data_root / f"compiled_{args.scale}_candidate.json",
    }
    require(
        not any(path.exists() for path in final_paths.values()),
        "immutable canonical campaign output exists",
    )
    temporary_paths = {
        key: path.with_name(f".{path.name}.prefreeze")
        for key, path in final_paths.items()
        if key != "compiled"
    }
    require(
        not any(path.exists() for path in temporary_paths.values()),
        "canonical prefreeze output exists",
    )
    try:
        write_json_atomic(temporary_paths["campaign"], campaign)
        write_json_atomic(temporary_paths["runtime"], runtime)
        write_json_atomic(temporary_paths["evaluation"], evaluation)
        compile_contracts(
            temporary_paths["campaign"],
            temporary_paths["runtime"],
            temporary_paths["evaluation"],
        )
        for key in ("campaign", "runtime", "evaluation"):
            temporary_paths[key].replace(final_paths[key])
        compiled = compile_contracts(
            final_paths["campaign"], final_paths["runtime"], final_paths["evaluation"]
        )
        write_json_atomic(final_paths["compiled"], compiled)
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
    for key in ("campaign", "runtime", "evaluation", "compiled"):
        print(final_paths[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

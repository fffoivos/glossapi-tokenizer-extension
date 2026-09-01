#!/usr/bin/env python3
"""Build the exact checkpoint/validation/GreekMMLU cadence for D0-D4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


FILLER_ID = np.uint64(2**64 - 1)
POOL_SHIFT = 62
GLOBAL_BATCH_SEQUENCES = 512
REGULAR_CADENCE = 512
WARMUP_STEPS = 800


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_receipt(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def hard_transition_update(ids: np.memmap, arm_id: str) -> tuple[int, int]:
    real = ids[ids != FILLER_ID]
    pools = real >> np.uint64(POOL_SHIFT)
    if arm_id == "D1_hard_h_to_g":
        positions = np.flatnonzero(pools == 1)
    elif arm_id == "D2_hard_g_to_h":
        positions = np.flatnonzero(pools == 0)
    else:
        raise ValueError(arm_id)
    if positions.size == 0:
        raise ValueError(f"hard arm has no destination-pool sequence: {arm_id}")
    first_destination_real_position = int(positions[0])
    sequence_id = int(real[first_destination_real_position])
    full_positions = np.flatnonzero(ids == sequence_id)
    if full_positions.size != 1:
        raise ValueError("stable sequence ID is not unique in hard schedule")
    transition_slot = int(full_positions[0])
    update_containing_transition = transition_slot // GLOBAL_BATCH_SEQUENCES + 1
    return transition_slot, update_containing_transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--experiment-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    schedule = read_json(args.schedule_manifest)
    matrix = read_json(args.experiment_matrix)
    if (
        schedule.get("schema_version")
        != "apertus_mini_five_data_order_schedules_v1"
        or schedule.get("status") != "completed"
    ):
        raise ValueError("five-arm schedule is not completed")
    evaluation = matrix["evaluation"]
    greekmmlu = evaluation["greekmmlu"]
    parallelism = matrix["training_control"]["parallelism"]
    checkpoint_policy = matrix["training_control"]["checkpoint_policy"]
    if (
        matrix["training_control"]["checkpoint_cadence_steps"] != REGULAR_CADENCE
        or matrix["training_control"]["learning_rate"]["warmup_steps"] != WARMUP_STEPS
        or greekmmlu["dataset_repo_id"] != "dascim/GreekMMLU"
        or greekmmlu["dataset_revision"]
        != "6a03aa06b68beb932fb75edff3a34e50b3674649"
        or greekmmlu["dataset_config"] != "All"
        or greekmmlu["dataset_split"] != "test"
        or greekmmlu["benchmark_origin"] != "natively_authored_greek"
        or greekmmlu["checkpoint_evaluation_is_mandatory"] is not True
        or greekmmlu["evaluation_scope"]
        != "every_required_training_point_for_every_arm_not_endpoint_only"
        or checkpoint_policy["greekmmlu_requires_exact_checkpoint_state"] is not True
    ):
        raise ValueError("experiment matrix checkpoint/GreekMMLU policy drift")

    arms = {row["arm_id"]: row for row in schedule["arms"]}
    if len(arms) != 5:
        raise ValueError("expected five schedules")
    total_steps = {arm: int(row["training_slots"]) // GLOBAL_BATCH_SEQUENCES for arm, row in arms.items()}
    if len(set(total_steps.values())) != 1:
        raise ValueError("schedule arms have different optimizer horizons")
    steps = next(iter(total_steps.values()))
    if any(int(row["training_slots"]) % GLOBAL_BATCH_SEQUENCES for row in arms.values()):
        raise ValueError("schedule is not aligned to complete global batches")
    if steps <= WARMUP_STEPS:
        raise ValueError("training horizon is shorter than warmup")

    transitions = {}
    for arm_id in ("D1_hard_h_to_g", "D2_hard_g_to_h"):
        row = arms[arm_id]
        ids_receipt = row["sequence_ids"]
        path = Path(ids_receipt["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(ids_receipt["bytes"])
            or sha256_file(path) != ids_receipt["sha256"]
        ):
            raise ValueError(f"hard-arm sequence schedule drift: {path}")
        ids = np.memmap(path, mode="r", dtype=np.uint64)
        slot, update = hard_transition_update(ids, arm_id)
        transitions[arm_id] = {
            "first_destination_sequence_slot": slot,
            "optimizer_update_containing_first_destination_sequence": update,
            "checkpoint_immediately_before": update - 1,
            "checkpoint_after_first_complete_transition_update": update,
        }

    common_points: dict[int, set[str]] = {0: {"initial_checkpoint"}}

    def add(step: int, reason: str) -> None:
        if step < 0 or step > steps:
            raise ValueError(f"checkpoint step outside horizon: {step}/{steps}")
        common_points.setdefault(step, set()).add(reason)

    add(WARMUP_STEPS, "after_warmup")
    for step in range(REGULAR_CADENCE, steps + 1, REGULAR_CADENCE):
        add(step, "regular_512_step_cadence")
    for step in parallelism.get(
        "selected_normal_partition_segment_boundary_iterations", []
    ):
        if int(step) <= steps:
            add(int(step), "normal_partition_segment_boundary")
    cooldown_start = math.floor(0.8 * steps)
    add(cooldown_start, "cooldown_start")
    add(steps, "raw_final_endpoint")
    for arm_id, transition in transitions.items():
        add(transition["checkpoint_immediately_before"], f"matched_{arm_id}_pre_transition")
        add(
            transition["checkpoint_after_first_complete_transition_update"],
            f"matched_{arm_id}_post_transition",
        )

    checkpoint_rows = []
    required_metrics = list(greekmmlu["metrics"])
    for step in sorted(common_points):
        checkpoint_rows.append(
            {
                "iteration": step,
                "nominal_consumed_tokens": step
                * int(matrix["training_control"]["global_batch_tokens"]),
                "reasons": sorted(common_points[step]),
                "all_arms": sorted(arms),
                "full_state_checkpoint_required": True,
                "fast_source_conditioned_panel_required": True,
                "native_greekmmlu_required": True,
                "native_greekmmlu_metrics": required_metrics,
                "same_frozen_evaluator_contract": True,
            }
        )
    if steps == 38_496 and (
        len(checkpoint_rows) != 83 or len(checkpoint_rows) * len(arms) != 415
    ):
        raise ValueError(
            "production checkpoint cadence must contain exactly 83 points per arm "
            "and 415 native-GreekMMLU evaluations"
        )
    payload = {
        "schema_version": "apertus_mini_checkpoint_evaluation_plan_v1",
        "status": "frozen",
        "schedule_manifest": input_receipt(args.schedule_manifest),
        "experiment_matrix": input_receipt(args.experiment_matrix),
        "optimizer_steps": steps,
        "checkpoint_count_per_arm": len(checkpoint_rows),
        "native_greekmmlu_evaluations_total": len(checkpoint_rows) * len(arms),
        "greekmmlu_origin": "natively_authored_greek",
        "greekmmlu_dataset": {
            "repo_id": greekmmlu["dataset_repo_id"],
            "revision": greekmmlu["dataset_revision"],
            "config": greekmmlu["dataset_config"],
            "split": greekmmlu["dataset_split"],
        },
        "hard_transitions": transitions,
        "checkpoint_rows": checkpoint_rows,
        "pruning_policy": checkpoint_policy[
            "prune_nonboundary_payload_only_after_all_evaluation_receipts_are_frozen"
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "ok": True,
                "scope": "supplied_schedule_manifest",
                "optimizer_steps": steps,
                "checkpoints_per_arm": len(checkpoint_rows),
                "greekmmlu_evaluations": len(checkpoint_rows) * len(arms),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

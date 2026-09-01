#!/usr/bin/env python3
"""Bind a promoted parallel profile and selected LR to one training scale."""

from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from producer_bundle_compatibility import load_authority, require_accepted_producer

PROFILE_CHECKS = (
    "fixed_batch_loss_parity",
    "fixed_batch_gradient_parity",
    "restart_next_step_parity",
    "phase2_entry_and_restart_parity",
    "sample_and_mask_cursor_continuity",
    "zero_skipped_or_nonfinite_updates",
    "tokens_per_gpu_hour_measured",
)
LR_CHECKS = (
    "candidate_grid_predeclared",
    "same_initialization_and_data_prefix",
    "independent_optimizer_state_per_candidate",
    "greekmmlu_and_downstream_benchmarks_not_used",
    "stability_limits_passed",
    "source_conditioned_selector_applied",
)


def validate_checks(value: object, names: tuple[str, ...], label: str) -> None:
    require(
        isinstance(value, dict)
        and set(value) == set(names)
        and all(value[name] is True for name in names),
        f"{label} checks are incomplete",
    )


def validate_evidence(value: object, label: str) -> None:
    require(isinstance(value, list) and value, f"{label} backing evidence is empty")
    for row in value:
        require(isinstance(row, dict), f"{label} evidence row is malformed")
        path = Path(str(row.get("path", "")))
        require(path.is_file(), f"{label} evidence file missing: {path}")
        require(row == file_binding(path), f"{label} evidence binding drift: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--profile-promotion", type=Path, required=True)
    parser.add_argument("--lr-selection", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_permit(
    value: dict[str, object],
    *,
    scale: str,
    nodes: int,
    tensor_parallel: int,
    microbatch: int,
    peak_lr: str,
    floor_lr: str,
) -> None:
    require(value.get("schema_version") == "apertus_hard_h_to_g_training_run_permit_v1", "training run-permit schema drift")
    require(value.get("status") == "passed" and value.get("scale") == scale, "training run-permit status/scale drift")
    profile = value.get("profile")
    require(isinstance(profile, dict), "training run-permit profile missing")
    require(
        (profile.get("nodes"), profile.get("tensor_parallel"), profile.get("microbatch"))
        == (nodes, tensor_parallel, microbatch),
        "training run-permit profile drift",
    )
    require(profile.get("global_batch_sequences") == 1024, "training run-permit global batch drift")
    lr = value.get("learning_rate")
    require(isinstance(lr, dict), "training run-permit LR missing")
    require(str(lr.get("peak")) == peak_lr and str(lr.get("floor")) == floor_lr, "training run-permit LR drift")
    require(Decimal(floor_lr) == Decimal(peak_lr) * Decimal("0.1"), "training run-permit floor ratio drift")
    code = value.get("executing_code_bundle")
    current = executing_code_bundle()
    require(
        isinstance(code, dict)
        and code.get("root") == current["root"]
        and code.get("tree_sha256") == current["tree_sha256"],
        "training run-permit code-bundle drift",
    )


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable training run permit exists: {args.output}")
    experiment = read_json(args.experiment)
    allocation = read_json(args.allocation)
    require(experiment.get("schema_version") == "apertus_hard_h_to_g_replication_v2", "experiment contract drift")
    require(allocation.get("schema_version") == "apertus_hard_h_to_g_allocation_v1", "allocation contract drift")
    current = executing_code_bundle()
    _, accepted_producers = load_authority(args.producer_compatibility, current)

    promotion = read_json(args.profile_promotion)
    require(promotion.get("schema_version") == "apertus_hard_h_to_g_profile_promotion_v1", "profile-promotion schema drift")
    require(promotion.get("status") == "promoted" and promotion.get("scale") == args.scale, "profile promotion status/scale drift")
    validate_checks(promotion.get("checks"), PROFILE_CHECKS, "profile-promotion")
    validate_evidence(promotion.get("evidence"), "profile-promotion")
    require_accepted_producer(promotion, accepted_producers, "profile-promotion")
    profile = promotion.get("selection")
    require(isinstance(profile, dict), "promoted profile selection missing")
    nodes = int(profile.get("nodes", 0))
    tp = int(profile.get("tensor_parallel", 0))
    pp = int(profile.get("pipeline_parallel", 0))
    cp = int(profile.get("context_parallel", 0))
    dp = int(profile.get("data_parallel", 0))
    microbatch = int(profile.get("microbatch", 0))
    world = nodes * 4
    require(nodes > 0 and tp > 0 and pp == 1 and cp == 1 and microbatch > 0, "promoted profile geometry invalid")
    require(world == tp * pp * cp * dp, "promoted profile world-size arithmetic drift")
    require(1024 % (dp * microbatch) == 0, "promoted profile global batch is not divisible")
    if args.scale == "8b":
        require((nodes, tp, dp, microbatch) == (16, 2, 32, 2), "8B promoted profile drift")
    else:
        candidate_rows = allocation["profiles"]["1p5b_candidates"]
        candidate_ids = {row["profile_id"] for row in candidate_rows}
        require(profile.get("profile_id") in candidate_ids and nodes in {1, 2, 4} and tp == 1, "1.5B profile outside frozen candidate grid")

    lr_selection = read_json(args.lr_selection)
    require(lr_selection.get("schema_version") == "apertus_hard_h_to_g_lr_selection_v1", "LR-selection schema drift")
    require(lr_selection.get("status") == "selected" and lr_selection.get("scale") == args.scale, "LR selection status/scale drift")
    validate_checks(lr_selection.get("checks"), LR_CHECKS, "LR-selection")
    validate_evidence(lr_selection.get("evidence"), "LR-selection")
    require_accepted_producer(lr_selection, accepted_producers, "LR-selection")
    peak_lr = str(lr_selection.get("peak_lr"))
    floor_lr = str(lr_selection.get("floor_lr"))
    require(Decimal(floor_lr) == Decimal(peak_lr) * Decimal("0.1"), "selected LR floor ratio drift")
    if args.scale == "8b":
        require(peak_lr == "5.5e-5" and floor_lr == "5.5e-6", "8B LR selection drift")
    else:
        require(lr_selection.get("candidates") == ["7.5e-5", "1.0e-4", "1.25e-4"], "1.5B LR candidate grid drift")
        require(peak_lr in lr_selection["candidates"], "1.5B selected peak is outside candidate grid")

    payload: dict[str, object] = {
        "schema_version": "apertus_hard_h_to_g_training_run_permit_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scale": args.scale,
        "experiment": file_binding(args.experiment),
        "allocation": file_binding(args.allocation),
        "profile_promotion": file_binding(args.profile_promotion),
        "lr_selection": file_binding(args.lr_selection),
        "producer_bundle_compatibility": file_binding(args.producer_compatibility),
        "profile": profile,
        "learning_rate": {"peak": peak_lr, "floor": floor_lr, "terminal_ratio": "0.1"},
        "executing_code_bundle": current,
    }
    validate_permit(
        payload,
        scale=args.scale,
        nodes=nodes,
        tensor_parallel=tp,
        microbatch=microbatch,
        peak_lr=peak_lr,
        floor_lr=floor_lr,
    )
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

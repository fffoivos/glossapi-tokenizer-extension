#!/usr/bin/env python3
"""Freeze planning or launch-ready receipts for targeted 8B experiments.

Planning mode records all unresolved gates. Launch mode refuses to write a
receipt until the release-internal polytonic selection, decontamination,
selected-pool and continuation schedule receipts are supplied and internally
consistent.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from contract_utils import (
    GLOBAL_BATCH_TOKENS,
    TOKENIZER_SHA256,
    file_binding,
    geometry,
    nearest_replay_targets,
    read_json,
    require,
    token_milestones,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-a-config", type=Path, required=True)
    parser.add_argument("--experiment-b-config", type=Path, required=True)
    parser.add_argument("--allocation-config", type=Path, required=True)
    parser.add_argument("--poly-token-receipt", type=Path)
    parser.add_argument("--a-decontamination-receipt", type=Path)
    parser.add_argument("--a-selected-pools-receipt", type=Path)
    parser.add_argument("--b-schedule-manifest", type=Path)
    parser.add_argument("--mode", choices=("planning", "launch"), default="planning")
    parser.add_argument("--experiment", choices=("A", "B", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_static(a: dict[str, Any], b: dict[str, Any], allocation: dict[str, Any]) -> None:
    require(a.get("schema_version") == "apertus_targeted_8b_experiment_a_plan_v1", "A schema drift")
    require(b.get("schema_version") == "apertus_targeted_8b_experiment_b_plan_v1", "B schema drift")
    require(allocation.get("schema_version") == "apertus_targeted_8b_allocation_plan_v1", "allocation schema drift")
    release = a["source_release"]
    require(release["revision"] == "987b8955fcd395c6219e39df9e64715457f69065", "HF revision drift")
    require(release["additional_global_deduplication_allowed"] is False, "second dedup must be forbidden")
    require(a["decontamination"]["apply_after_anonymization"] is True, "decontamination order drift")
    require(a["decontamination"]["deduplication_stage_after_anonymization"] == "forbidden", "post-anonymization dedup drift")
    poly = a["modern_data"]["polytonic"]
    require(poly.get("selection_authority") == "pinned_hf_release_only", "poly source-authority drift")
    require(
        poly.get("external_dataset_allowed") is False
        and poly.get("row_level_reconstruction_allowed") is False
        and poly.get("additional_deduplication_allowed") is False,
        "poly release-only policy drift",
    )
    require(
        poly.get("source_datasets") == [
            "1000_prwta_xronia_ellhnikhs",
            "Ekklisiastika_Keimena",
            "Wikisource_Greek_texts",
            "klasikh_arx_ell_grammateia",
        ],
        "poly release source-set drift",
    )
    require(poly.get("historical_split_manifest_role") == "provenance_only_not_an_input_requirement", "historical poly manifest role drift")
    require(a["mix"] == {
        "modern": "0.79", "foreign_replay": "0.20", "old_greek_replay": "0.01",
        "stationary_window_balanced": True, "seed": 20260811,
    }, "A mixture drift")
    require(
        a["evaluation"]["source_conditioned_interval_updates"] == 25
        and a["evaluation"]["greekmmlu_cadence_active_tokens"] == 2_000_000_000,
        "A evaluation cadence drift",
    )
    require(
        b["evaluation"]["source_conditioned_interval_updates"] == 25
        and b["evaluation"]["greekmmlu_cadence_active_tokens"] == 1_000_000_000,
        "B evaluation cadence drift",
    )
    profile = allocation["production_profile"]
    expected = {
        "profile_id": "dp32_16node", "partition": "normal", "nodes": 16,
        "gpus_per_node": 4, "world_size": 64, "tensor_parallel": 2,
        "pipeline_parallel": 1, "context_parallel": 1, "data_parallel": 32,
        "ntasks_per_node": 4, "gpus_per_task": 1, "cpus_per_task": 72,
        "exclusive": True, "memory_gb_per_node": 450, "leaf_switches": 1,
        "wall_seconds": 43200, "signal": "B:USR1@600", "dp64_allowed": False,
    }
    require(profile == expected, "production DP32 profile drift")
    prep = allocation["preparation"]
    require(prep["partition"] == "debug" and prep["nodes"] == 1, "preparation must use one debug node")
    require(prep["normal_partition_for_these_roles"] == "forbidden", "normal preparation drift")
    distributed_smoke = allocation["distributed_prelaunch_restart_smoke"]
    require(
        distributed_smoke
        == {
            "partition": "normal",
            "profile_id": "dp32_16node",
            "nodes": 16,
            "leaf_switches": 1,
            "wall_seconds": 3600,
            "allocations_per_experiment": 1,
            "uninterrupted_updates": 2,
            "resume_source": "exact_uninterrupted_control_checkpoint",
            "resumed_updates": 1,
            "submit_only_after_debug_data_and_training_assets_are_frozen": True,
            "production_horizon_submission_still_forbidden_until_receipt_passes": True,
        },
        "distributed restart-smoke allocation drift",
    )
    handoff = allocation["experiment_a"]
    maximum_hold = profile["wall_seconds"] - handoff["conservative_segment_runtime_seconds"] - handoff["allocation_reserve_seconds"]
    require(maximum_hold == handoff["maximum_hold_seconds"] == 6000, "A maximum-hold arithmetic drift")
    source_trigger = handoff["conservative_segment_runtime_seconds"] - maximum_hold
    require(source_trigger == handoff["source_trigger_seconds"] == 30000, "A trigger arithmetic drift")
    require(handoff["source_trigger_minutes"] == 500, "A trigger minutes drift")


def freeze_a(
    config: dict[str, Any],
    poly_path: Path | None,
    decontam_path: Path | None,
    pools_path: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    bindings: dict[str, Any] = {}
    # Before the release-internal source audit exists, use only the pinned
    # release manifest estimate.  The authoritative value is replaced by the
    # bound post-extraction token receipt below; never fall back to a
    # historical/external polytonic artifact.
    poly_cfg = config["modern_data"]["polytonic"]
    poly_tokens = int(
        poly_cfg.get(
            "planning_training_tokens",
            poly_cfg["release_manifest_training_tokens_before_required_exclusions"],
        )
    )
    academic_tokens = config["modern_data"]["academic_training_tokens_pre_decontamination"]
    if poly_path is None:
        blockers.append("release_internal_polytonic_token_receipt_missing")
    else:
        poly = read_json(poly_path)
        require(poly.get("schema_version") == "targeted_8b_release_polytonic_token_receipt_v1" and poly.get("status") == "passed", "poly receipt is not passed")
        require(poly.get("selection_authority") == "pinned_hf_release_only", "poly receipt source authority drift")
        require(poly.get("source_datasets") == poly_cfg["source_datasets"], "poly receipt source-set drift")
        require(poly.get("tokenizer_json_sha256") == TOKENIZER_SHA256, "poly tokenizer drift")
        poly_tokens = int(poly["training_tokens"])
        bindings["poly_token_receipt"] = file_binding(poly_path)
    if decontam_path is None:
        blockers.append("selected_modern_greekmmlu_decontamination_receipt_missing")
    else:
        decontam = read_json(decontam_path)
        require(decontam.get("schema_version") == "targeted_8b_a_decontamination_summary_v1", "A decontamination schema drift")
        require(decontam.get("status") == "passed", "A decontamination did not pass")
        counts = decontam.get("counts", {})
        require(int(counts.get("input", -1)) == int(counts.get("kept", -2)) + int(counts.get("dropped", -3)), "A decontamination row accounting drift")
        require(int(counts.get("remaining_high_confidence_matches", -1)) == 0, "A kept pools still match GreekMMLU")
        require(decontam.get("no_global_deduplication") is True, "A decontamination summary does not prohibit deduplication")
        bindings["decontamination_receipt"] = file_binding(decontam_path)
    if pools_path is None:
        blockers.append("exact_selected_pool_receipt_missing")
        hplt_tokens = academic_tokens
    else:
        pools = read_json(pools_path)
        require(pools.get("schema_version") == "targeted_8b_a_selected_pools_v1" and pools.get("status") == "passed", "A selected-pool receipt is not passed")
        require(pools.get("no_global_deduplication") is True, "A selected pools do not prove no second dedup")
        require(pools.get("greekmmlu_postscan_high_confidence_matches") == 0, "A kept pools still match GreekMMLU")
        academic_tokens = int(pools["pool_active_tokens"]["academic"])
        hplt_tokens = int(pools["pool_active_tokens"]["hplt"])
        poly_tokens = int(pools["pool_active_tokens"]["polytonic"])
        require(hplt_tokens == academic_tokens, "A HPLT/academic token equality drift")
        bindings["selected_pools_receipt"] = file_binding(pools_path)
    modern = academic_tokens + hplt_tokens + poly_tokens
    foreign, old = nearest_replay_targets(modern)
    geom = geometry(modern, foreign, old)
    cooldown_start = int(0.8 * geom["updates"])
    boundaries = [0, geom["updates"] // 2, geom["updates"]]
    return {
        "status": "ready" if not blockers else "blocked",
        "bindings": bindings,
        "pool_active_tokens": {"academic": academic_tokens, "hplt": hplt_tokens, "polytonic": poly_tokens},
        "geometry": geom,
        "optimization": {
            "warmup_updates": 400,
            "alpha_warmup_updates": geom["updates"],
            "beta3_warmup_updates": geom["updates"],
            "cooldown_start_update": cooldown_start,
            "cooldown_updates": geom["updates"] - cooldown_start,
        },
        "segments": boundaries,
        "greekmmlu_checkpoint_updates": token_milestones(
            geom["updates"], cadence_tokens=2_000_000_000, warmup_updates=400,
            cooldown_start=cooldown_start, boundaries=boundaries,
        ),
    }, blockers


def freeze_b(config: dict[str, Any], schedule_path: Path | None) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if schedule_path is None:
        blockers.append("continuation_b_schedule_receipt_missing")
        geom = geometry(
            config["planning_geometry"]["modern_active_tokens"],
            config["planning_geometry"]["foreign_active_tokens"],
            config["planning_geometry"]["old_greek_active_tokens"],
        )
        binding = None
    else:
        schedule = read_json(schedule_path)
        require(schedule.get("schema_version") == "apertus_data_order_schedules_v1" and schedule.get("status") == "completed", "B schedule is not compatible/completed")
        continuations = schedule.get("continuation_contract", {})
        require(continuations.get("parent_checkpoint_iteration") == 9536, "B parent checkpoint drift")
        checkpoint_binding = continuations.get("parent_checkpoint_receipt", {})
        expected_checkpoint_receipt = Path(config["parent"]["checkpoint_receipt"])
        require(
            checkpoint_binding == file_binding(expected_checkpoint_receipt),
            "B parent checkpoint receipt binding drift",
        )
        expected_checkpoint_dir = (
            Path(config["parent"]["run_root"]) / "checkpoints/iter_0009536"
        ).resolve()
        require(
            Path(continuations.get("parent_checkpoint_directory", "")).resolve()
            == expected_checkpoint_dir,
            "B parent checkpoint directory drift",
        )
        require(continuations.get("parent_prefix_overlap_selected_sequences") == 0, "B reuses parent-prefix sequences")
        require(continuations.get("parent_prefix_is_byte_exact") is True, "B parent schedule prefix is not byte-exact")
        require(continuations.get("absolute_final_optimizer_update") == 12290, "B absolute final update drift")
        pools = continuations.get("realized_pool_active_tokens", {})
        require(int(pools.get("G", -1)) == 9123187023, "B remaining non-HPLT tokens drift")
        geom = geometry(int(pools["G"]), int(pools["F"]), int(pools["O"]))
        require(continuations.get("continuation_optimizer_updates") == geom["updates"], "B continuation update geometry drift")
        binding = file_binding(schedule_path)
    start = 9536
    checkpoints = [start + value for value in token_milestones(geom["updates"], cadence_tokens=1_000_000_000)]
    result = {
        "status": "ready" if not blockers else "blocked",
        "schedule_manifest": binding,
        "geometry": geom,
        "absolute_updates": {"start": start, "end": start + geom["updates"]},
        "optimization": {
            "warmup_updates": 0,
            "cooldown_updates": geom["updates"],
            "load_optimizer_rng_and_sample_cursor": True,
            "rescale_ademamix_ramps": False,
        },
        "greekmmlu_checkpoint_updates": checkpoints,
    }
    return result, blockers


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {args.output}")
    a = read_json(args.experiment_a_config)
    b = read_json(args.experiment_b_config)
    allocation = read_json(args.allocation_config)
    validate_static(a, b, allocation)
    frozen_a, blockers_a = freeze_a(a, args.poly_token_receipt, args.a_decontamination_receipt, args.a_selected_pools_receipt)
    frozen_b, blockers_b = freeze_b(b, args.b_schedule_manifest)
    blockers = {"experiment_a": blockers_a, "experiment_b": blockers_b}
    selected_names = ("experiment_a", "experiment_b") if args.experiment == "both" else (f"experiment_{args.experiment.lower()}",)
    selected_blockers = {name: blockers[name] for name in selected_names}
    if args.mode == "launch" and any(selected_blockers.values()):
        raise ValueError(f"launch contract has unresolved blockers: {selected_blockers}")
    payload = {
        "schema_version": "apertus_targeted_8b_frozen_contracts_v1",
        "status": "launch_ready" if not any(selected_blockers.values()) else "blocked",
        "mode": args.mode,
        "selected_experiment": args.experiment,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "configs": {
            "experiment_a": file_binding(args.experiment_a_config),
            "experiment_b": file_binding(args.experiment_b_config),
            "allocation": file_binding(args.allocation_config),
        },
        "global_batch_token_slots": GLOBAL_BATCH_TOKENS,
        "experiment_a": frozen_a,
        "experiment_b": frozen_b,
        "blockers": blockers,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Derive exact v45-compatible training recipes and DP32 profiles for A or B."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any

from contract_utils import copy_file_atomic, file_binding, read_json, require, token_milestones, write_json_atomic


PANELS = [
    "hplt", "non_hplt", "openarchives", "greek_phd", "historical_polytonic",
    "english", "de", "ru", "zh", "code", "math", "old_greek",
    "neutral_external_modern_greek",
]
SOURCE_CONDITIONED_INTERVAL_UPDATES = 25


def exact_geometry(pool: dict[str, Any], schedule: dict[str, Any]) -> tuple[dict[str, int], int, int]:
    arms = {row["arm_id"]: row for row in schedule["arms"]}
    require("D0_mixed" in arms, "D0_mixed schedule arm missing")
    arm = arms["D0_mixed"]
    active = {key: int(value) for key, value in arm["pool_active_tokens"].items()}
    require(set(active) == {"H", "G", "F", "O"}, "schedule pool labels drift")
    active_total = sum(active.values())
    require(active_total == int(pool["integer_79_20_1_geometry"]["active_tokens"]), "pool/schedule active-token drift")
    updates = int(arm["optimizer_updates"])
    require(int(arm["training_slots"]) == updates * 1024, "schedule update/slot drift")
    return active, active_total, updates


def evaluation_checkpoints(greek: list[int], boundaries: list[int]) -> list[int]:
    return sorted(set(greek[1:]) | set(boundaries[1:]))


def build_recipe(experiment: str, base: dict[str, Any], pool: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    recipe = copy.deepcopy(base)
    active, active_total, updates = exact_geometry(pool, schedule)
    recipe["status"] = "frozen_pending_launch_gates"
    recipe["data"]["planning_post_dedup_active_tokens"] = active_total
    recipe["data"]["planning_training_slots_tokens"] = updates * 1024 * 4096
    recipe["data"]["planning_loss_inactive_tail_tokens"] = recipe["data"]["planning_training_slots_tokens"] - active_total
    recipe["data"]["source_dataset"] = {
        "repo_id": "fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2",
        "revision": "987b8955fcd395c6219e39df9e64715457f69065",
        "complete_eligible_training_pass": False,
        "libduth": {
            "selected": False,
            "legal_conclusion_claimed": False,
            "technical_selection": "not_selected_by_either_targeted_experiment",
            "evidence_conflict": "not_applicable_to_selected_rows",
            "required_resolution": "not_applicable_to_selected_rows",
        },
    }
    recipe["data"]["exact_pool_active_tokens"] = active
    recipe["data"]["additional_global_deduplication"] = False
    batch = recipe["batch_and_parallelism"]
    batch["training_updates"] = updates
    batch["training_samples"] = updates * batch["global_batch_sequences"]
    require(batch["global_batch_tokens"] == 4_194_304, "global batch drift")
    # train_segment.sbatch deliberately evaluates the 13 aggregate panels every
    # 25 optimizer updates. Do not inherit the parent's stale descriptive
    # 238-update cadence into a targeted receipt while executing 25.
    source_validation = recipe["evaluation"]["source_conditioned"]
    source_validation["interval_updates"] = SOURCE_CONDITIONED_INTERVAL_UPDATES
    source_validation["cadence_tokens_approx"] = (
        SOURCE_CONDITIONED_INTERVAL_UPDATES * batch["global_batch_tokens"]
    )

    if experiment == "A":
        recipe["recipe_id"] = "targeted8b-a-academic-hplt-poly-wsd10-v1"
        recipe["scientific_decision"] = {
            "experiment": "A",
            "data_order": "stationary_random_mixture",
            "checkpoint_averaging": False,
            "lr_floor": "WSD-10",
        }
        midpoint = updates // 2
        boundaries = [0, midpoint, updates]
        cooldown_start = int(0.8 * updates)
        recipe["optimization"]["beta3_warmup_updates"] = updates
        recipe["optimization"]["alpha_warmup_updates"] = updates
        lr = recipe["optimization"]["learning_rate"]
        lr.update({"warmup_updates": 400, "stable_until_update": cooldown_start, "cooldown_updates": updates - cooldown_start})
        greek = token_milestones(
            updates, cadence_tokens=2_000_000_000, warmup_updates=400,
            cooldown_start=cooldown_start, boundaries=boundaries,
        )
        recipe["evaluation"]["greekmmlu"].update({
            "cadence_policy": "initial_post_warmup_about_every_2B_tokens_cooldown_start_boundaries_final",
            "cadence_active_tokens": 2_000_000_000,
            "cadence_updates": round(2_000_000_000 / batch["global_batch_tokens"]),
        })
        if not greek or greek[0] != 0:
            greek = [0, *greek]
        recipe["data"]["selected_components"] = {
            "openarchives_gr_passes": 1,
            "greek_phd_passes": 1,
            "hplt_active_tokens_equal_academic": True,
            "release_polytonic_sources_passes": 1,
        }
    else:
        continuation = schedule["continuation_contract"]
        require(continuation.get("parent_prefix_is_byte_exact") is True, "B prefix is not exact")
        parent_update = int(continuation["parent_checkpoint_iteration"])
        require(updates == int(continuation["absolute_final_optimizer_update"]), "B final update drift")
        boundaries = [0, parent_update, updates]
        recipe["recipe_id"] = "targeted8b-b-iter9536-unseen-glossapi-wsd10-v1"
        recipe["scientific_decision"] = {
            "experiment": "B",
            "data_order": "parent_D0_prefix_then_unseen_non_hplt_suffix",
            "checkpoint_averaging": False,
            "lr_floor": "WSD-10",
        }
        lr = recipe["optimization"]["learning_rate"]
        lr.update({"warmup_updates": 400, "stable_until_update": parent_update, "cooldown_updates": updates - parent_update})
        # Preserve the exact sanitized parent's optimizer ramp horizon.
        parent_alpha_ramp = int(base["optimization"]["alpha_warmup_updates"])
        parent_beta3_ramp = int(base["optimization"]["beta3_warmup_updates"])
        require(recipe["optimization"]["alpha_warmup_updates"] == parent_alpha_ramp, "B alpha ramp drift")
        require(recipe["optimization"]["beta3_warmup_updates"] == parent_beta3_ramp, "B beta3 ramp drift")
        old_greek = [int(value) for value in recipe["evaluation"]["greekmmlu"]["checkpoint_updates"] if int(value) <= parent_update]
        local = token_milestones(updates - parent_update, cadence_tokens=1_000_000_000)
        greek = sorted(set(old_greek) | {parent_update + value for value in local} | {parent_update, updates})
        require(greek[0] == 0, "B GreekMMLU history lost update zero")
        recipe["evaluation"]["greekmmlu"].update({
            "cadence_policy": "parent_anchor_about_every_1B_continuation_tokens_and_final",
            "cadence_active_tokens": 1_000_000_000,
            "cadence_updates": round(1_000_000_000 / batch["global_batch_tokens"]),
        })
        recipe["data"]["selected_components"] = {
            "parent_prefix_updates": parent_update,
            "parent_prefix_byte_exact": True,
            "continuation_only_parent_suffix_sequences": True,
            "continuation_modern_pool": "non_hplt",
        }
        recipe["optimization"]["continuation"] = {
            "load_optimizer_rng_and_sample_cursor": True,
            "restart_warmup": False,
            "rescale_ademamix_ramps": False,
            "inherited_alpha_warmup_updates": parent_alpha_ramp,
            "inherited_beta3_warmup_updates": parent_beta3_ramp,
            "decay_starts_at_update": parent_update,
        }
        recipe["initialization"]["targeted_resume"] = {
            "mode": "exact_distributed_checkpoint_resume",
            "iteration": parent_update,
            "checkpoint_directory": continuation["parent_checkpoint_directory"],
            "checkpoint_receipt": continuation["parent_checkpoint_receipt"],
            "checkpoint_metadata": continuation["parent_checkpoint_metadata"],
            "load_model_optimizer_rng_and_consumed_sample_cursor": True,
            "restart_warmup": False,
        }

    segments = recipe["segments"]
    segments["boundaries"] = boundaries
    segments["count"] = len(boundaries) - 1
    recipe["evaluation"]["greekmmlu"]["checkpoint_updates"] = greek
    recipe["evaluation"]["checkpoint_updates"] = evaluation_checkpoints(greek, boundaries)
    recipe["evaluation"]["per_document_validation"]["milestone_updates"] = sorted(set((0, boundaries[-2], boundaries[-1])))
    recipe["evaluation"]["checkpoint_averaging"] = False
    recipe["evaluation"]["source_conditioned"]["panels"] = PANELS
    recipe["launch_gates"] = sorted(set(recipe["launch_gates"]) | {
        "targeted_frozen_data_and_schedule_receipts_pass",
        "targeted_decontamination_and_validation_exclusion_audits_pass",
        "dp32_restart_and_two_update_smoke_pass",
        "initial_source_validation_and_greekmmlu_anchor_pass",
        "fresh_scheduler_snapshot_is_recorded",
    })
    return recipe


def build_profiles(recipe: dict[str, Any]) -> dict[str, Any]:
    batch = recipe["batch_and_parallelism"]
    boundaries = recipe["segments"]["boundaries"]
    return {
        "schema_version": "apertus_full_8b_execution_profiles_v1",
        "status": "frozen_candidates",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_recipe_id": recipe["recipe_id"],
        "profiles": {
            "dp32_16node": {
                "status": "proven_fallback",
                "nodes": 16,
                "gpus_per_node": 4,
                "world_size": 64,
                "data_parallel": 32,
                "gradient_accumulation_steps": 16,
                "segment_boundaries": boundaries,
            }
        },
        "scientific_invariants": {
            "sequence_length": 4096,
            "global_batch_sequences": 1024,
            "global_batch_tokens": 4_194_304,
            "micro_batch_sequences": 2,
            "training_updates": batch["training_updates"],
            "tensor_parallel": 2,
            "pipeline_parallel": 1,
            "context_parallel": 1,
            "optimizer": "AdEMAMix",
            "precision": "bf16_parameters_fp32_main_gradients",
            "schedule_arm": "D0_mixed",
            "dp64_allowed": False,
        },
        "restart_parity": {
            "loss_and_parameter_norm_require_exact_logged_equality": True,
            "gradient_norm_atol": 0.001,
            "gradient_norm_rtol": 0.02,
            "threshold_source": "predeclared_inherited_dp32_restart_acceptance_before_targeted_smoke",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--token-byte-lengths", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), f"immutable training-assets receipt exists: {args.receipt}")
    recipe_path = args.stage_root / "contracts/recipe_8b_full_mixed.sanitized.json"
    profiles_path = args.stage_root / "contracts/execution_profiles.sanitized.json"
    validation_output = args.stage_root / "validation/validation_manifest.json"
    token_bytes_output = args.stage_root / "validation/token_utf8_byte_lengths.npy"
    for path in (recipe_path, profiles_path, validation_output, token_bytes_output):
        require(not path.exists(), f"immutable training asset exists: {path}")
    base = read_json(args.base_recipe)
    pool = read_json(args.pool_receipt)
    schedule = read_json(args.schedule_manifest)
    validation = read_json(args.validation_manifest)
    require(base.get("schema_version") == "apertus_full_8b_mixed_cpt_recipe_v1", "base recipe schema drift")
    require(pool.get("schema_version") == "apertus_schedule_pool_corpus_v1" and pool.get("status") == "completed", "pool receipt drift")
    require(schedule.get("schema_version") == "apertus_data_order_schedules_v1" and schedule.get("status") == "completed", "schedule receipt drift")
    require(
        validation.get("schema_version") == "apertus_full_8b_validation_manifest_v1"
        and validation.get("status") == "frozen"
        and validation.get("all_panels_training_exact_content_disjoint") is True
        and len(validation.get("panels", [])) == 13,
        "validation manifest drift",
    )
    require(args.token_byte_lengths.is_file(), "token byte-length table missing")
    recipe = build_recipe(args.experiment, base, pool, schedule)
    recipe["data"]["source_binary_root"] = (
        str(args.stage_root.resolve())
        if args.experiment == "A"
        else str(pool.get("source_root", Path(schedule["packed_corpus_receipt"]["path"]).resolve().parent))
    )
    profiles = build_profiles(recipe)
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        write_json_atomic(recipe_path, recipe)
        created.append(recipe_path)
        write_json_atomic(profiles_path, profiles)
        created.append(profiles_path)
        copy_file_atomic(args.validation_manifest, validation_output)
        created.append(validation_output)
        copy_file_atomic(args.token_byte_lengths, token_bytes_output)
        created.append(token_bytes_output)
        payload = {
            "schema_version": "targeted_8b_training_assets_v1",
            "status": "frozen",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "experiment": args.experiment,
            "base_recipe": file_binding(args.base_recipe),
            "pool_receipt": file_binding(args.pool_receipt),
            "schedule_manifest": file_binding(args.schedule_manifest),
            "recipe": file_binding(recipe_path),
            "execution_profiles": file_binding(profiles_path),
            "validation_manifest": file_binding(validation_output),
            "token_byte_lengths": file_binding(token_bytes_output),
            "dp64_allowed": False,
        }
        # The receipt is the commit marker for the four immutable assets.
        write_json_atomic(args.receipt, payload)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    print(json.dumps({"ok": True, "experiment": args.experiment, "updates": recipe["batch_and_parallelism"]["training_updates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

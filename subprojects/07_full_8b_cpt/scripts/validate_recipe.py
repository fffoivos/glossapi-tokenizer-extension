#!/usr/bin/env python3
"""Validate the full-8B scientific geometry and optional frozen data receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--pool-receipt", type=Path)
    parser.add_argument("--schedule-manifest", type=Path)
    args = parser.parse_args()
    recipe = read(args.recipe)
    if recipe.get("schema_version") != "apertus_full_8b_mixed_cpt_recipe_v1":
        raise ValueError("unsupported recipe")
    if recipe.get("recipe_id") == "full8b-mixed-79-20-1-wsd10-sanitized-v1":
        sanitized = recipe.get("data", {}).get("sanitized_source_receipt", {})
        eligibility = recipe.get("data", {}).get("eligibility_policy", {})
        dropout = recipe.get("initialization", {}).get(
            "token_distillation_dropout_context", {}
        )
        if (
            not sanitized.get("sha256")
            or eligibility.get("openarchives_needs_ocr_true") != "excluded"
            or dropout
            != {
                "model_mode_during_teacher_hidden_state_extraction": "train",
                "attention_dropout": 0.1,
                "hidden_dropout": 0.1,
                "existing_verified_initialization_preserved": True,
                "claim": "dropout_active_token_distillation_initialization",
            }
        ):
            raise ValueError("sanitized recipe disclosure drift")
    tokenizer = recipe["tokenizer"]
    if tokenizer["vocab_size"] != 148_992 or tokenizer["vocab_size"] % tokenizer["alignment_divisor"]:
        raise ValueError("tokenizer alignment drift")
    if tokenizer["modern_added_tokens"] + tokenizer["polytonic_added_tokens"] != 17_920:
        raise ValueError("both extension stages are not present")
    batch = recipe["batch_and_parallelism"]
    if batch["global_batch_tokens"] != batch["global_batch_sequences"] * recipe["data"]["sequence_length"]:
        raise ValueError("global token batch drift")
    if batch["world_size"] != batch["nodes"] * batch["gpus_per_node"]:
        raise ValueError("world-size drift")
    if batch["data_parallel"] != batch["world_size"] // batch["tensor_parallel"]:
        raise ValueError("data-parallel geometry drift")
    divisor = batch["data_parallel"] * batch["micro_batch_sequences"]
    if batch["global_batch_sequences"] % divisor:
        raise ValueError("global batch is not divisible by DP times microbatch")
    if batch["gradient_accumulation_steps"] != batch["global_batch_sequences"] // divisor:
        raise ValueError("gradient accumulation drift")
    if batch["training_samples"] != batch["training_updates"] * batch["global_batch_sequences"]:
        raise ValueError("training sample horizon drift")
    if recipe["data"]["planning_training_slots_tokens"] != batch["training_samples"] * recipe["data"]["sequence_length"]:
        raise ValueError("training token-slot horizon drift")
    if recipe["optimization"]["nan_and_inf_checks_enabled"] is not True:
        raise ValueError("NaN/Inf checks must remain enabled")
    rope = recipe["model"]["rope"]
    if (recipe["model"]["max_position_embeddings"], rope["base"], rope["use_scaling"], rope["scaling_factor"]) != (4096, 500000, True, 8.0):
        raise ValueError("corrected RoPE geometry drift")
    boundaries = recipe["segments"]["boundaries"]
    if boundaries[0] != 0 or boundaries[-1] != batch["training_updates"] or any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("segment boundaries do not cover the run exactly")
    if recipe["segments"]["count"] != len(boundaries) - 1:
        raise ValueError("segment-count drift")
    checkpoint_updates = set(recipe["evaluation"]["checkpoint_updates"])
    if not set(boundaries[1:]).issubset(checkpoint_updates):
        raise ValueError("every segment boundary must be an exact checkpoint")
    greek = recipe["evaluation"]["greekmmlu"]["checkpoint_updates"]
    if greek[0] != 0 or greek[-1] != batch["training_updates"] or any(b <= a for a, b in zip(greek, greek[1:])):
        raise ValueError("GreekMMLU milestones are not ordered and terminal")
    if set(greek[1:]) - checkpoint_updates:
        raise ValueError("GreekMMLU milestone lacks a saved checkpoint")
    if recipe["evaluation"]["checkpoint_averaging"] is not False:
        raise ValueError("checkpoint averaging was explicitly excluded")
    if len(recipe["evaluation"]["source_conditioned"]["panels"]) != 13:
        raise ValueError("source-conditioned panel count drift")
    if (
        recipe.get("recipe_id") == "full8b-mixed-79-20-1-wsd10-sanitized-v1"
        and recipe["evaluation"]["source_conditioned"]["interval_updates"] != 238
    ):
        raise ValueError("source-conditioned validation cadence drift")
    if recipe.get("recipe_id") == "full8b-mixed-79-20-1-wsd10-sanitized-v1":
        retention = recipe["evaluation"].get("retention_alerts", {})
        if (
            retention.get("status") != "pre_registered_before_sanitized_restart"
            or retention.get("metric") != "fixed_panel_mean_nll_nats"
            or retention.get("reference")
            != {"type": "running_minimum_after_warmup", "start_update": 400}
            or retention.get("warning")
            != {"any_panel_increase_nats": 0.05, "consecutive_observations": 2}
            or retention.get("critical")
            != {
                "any_panel_increase_nats": 0.08,
                "macro_mean_increase_nats": 0.05,
                "consecutive_observations": 2,
            }
            or retention.get("action", {}).get("automatic_training_stop") is not False
        ):
            raise ValueError("pre-registered retention alert policy drift")
        disclosure = recipe.get("provenance_disclosures", {}).get(
            "openarchives_needs_ocr", ""
        )
        if "excludes exactly 6,648" not in disclosure:
            raise ValueError("sanitized OpenArchives exclusion disclosure drift")
    libduth = recipe["data"]["source_dataset"]["libduth"]
    if libduth["legal_conclusion_claimed"] is not False:
        raise ValueError("recipe must not manufacture a libduth legal conclusion")
    if "D0_selection_confirmed_after_per_document_rerun_or_explicit_point_estimate_acceptance" not in recipe["launch_gates"]:
        raise ValueError("D0 selection uncertainty gate missing")
    if "libduth_permission_evidence_conflict_is_reconciled_or_explicitly_accepted" not in recipe["launch_gates"]:
        raise ValueError("libduth permission evidence gate missing")
    init = recipe["initialization"]
    if init["production_verification_file_sha256"] != "7b4adb065f401305064cda8c45cc1b2c1430366ef8fbf5ee060f154be5b48e26":
        raise ValueError("production initialization receipt hash drift")
    if init["roundtrip_verification_file_sha256"] != "06ddc69f4787f470b85d4be55992f898b98d8ab4fd8ab34a89a4d3bb96eb7e33":
        raise ValueError("round-trip receipt hash drift")
    software = recipe["software"]
    if software["megatron_upstream_commit"] != "c92402e39ef3c8e69ea378a59e79059dc14541f4":
        raise ValueError("Megatron upstream commit drift")
    patches = {row["path"]: row["sha256"] for row in software["megatron_runtime_patchset"]["patches"]}
    expected_patches = {
        "subprojects/06_dataset_scheduling_experiments/training/runtime_patches/megatron_extra_valid_c92402e.patch": "2e6810fa8b6c25597ccb3bcb9dc1ff5bf843ead2337e3edde0344605a23ec4c6",
        "subprojects/06_dataset_scheduling_experiments/training/runtime_patches/megatron_exact_eval_iterations_c92402e.patch": "6d9392cfb0dd08e62089d0a98e2817b222bb9a25ee5cefa8f3cdf29a8ce16bea",
    }
    if patches != expected_patches or software["megatron_runtime_patchset"]["runtime_receipt_required"] is not True:
        raise ValueError("Megatron runtime patch contract drift")
    runtime = software["validated_runtime_receipt"]
    if runtime["file_sha256"] != "99b9ecbd49bec162941d1b9bd11996b39ab421f4c43eca962c09ca37fdc7b36a":
        raise ValueError("Megatron runtime receipt file hash drift")
    if runtime["git_diff_sha256"] != "550a9f570a99f1ca20773bd2d97795210ce153ace1be794cf9f11aeb4b2238e6":
        raise ValueError("Megatron runtime diff hash drift")
    if runtime["revalidate_at_every_job_start"] is not True:
        raise ValueError("Megatron runtime must be revalidated at every job start")

    if args.pool_receipt:
        pool = read(args.pool_receipt)
        if pool.get("schema_version") != "apertus_schedule_pool_corpus_v1" or pool.get("status") != "completed":
            raise ValueError("pool receipt is not complete")
        geometry = pool["integer_79_20_1_geometry"]
        if geometry["active_tokens"] != recipe["data"]["planning_post_dedup_active_tokens"]:
            raise ValueError("active-token receipt differs from the frozen recipe; regenerate the recipe")
        if pool["tokenizer"]["tokenizer_json_sha256"] != tokenizer["tokenizer_json_sha256"]:
            raise ValueError("pool/tokenizer binding drift")
    if args.schedule_manifest:
        schedule = read(args.schedule_manifest)
        if schedule.get("schema_version") != "apertus_data_order_schedules_v1" or schedule.get("status") != "completed":
            raise ValueError("schedule receipt is not complete")
        if int(schedule["common_contract"]["global_batch_sequences"]) != batch["global_batch_sequences"]:
            raise ValueError("schedule/global-batch drift")
        arms = {row["arm_id"]: row for row in schedule["arms"]}
        d0 = arms.get("D0_mixed")
        if d0 is None or int(d0["optimizer_updates"]) != batch["training_updates"]:
            raise ValueError("D0 schedule horizon drift")
        if int(d0["pool_active_tokens"]["H"] + d0["pool_active_tokens"]["G"] + d0["pool_active_tokens"]["F"] + d0["pool_active_tokens"]["O"]) != recipe["data"]["planning_post_dedup_active_tokens"]:
            raise ValueError("D0 active-token accounting drift")
    print(json.dumps({"ok": True, "recipe_id": recipe["recipe_id"], "updates": batch["training_updates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate all reused artifacts and freeze the branch recipe plus launch gate."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path

from contract_utils import atomic_json, file_binding, read_json, require, verify_bound_file


EXPECTED_MILESTONES = [10728, 11920, 13112, 13193]
EXPECTED_PANELS = [
    "hplt", "non_hplt", "openarchives", "greek_phd", "historical_polytonic",
    "english", "de", "ru", "zh", "code", "math", "old_greek",
    "neutral_external_modern_greek",
]


def validate_static(contract: dict) -> None:
    require(contract.get("schema_version") == "apertus_full8_early_cooldown_contract_v3", "contract schema drift")
    require(contract.get("status") == "owner_authorized", "contract is not owner-authorized")
    require(contract.get("experiment_id") == "C_iter9536_direct_paired_early_wsd10_v3", "experiment id drift")
    training = contract["training"]
    require(training["start_iteration"] == 9536 and training["intervention_iteration"] == 9536 and training["end_iteration"] == 13193, "iteration geometry drift")
    require(training["total_executed_scientific_updates"] == 3657, "scientific update count drift")
    require(training["branch_updates"] == 3657, "branch update count drift")
    require(training["train_samples"] == 13193 * 1024, "training sample count drift")
    require(training["branch_token_slots"] == 3657 * 1024 * 4096, "branch token-slot count drift")
    lr = training["learning_rate"]
    require(lr == {
        "style": "WSD", "cooldown_shape": "1-sqrt", "peak": 5.5e-5,
        "floor": 5.5e-6, "decay_samples": 3657 * 1024,
        "decay_updates": 3657, "decay_starts_at_iteration": 9536,
    }, "LR intervention drift")
    ademamix = training["ademamix"]
    require(ademamix["alpha_warmup_updates"] == 18284, "alpha horizon was shortened")
    require(ademamix["beta3_warmup_updates"] == 18284, "beta3 horizon was shortened")
    require(contract["evaluation"]["milestone_iterations"] == EXPECTED_MILESTONES, "evaluation milestone drift")
    require(contract["data"]["arm"] == "D0_mixed", "data arm drift")
    require(contract["allocation_policy"]["normal_allocations"] == 1, "allocation count drift")
    paired = contract["paired_same_allocation_gate"]
    require(paired["source_iteration"] == 9536 and paired["comparison_iteration"] == 9537, "paired-control iteration drift")
    require(paired["same_nodes_and_allocation"] is True, "paired control allocation drift")
    require(paired["reference_probe_saves_checkpoint"] is False, "reference probe must not save")
    require(paired["intervention_probe_saves_checkpoint"] is True and paired["intervention_probe_checkpoint_becomes_branch_state"] is True, "intervention checkpoint contract drift")
    require("grad norm" in paired["exact_fields"] and paired["historical_absolute_gradient_is_not_an_acceptance_criterion"] is True, "gradient parity contract drift")
    allocation = contract["allocation_policy"]
    require(allocation["branch_conservative_runtime_seconds"] + allocation["branch_reserve_seconds"] == 12 * 3600, "allocation budget does not close")


def verify_parent(contract: dict) -> tuple[dict, dict, dict]:
    parent_recipe_path = verify_bound_file(contract["parent"]["recipe"])
    schedule_path = verify_bound_file(contract["data"]["schedule_manifest"])
    validation_path = verify_bound_file(contract["data"]["validation_manifest"])
    verify_bound_file(contract["data"]["sequence_ids"], require_bytes=True)
    verify_bound_file(contract["data"]["active_tokens"], require_bytes=True)
    verify_bound_file(contract["parent"]["checkpoint_receipt"])
    verify_bound_file(contract["parent"]["training_log"])
    verify_bound_file(contract["parent"]["greekmmlu_baseline"])
    verify_bound_file(contract["parent"]["initial_validation_receipt"])

    checkpoint = Path(contract["parent"]["run_root"]) / "checkpoints/iter_0009536/.metadata"
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "parent checkpoint is incomplete")
    checkpoint_receipt = read_json(Path(contract["parent"]["checkpoint_receipt"]["path"]))
    require(checkpoint_receipt.get("schema_version") == "megatron_exact_checkpoint_view_v1" and checkpoint_receipt.get("iteration") == 9536, "parent checkpoint receipt drift")
    require(len(checkpoint_receipt.get("source_files", [])) == 131, "parent checkpoint receipt is not a complete file inventory")
    require(Path(checkpoint_receipt.get("source_checkpoint_root", "")).resolve() == checkpoint.parent.parent.resolve(), "parent checkpoint root drift")
    for row in checkpoint_receipt["source_files"]:
        payload = checkpoint.parent / row["relative_path"]
        require(payload.is_file() and payload.stat().st_size == int(row["bytes"]), f"parent checkpoint payload missing or resized: {payload}")
    metadata_rows = [row for row in checkpoint_receipt["source_files"] if row.get("relative_path") == ".metadata"]
    require(len(metadata_rows) == 1 and metadata_rows[0].get("sha256") == contract["parent"]["checkpoint_metadata_sha256"], "parent checkpoint metadata hash drift")
    per_document = Path(contract["parent"]["per_document_baseline_root"])
    receipts = sorted(per_document.glob("*.receipt.json"))
    require(len(receipts) == 13, "iteration-9536 per-document baseline does not have 13 panels")
    expected_model = Path(contract["parent"]["run_root"]) / "checkpoint_evaluations/iter_0009536/attempt_0/export/hf"
    observed_baseline_panels = []
    for path in receipts:
        row = read_json(path)
        require(row.get("status") == "completed" and int(row.get("aggregate", {}).get("documents", 0)) > 0, f"invalid baseline receipt: {path}")
        require(Path(row.get("model", "")).resolve() == expected_model.resolve(), f"baseline model drift: {path}")
        observed_baseline_panels.append(path.name.removesuffix(".receipt.json"))

    parent, schedule, validation = read_json(parent_recipe_path), read_json(schedule_path), read_json(validation_path)
    require(parent.get("recipe_id") == "full8b-mixed-79-20-1-wsd10-sanitized-v1", "parent recipe id drift")
    require(parent["batch_and_parallelism"]["training_updates"] == 18284, "parent horizon drift")
    require(parent["batch_and_parallelism"]["global_batch_sequences"] == 1024, "parent global batch drift")
    require(parent["optimization"]["alpha_warmup_updates"] == 18284, "parent alpha horizon drift")
    require(parent["optimization"]["beta3_warmup_updates"] == 18284, "parent beta3 horizon drift")
    require(parent["optimization"]["learning_rate"]["stable_until_update"] == 14627, "parent LR schedule drift")
    require(parent["model"]["rope"] == {"base": 500000, "position_embedding_type": "rope", "scaling_factor": 8, "use_scaling": True}, "RoPE geometry drift")
    require(parent["tokenizer"]["tokenizer_json_sha256"] == "bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b", "tokenizer drift")
    arms = {row["arm_id"]: row for row in schedule["arms"]}
    require("D0_mixed" in arms and arms["D0_mixed"]["training_slots"] == 18284 * 1024, "parent D0 geometry drift")
    require(arms["D0_mixed"]["sequence_ids"]["sha256"] == contract["data"]["sequence_ids"]["sha256"], "D0 sequence hash drift")
    require(arms["D0_mixed"]["active_tokens"]["sha256"] == contract["data"]["active_tokens"]["sha256"], "D0 active-token hash drift")
    observed_panels = [row["name"] for row in validation["panels"]]
    require(len(observed_panels) == len(set(observed_panels)) and sorted(observed_panels) == sorted(EXPECTED_PANELS), "validation panel identity/uniqueness drift")
    require(sorted(observed_baseline_panels) == sorted(EXPECTED_PANELS), "baseline panel identity drift")
    validation_inputs = {row["name"]: row["raw_jsonl"]["sha256"] for row in validation["panels"]}
    for path in receipts:
        row = read_json(path)
        name = path.name.removesuffix(".receipt.json")
        require(row.get("input", {}).get("sha256") == validation_inputs[name], f"baseline validation input drift: {name}")
    initial = read_json(Path(contract["parent"]["initial_validation_receipt"]["path"]))
    require(initial.get("checkpoint_iteration") == 9536 and Path(initial.get("model", "")).resolve() == expected_model.resolve(), "iteration-9536 initial-validation anchor drift")
    return parent, schedule, validation


def derive_recipe(parent: dict, contract: dict) -> dict:
    result = copy.deepcopy(parent)
    result["schema_version"] = "apertus_full8_early_cooldown_recipe_v2"
    result["recipe_id"] = contract["experiment_id"]
    result["status"] = "frozen"
    result["causal_intervention"] = {
        "parent_checkpoint_iteration": 9536,
        "parent_state_loaded_directly_from_full_hash_receipt": True,
        "paired_same_allocation_peak_and_cooldown_probe": True,
        "parent_schedule_arm": "D0_mixed",
        "same_parent_schedule_prefix": True,
        "changed_field": "optimization.learning_rate",
        "preserved_ademamix_horizon_updates": 18284,
    }
    result["batch_and_parallelism"]["training_updates"] = 13193
    result["batch_and_parallelism"]["training_samples"] = 13193 * 1024
    result["segments"] = {
        "partition": "normal", "account": "a0140", "wall_limit": "12:00:00",
        "boundaries": [9536, 13193], "count": 1, "signal_before_limit_seconds": 600,
        "resume_requires_optimizer_rng_sample_cursor_parity": True,
    }
    result["optimization"]["learning_rate"] = {
        "schedule": "WSD", "peak": 5.5e-5, "warmup_initial": 5.5e-6,
        "final": 5.5e-6, "warmup_updates": 400, "stable_until_update": 9536,
        "cooldown_updates": 3657, "cooldown_shape": "1-sqrt",
    }
    result["optimization"]["alpha_warmup_updates"] = 18284
    result["optimization"]["beta3_warmup_updates"] = 18284
    result["evaluation"]["source_conditioned"]["interval_updates"] = 238
    result["evaluation"]["checkpoint_updates"] = list(EXPECTED_MILESTONES)
    result["evaluation"]["greekmmlu"]["checkpoint_updates"] = list(EXPECTED_MILESTONES)
    result["evaluation"]["per_document_validation"]["milestone_updates"] = list(EXPECTED_MILESTONES)
    result["evaluation"]["checkpoint_averaging"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path)
    parser.add_argument("--scheduler-snapshot", type=Path)
    parser.add_argument("--recipe-output", type=Path)
    parser.add_argument("--launch-gate-output", type=Path)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    contract = read_json(args.contract)
    validate_static(contract)
    if args.static_only:
        print(json.dumps({"ok": True, "mode": "static", "experiment": contract["experiment_id"]}, sort_keys=True))
        return 0
    for value, name in ((args.code_bundle_receipt, "code bundle receipt"), (args.scheduler_snapshot, "scheduler snapshot"), (args.recipe_output, "recipe output"), (args.launch_gate_output, "launch gate output")):
        require(value is not None, f"deep validation requires {name}")
    parent, schedule, validation = verify_parent(contract)
    scheduler = read_json(args.scheduler_snapshot)
    require(scheduler.get("schema_version") == "apertus_early_cooldown_scheduler_snapshot_v1", "scheduler snapshot schema drift")
    require(scheduler.get("status") == "captured" and scheduler.get("partition") == "debug", "scheduler snapshot was not captured on debug")
    bundle = read_json(args.code_bundle_receipt)
    require(bundle.get("schema_version") == "apertus_mini_immutable_code_bundle_v1" and bundle.get("status") == "frozen" and bundle.get("kind") == "scientific", "code bundle receipt drift")
    recipe = derive_recipe(parent, contract)
    atomic_json(args.recipe_output, recipe)
    gate = {
        "schema_version": "apertus_full8_early_cooldown_launch_gate_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": contract["experiment_id"],
        "checks": {
            "owner_authorized_contract": True,
            "parent_checkpoint_complete": True,
            "parent_checkpoint_all_payload_files_hash_receipted": True,
            "parent_recipe_byte_bound": True,
            "exact_D0_schedule_byte_bound": True,
            "schedule_prefix_covers_branch_endpoint": 13193 * 1024 <= {row["arm_id"]: row for row in schedule["arms"]}["D0_mixed"]["training_slots"],
            "all_13_validation_panels_frozen": len(validation["panels"]) == 13,
            "iteration_9536_baselines_reused": True,
            "same_allocation_peak_and_cooldown_update_9537_probe_required": True,
            "alpha_beta3_parent_horizon_preserved": True,
            "same_allocation_one_update_paired_parity_required": True,
            "one_16_node_allocation_only": True,
            "fresh_debug_scheduler_snapshot": True,
        },
        "contract": file_binding(args.contract),
        "branch_recipe": file_binding(args.recipe_output),
        "code_bundle_receipt": file_binding(args.code_bundle_receipt),
        "scheduler_snapshot": file_binding(args.scheduler_snapshot),
        "parent_recipe": file_binding(Path(contract["parent"]["recipe"]["path"])),
        "schedule_manifest": file_binding(Path(contract["data"]["schedule_manifest"]["path"])),
        "validation_manifest": file_binding(Path(contract["data"]["validation_manifest"]["path"])),
        "runtime_evidence": {
            "parent_checkpoint_metadata": file_binding(Path(contract["parent"]["run_root"]) / "checkpoints/iter_0009536/.metadata"),
            "parent_checkpoint_receipt": file_binding(Path(contract["parent"]["checkpoint_receipt"]["path"])),
            "parent_training_log": file_binding(Path(contract["parent"]["training_log"]["path"])),
            "greekmmlu_baseline": file_binding(Path(contract["parent"]["greekmmlu_baseline"]["path"])),
            "initial_validation_baseline": file_binding(Path(contract["parent"]["initial_validation_receipt"]["path"])),
            "per_document_baselines": [file_binding(path) for path in sorted(Path(contract["parent"]["per_document_baseline_root"]).glob("*.receipt.json"))],
            "D0_sequence_ids": file_binding(Path(contract["data"]["sequence_ids"]["path"])),
            "D0_active_tokens": file_binding(Path(contract["data"]["active_tokens"]["path"])),
        },
    }
    require(all(gate["checks"].values()), "launch checks failed")
    atomic_json(args.launch_gate_output, gate)
    print(json.dumps({"ok": True, "gate": str(args.launch_gate_output), "recipe": str(args.recipe_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the fail-closed production launch gate for targeted experiment A or B."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from contract_utils import TOKENIZER_SHA256, file_binding, read_json, require, sha256_file, write_json_atomic


PASSING = {"accepted", "completed", "frozen", "passed", "promoted", "launch_ready"}


def passed(path: Path, schemas: set[str] | None = None) -> dict[str, Any]:
    value = read_json(path)
    if schemas is not None:
        require(value.get("schema_version") in schemas, f"{path}: schema drift")
    require(str(value.get("status", "")).lower() in PASSING, f"{path}: non-passing status")
    return value


def binding_matches(binding: dict[str, Any], path: Path) -> bool:
    return (
        Path(str(binding.get("path", ""))).resolve() == path.resolve()
        and int(binding.get("bytes", -1)) == path.stat().st_size
        and binding.get("sha256") == sha256_file(path)
    )


def verify_checkpoint_inventory(
    *, experiment: str, receipt_path: Path, recipe: dict[str, Any], schedule: dict[str, Any]
) -> tuple[dict[str, Any], Path, str]:
    receipt = read_json(receipt_path)
    if experiment == "A":
        require(
            receipt.get("schema_version") == "apertus_full_8b_initial_checkpoint_tree_v1"
            and str(receipt.get("status", "")).lower() in PASSING,
            "A initial checkpoint receipt drift",
        )
        root = Path(receipt.get("root", "")).resolve()
        require(
            root == Path(recipe["initialization"]["megatron_tp2_checkpoint"]).resolve(),
            "A initial checkpoint root differs from recipe",
        )
        rows = receipt.get("files", [])
        expected = {
            "latest_checkpointed_iteration.txt",
            "release/mp_rank_00/model_optim_rng.pt",
            "release/mp_rank_01/model_optim_rng.pt",
        }
        require(
            int(receipt.get("file_count", -1)) == len(rows) == 3
            and {row.get("relative_path") for row in rows} == expected,
            "A initial checkpoint inventory drift",
        )
        expected_tree = receipt.get("tree_sha256", "")
    else:
        continuation = schedule["continuation_contract"]
        require(
            receipt.get("schema_version") == "megatron_exact_checkpoint_view_v1"
            and int(receipt.get("iteration", -1)) == 9536
            and file_binding(receipt_path) == continuation.get("parent_checkpoint_receipt"),
            "B initial checkpoint receipt drift",
        )
        root = Path(receipt.get("source_iteration", "")).resolve()
        require(
            root == Path(continuation.get("parent_checkpoint_directory", "")).resolve(),
            "B initial checkpoint root differs from schedule",
        )
        rows = receipt.get("source_files", [])
        require(len(rows) == 131, "B initial checkpoint inventory count drift")
        expected_tree = receipt.get("source_tree_manifest_sha256", "")
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get("relative_path", ""))
        path = root / relative
        require(
            relative
            and relative not in seen
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == int(row.get("bytes", -1))
            and sha256_file(path) == row.get("sha256"),
            f"initial checkpoint file drift: {path}",
        )
        seen.add(relative)
    canonical = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    observed_tree = hashlib.sha256(canonical).hexdigest()
    require(observed_tree == expected_tree, "initial checkpoint tree hash drift")
    return receipt, root, observed_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--frozen-contract", type=Path, required=True)
    parser.add_argument("--training-assets", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--owner-authorization", type=Path, required=True)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--packed-receipt", type=Path, required=True)
    parser.add_argument("--packed-integrity", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--decontamination-summary", type=Path)
    parser.add_argument("--selected-pools", type=Path)
    parser.add_argument("--restart-smoke", type=Path, required=True)
    parser.add_argument("--graceful-stop-smoke", type=Path, required=True)
    parser.add_argument("--initial-validation", type=Path, required=True)
    parser.add_argument("--initial-greekmmlu", type=Path, required=True)
    parser.add_argument("--initial-per-document-root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--conversion-smoke", type=Path, required=True)
    parser.add_argument("--launch-environment", type=Path, required=True)
    parser.add_argument("--nested-sbatch-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable launch gate exists: {args.output}")
    sys.path.insert(0, str(args.code_root / "subprojects/07_full_8b_cpt/scripts"))
    from contract import scientific_digest, verify_code_bundle_receipt  # pylint: disable=import-error,import-outside-toplevel
    from validate_execution_profile import validate  # pylint: disable=import-error,import-outside-toplevel

    bundle = verify_code_bundle_receipt(args.code_bundle_receipt, args.code_root, "scientific")
    contract = passed(args.frozen_contract, {"apertus_targeted_8b_frozen_contracts_v1"})
    require(contract.get("selected_experiment") in {args.experiment, "both"}, "frozen-contract experiment drift")
    assets = passed(args.training_assets, {"targeted_8b_training_assets_v1"})
    require(assets.get("experiment") == args.experiment, "training-assets experiment drift")
    recipe = read_json(args.recipe)
    profiles = read_json(args.profiles)
    selected = passed(args.selected_profile, {"apertus_full_8b_selected_execution_profile_v1"})
    owner = passed(args.owner_authorization, {"apertus_targeted_8b_owner_authorization_v1"})
    require(args.experiment in owner.get("scope", []), "owner authorization scope drift")
    for decision in owner.get("decisions", {}).values():
        require(decision.get("accepted") is True, "owner authorization contains an unaccepted decision")
    require(binding_matches(assets["recipe"], args.recipe), "training-assets recipe binding drift")
    require(binding_matches(assets["execution_profiles"], args.profiles), "training-assets profile binding drift")
    require(selected.get("recipe", {}).get("sha256") == sha256_file(args.recipe), "selected profile recipe drift")
    require(selected.get("profiles", {}).get("sha256") == sha256_file(args.profiles), "selected profile candidates drift")
    selection = selected["selection"]
    validated = validate(recipe, profiles, selection["profile_id"])
    require(
        validated["scientific_digest"] == selection.get("scientific_digest")
        and scientific_digest(recipe) == selection.get("scientific_digest"),
        "selected profile scientific digest drift",
    )
    require(
        selection.get("profile_id") == "dp32_16node"
        and int(selection.get("nodes", -1)) == 16
        and int(selection.get("world_size", -1)) == 64
        and int(selection.get("data_parallel", -1)) == 32,
        "selected execution geometry drift",
    )
    require(profiles.get("scientific_invariants", {}).get("dp64_allowed") is False, "DP64 was enabled")
    require(recipe["tokenizer"]["tokenizer_json_sha256"] == TOKENIZER_SHA256, "tokenizer drift")
    require(
        recipe["model"]["rope"]
        == {
            "base": 500000,
            "position_embedding_type": "rope",
            "scaling_factor": 8.0,
            "use_scaling": True,
        },
        "RoPE drift",
    )
    require(recipe["model"]["max_position_embeddings"] == 4096, "context geometry drift")
    require(recipe["evaluation"]["checkpoint_averaging"] is False, "checkpoint averaging was enabled")
    require(recipe["optimization"]["nan_and_inf_checks_enabled"] is True, "NaN checks disabled")

    pool = passed(args.pool_receipt, {"apertus_schedule_pool_corpus_v1"})
    packed = passed(args.packed_receipt, {"apertus_packed_sequence_corpus_v1"})
    integrity = passed(
        args.packed_integrity,
        {"targeted_8b_packed_payload_integrity_v1", "apertus_full_8b_packed_payload_integrity_v1"},
    )
    require(integrity.get("packed_receipt", {}).get("sha256") == sha256_file(args.packed_receipt), "packed-integrity binding drift")
    manifests = packed.get("packing_task_manifests", [])
    require(
        int(integrity.get("manifest_count", -1)) == len(manifests)
        and int(integrity.get("payload_count", -1)) == 3 * len(manifests),
        "packed-integrity inventory drift",
    )
    schedule = passed(args.schedule_manifest, {"apertus_data_order_schedules_v1"})
    arms = {row["arm_id"]: row for row in schedule["arms"]}
    require(set(arms) == {"D0_mixed"} or "D0_mixed" in arms, "D0 schedule missing")
    arm = arms["D0_mixed"]
    require(int(arm["optimizer_updates"]) == int(recipe["batch_and_parallelism"]["training_updates"]), "schedule horizon drift")
    require(sum(int(value) for value in arm["pool_active_tokens"].values()) == int(pool["integer_79_20_1_geometry"]["active_tokens"]), "schedule/pool active-token drift")
    require(int(pool["integer_79_20_1_geometry"]["active_tokens"]) == int(recipe["data"]["planning_post_dedup_active_tokens"]), "recipe/pool token drift")
    require(schedule["packed_corpus_receipt"]["sha256"] == sha256_file(args.packed_receipt), "schedule/packed receipt drift")

    validation = read_json(args.validation_manifest)
    require(validation.get("schema_version") == "apertus_full_8b_validation_manifest_v1" and validation.get("status") == "frozen", "validation manifest drift")
    require(len(validation.get("panels", [])) == 13, "validation panel count drift")
    require(validation.get("all_panels_training_exact_content_disjoint") is True, "validation panels are contaminated")
    require(all(int(row.get("training_exact_content_overlap_documents", -1)) == 0 for row in validation["panels"]), "validation overlap count drift")
    overlap_binding = validation.get("training_content_overlap_audit", {})
    overlap_path = Path(overlap_binding.get("path", ""))
    require(binding_matches(overlap_binding, overlap_path), "validation overlap-audit binding drift")
    overlap_audit = passed(overlap_path, {"apertus_full_8b_all_panel_training_content_audit_v1"})
    require(
        overlap_audit.get("all_final_panels_training_disjoint") is True
        and len(overlap_audit.get("panels", [])) == 13
        and all(int(row.get("final_training_exact_content_overlap_documents", -1)) == 0 for row in overlap_audit["panels"]),
        "validation overlap audit failed",
    )

    if args.experiment == "A":
        require(args.decontamination_summary is not None and args.selected_pools is not None, "A selection receipts missing")
        decontamination = passed(args.decontamination_summary, {"targeted_8b_a_decontamination_summary_v1"})
        pools = passed(args.selected_pools, {"targeted_8b_a_selected_pools_v1"})
        require(decontamination.get("no_global_deduplication") is True, "A decontamination performed deduplication")
        require(pools.get("no_global_deduplication") is True, "A selected pools performed deduplication")
        require(int(pools["pool_active_tokens"]["academic"]) == int(pools["pool_active_tokens"]["hplt"]), "A academic/HPLT ratio drift")
        require(pools.get("pool_corpus_receipt") == file_binding(args.pool_receipt), "A selected-pools/pool binding drift")
        require(
            pools.get("decontamination_summary") == file_binding(args.decontamination_summary)
            and int(pools.get("greekmmlu_postscan_high_confidence_matches", -1)) == 0
            and int(pools.get("frozen_validation_exact_content_overlaps", -1)) == 0,
            "A selected-pool exclusion evidence drift",
        )
        targeted_data_gate = True
    else:
        continuation = schedule.get("continuation_contract", {})
        require(continuation.get("parent_checkpoint_iteration") == 9536, "B parent checkpoint drift")
        require(continuation.get("parent_prefix_is_byte_exact") is True, "B prefix drift")
        require(continuation.get("parent_prefix_overlap_selected_sequences") == 0, "B repeats parent-prefix sequences")
        require(continuation.get("absolute_final_optimizer_update") == 12290, "B final update drift")
        checkpoint_binding = continuation.get("parent_checkpoint_receipt", {})
        checkpoint_path = Path(checkpoint_binding.get("path", ""))
        require(binding_matches(checkpoint_binding, checkpoint_path), "B parent checkpoint receipt binding drift")
        checkpoint_source = read_json(checkpoint_path)
        require(
            checkpoint_source.get("schema_version") == "megatron_exact_checkpoint_view_v1"
            and int(checkpoint_source.get("iteration", -1)) == 9536,
            "B parent checkpoint receipt identity drift",
        )
        checkpoint_dir = Path(continuation.get("parent_checkpoint_directory", "")).resolve()
        require(
            checkpoint_dir
            == Path(checkpoint_source.get("source_checkpoint_root", "")).resolve() / "iter_0009536",
            "B parent checkpoint directory drift",
        )
        require(binding_matches(continuation.get("parent_checkpoint_metadata", {}), checkpoint_dir / ".metadata"), "B parent checkpoint metadata drift")
        targeted_data_gate = True

    restart = passed(args.restart_smoke, {"targeted_8b_restart_smoke_v1"})
    require(
        restart.get("code_bundle_receipt") == file_binding(args.code_bundle_receipt),
        "restart-smoke code-bundle binding drift",
    )
    require(restart.get("experiment") == args.experiment and restart.get("profile_id") == "dp32_16node", "restart-smoke identity drift")
    require(restart.get("schedule_manifest") == file_binding(args.schedule_manifest), "restart-smoke schedule drift")
    require(restart.get("recipe") == file_binding(args.recipe), "restart-smoke recipe drift")
    require(restart.get("profiles") == file_binding(args.profiles), "restart-smoke profiles drift")
    require(restart.get("selected_profile") == file_binding(args.selected_profile), "restart-smoke profile drift")
    require(restart.get("thresholds") == profiles.get("restart_parity"), "restart-smoke threshold drift")
    restart_allocation = restart.get("allocation", {})
    require(
        restart_allocation.get("normal_nodes") == 16
        and restart_allocation.get("single_leaf") is True
        and restart_allocation.get("single_allocation_for_control_and_resume") is True
        and restart_allocation.get("total_optimizer_updates_executed") == 3,
        "restart-smoke allocation drift",
    )
    restart_checks = restart.get("checks", {})
    require(
        restart_checks.get("uninterrupted_two_update_run_passed") is True
        and restart_checks.get("resume_from_exact_control_checkpoint_passed") is True
        and restart_checks.get("control_checkpoint_boundary_proven") is True
        and restart_checks.get("checkpoint_sample_cursor_exact") is True
        and restart_checks.get("first_post_checkpoint_update_within_frozen_bounds") is True
        and restart_checks.get("nonfinite_updates") == 0
        and restart_checks.get("skipped_updates") == 0,
        "restart trajectory failed",
    )
    graceful = passed(args.graceful_stop_smoke, {"apertus_full_8b_graceful_stop_smoke_v1"})
    require(graceful.get("profile_id") == "dp32_16node" and graceful.get("resume", {}).get("passed") is True, "graceful-stop smoke drift")
    initial_validation = passed(args.initial_validation, {"targeted_8b_initial_validation_v1"})
    expected_iteration = 0 if args.experiment == "A" else 9536
    initial_validation_manifest = initial_validation.get("validation_manifest", {})
    initial_validation_manifest_path = Path(initial_validation_manifest.get("path", ""))
    require(
        initial_validation.get("experiment") == args.experiment
        and int(initial_validation.get("checkpoint_iteration", -1)) == expected_iteration
        and binding_matches(initial_validation_manifest, initial_validation_manifest_path)
        and initial_validation_manifest.get("sha256") == sha256_file(args.validation_manifest)
        and int(initial_validation_manifest.get("bytes", -1)) == args.validation_manifest.stat().st_size
        and len(initial_validation.get("panels", [])) == 13,
        "initial validation drift",
    )
    require(all(math.isfinite(float(row["bpb"])) and math.isfinite(float(row["lm_loss"])) for row in initial_validation["panels"]), "initial validation non-finite")
    model_root = Path(initial_validation.get("model", "")).resolve()
    require(model_root.is_dir(), "initial validation model root missing")
    initial_greek = passed(
        args.initial_greekmmlu,
        {"apertus_full_8b_initial_greekmmlu_v1", "exact_checkpoint_native_greekmmlu_receipt_v1", "targeted_8b_initial_greekmmlu_v1"},
    )
    greek_dataset = initial_greek.get("dataset", {})
    require(
        greek_dataset.get("source") == recipe["evaluation"]["greekmmlu"]["dataset"]
        and greek_dataset.get("revision") == recipe["evaluation"]["greekmmlu"]["revision"]
        and greek_dataset.get("resolved_split") == "test"
        and int(greek_dataset.get("rows_before_sampling", -1)) == int(recipe["evaluation"]["greekmmlu"]["public_items"])
        and greek_dataset.get("fingerprint"),
        "initial GreekMMLU dataset drift",
    )
    if args.experiment == "A":
        require(Path(initial_greek.get("model", "")).resolve() == model_root, "A GreekMMLU model drift")
        require(
            initial_greek.get("model_config", {}).get("rope_theta") == 500000.0
            and initial_greek.get("model_config", {}).get("max_position_embeddings") == 4096,
            "A GreekMMLU model geometry drift",
        )
    else:
        require(initial_greek.get("checkpoint", {}).get("iteration") == 9536, "B GreekMMLU checkpoint drift")
    checkpoint, checkpoint_root, checkpoint_tree = verify_checkpoint_inventory(
        experiment=args.experiment,
        receipt_path=args.initial_checkpoint_receipt,
        recipe=recipe,
        schedule=schedule,
    )
    conversion = passed(args.conversion_smoke, {"targeted_8b_conversion_greekmmlu_smoke_v1"})
    require(
        conversion.get("experiment") == args.experiment
        and int(conversion.get("checkpoint_iteration", -1)) == expected_iteration
        and Path(conversion.get("model", "")).resolve() == model_root
        and conversion.get("geometry", {}).get("rope_theta") == 500000
        and conversion.get("geometry", {}).get("max_position_embeddings") == 4096
        and conversion.get("geometry", {}).get("tie_word_embeddings") is False
        and conversion.get("geometry", {}).get("vocab_size") == 148992
        and conversion.get("tokenizer_semantically_identical_to_frozen_overlay") is True
        and conversion.get("exact_weight_mapping_passed") is True,
        "conversion/GreekMMLU smoke drift",
    )
    source_validation = recipe.get("evaluation", {}).get("source_conditioned", {})
    require(
        int(source_validation.get("interval_updates", -1)) == 25
        and int(source_validation.get("cadence_tokens_approx", -1)) == 104_857_600,
        "source-validation cadence receipt/executable drift",
    )
    greek_cadence = recipe.get("evaluation", {}).get("greekmmlu", {})
    expected_greek_tokens = 2_000_000_000 if args.experiment == "A" else 1_000_000_000
    expected_greek_updates = 477 if args.experiment == "A" else 238
    require(
        int(greek_cadence.get("cadence_active_tokens", -1)) == expected_greek_tokens
        and int(greek_cadence.get("cadence_updates", -1)) == expected_greek_updates,
        "GreekMMLU cadence receipt/checkpoint-plan drift",
    )
    environment = passed(args.launch_environment, {"apertus_full_8b_launch_environment_v1"})
    require(int(environment.get("nodes", -1)) == 16, "launch environment node drift")
    require(environment.get("checks", {}).get("storage_headroom_at_least_6TB") is True, "storage headroom gate failed")
    require(environment.get("checks", {}).get("normal_partition_snapshot_succeeded") is True, "scheduler snapshot failed")
    require(environment.get("checks", {}).get("test_only_prediction_succeeded") is True, "Slurm test-only failed")
    placement = environment.get("placement", {})
    test_only_command = environment.get("test_only", {}).get("command", [])
    require(
        str(placement.get("leaf_switch", "")).startswith("group")
        and placement.get("excluded_nodes")
        and "--switches=1" in test_only_command
        and any(str(value).startswith("--exclude=") for value in test_only_command),
        "scheduler single-leaf placement evidence drift",
    )
    nested = passed(args.nested_sbatch_proof, {"apertus_full_8b_nested_sbatch_proof_v1"})
    require(
        Path(nested.get("code_root", "")).resolve() == args.code_root.resolve()
        and Path(nested.get("code_bundle_receipt", "")).resolve() == args.code_bundle_receipt.resolve()
        and nested.get("parent_runtime") == "uenv run pytorch/v2.9.1:v2 --view=default -- python3"
        and nested.get("nested_submit_flag") == "--uenv-passthrough=ignore"
        and nested.get("rank_runtime") == "uenv run pytorch/v2.9.1:v2 --view=default -- torchrun"
        and nested.get("rank_torchrun")
        and nested.get("rank_megatron_import") == "megatron-import-ok"
        and nested.get("parent_job_id")
        and nested.get("child_job_id"),
        "nested-submit runtime proof drift",
    )
    per_document = sorted(args.initial_per_document_root.glob("*.receipt.json"))
    require(len(per_document) == 13, "initial per-document panel count drift")
    panel_names = {row["name"] for row in validation["panels"]}
    require({path.name.removesuffix(".receipt.json") for path in per_document} == panel_names, "initial per-document panel identity drift")
    for path in per_document:
        value = passed(path, {"apertus_per_document_validation_v1"})
        require(int(value.get("aggregate", {}).get("documents", 0)) > 0, f"empty per-document panel: {path}")
        panel = path.name.removesuffix(".receipt.json")
        expected_input = {row["name"]: row for row in validation["panels"]}[panel]["raw_jsonl"]
        observed_input = value.get("input", {})
        require(
            Path(observed_input.get("path", "")).resolve() == Path(expected_input["path"]).resolve()
            and observed_input.get("sha256") == expected_input["sha256"]
            and Path(value.get("model", "")).resolve() == model_root,
            f"per-document evidence binding drift: {panel}",
        )

    actual_gates = {
        "frozen_pool_corpus_receipt_matches_recipe": True,
        "D0_selection_confirmed_after_per_document_rerun_or_explicit_point_estimate_acceptance": owner["decisions"]["D0_selection_confirmed_after_per_document_rerun_or_explicit_point_estimate_acceptance"]["accepted"] is True,
        "libduth_permission_evidence_conflict_is_reconciled_or_explicitly_accepted": owner["decisions"]["libduth_permission_evidence_conflict_is_reconciled_or_explicitly_accepted"]["accepted"] is True,
        "packed_schedule_receipt_matches_recipe": True,
        "all_13_validation_panels_are_frozen": True,
        "tokenizer_and_initialization_hashes_pass": bool(checkpoint_tree),
        "nan_checks_are_enabled": True,
        "initial_validation_is_finite": True,
        "two_update_train_and_resume_smoke_passes": True,
        "graceful_stop_and_resume_smoke_passes": True,
        "greekmmlu_checkpoint_conversion_and_eval_smoke_passes": str(conversion.get("status", "")).lower() in PASSING and str(initial_greek.get("status", "")).lower() in PASSING,
        "initial_per_document_validation_all_13_panels_passes": True,
        "at_least_6TB_checkpoint_conversion_and_evaluation_space_is_available": True,
        "fresh_scheduler_snapshot_is_recorded": True,
        "explicit_production_launch_authorization_is_received": owner["decisions"]["explicit_production_launch_authorization_is_received"]["accepted"] is True,
        "targeted_frozen_data_and_schedule_receipts_pass": targeted_data_gate,
        "targeted_decontamination_and_validation_exclusion_audits_pass": targeted_data_gate,
        "dp32_restart_and_two_update_smoke_pass": True,
        "initial_source_validation_and_greekmmlu_anchor_pass": True,
    }
    gates = {name: bool(actual_gates.get(name, False)) for name in recipe["launch_gates"]}
    require(all(gates.values()), f"launch gates failed: {[name for name, value in gates.items() if not value]}")
    evidence_paths = {
        "code_bundle": args.code_bundle_receipt,
        "frozen_contract": args.frozen_contract,
        "training_assets": args.training_assets,
        "recipe": args.recipe,
        "profiles": args.profiles,
        "selected_profile": args.selected_profile,
        "owner_authorization": args.owner_authorization,
        "pool": args.pool_receipt,
        "packed": args.packed_receipt,
        "packed_integrity": args.packed_integrity,
        "schedule": args.schedule_manifest,
        "validation": args.validation_manifest,
        "validation_training_content_audit": overlap_path,
        "restart_smoke": args.restart_smoke,
        "graceful_stop_smoke": args.graceful_stop_smoke,
        "initial_validation": args.initial_validation,
        "initial_greekmmlu": args.initial_greekmmlu,
        "initial_checkpoint": args.initial_checkpoint_receipt,
        "conversion_smoke": args.conversion_smoke,
        "launch_environment": args.launch_environment,
        "nested_sbatch_proof": args.nested_sbatch_proof,
    }
    if args.decontamination_summary is not None:
        evidence_paths["decontamination_summary"] = args.decontamination_summary
    if args.selected_pools is not None:
        evidence_paths["selected_pools"] = args.selected_pools
    value = {
        "schema_version": "apertus_full_8b_launch_gate_v1",
        "status": "passed",
        "experiment": args.experiment,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recipe_id": recipe["recipe_id"],
        "scientific_digest": selection["scientific_digest"],
        "selected_profile": selection,
        "gates": gates,
        "evidence": {name: file_binding(path) for name, path in evidence_paths.items()},
        "initial_per_document": [file_binding(path) for path in per_document],
        "checkpoint_averaging": False,
        "executing_code_bundle": {"root": str(args.code_root.resolve()), "tree_sha256": bundle["tree_sha256"]},
        "initialization_checkpoint": {
            "root": str(checkpoint_root),
            "tree_sha256": checkpoint_tree,
            "iteration": 0 if args.experiment == "A" else 9536,
        },
    }
    write_json_atomic(args.output, value)
    print(json.dumps({"ok": True, "experiment": args.experiment, "gates": len(gates)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

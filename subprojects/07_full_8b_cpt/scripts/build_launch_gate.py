#!/usr/bin/env python3
"""Build the sole authoritative full-8B production launch gate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json, scientific_digest, sha256_file, verify_code_bundle_receipt
from validate_execution_profile import validate


def status(path: Path, schemas: set[str]) -> dict:
    value = read_json(path)
    if value.get("schema_version") not in schemas:
        raise ValueError(f"{path}: schema drift")
    if str(value.get("status", "")).lower() not in {"accepted", "completed", "frozen", "passed", "promoted"}:
        raise ValueError(f"{path}: non-passing status")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--owner-decisions", type=Path, required=True)
    parser.add_argument("--hf-visibility", type=Path, required=True)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--packed-receipt", type=Path, required=True)
    parser.add_argument("--packed-payload-integrity", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--token-bytes-receipt", type=Path, required=True)
    parser.add_argument("--benchmark-receipt", type=Path, required=True)
    parser.add_argument("--graceful-stop-smoke", type=Path, required=True)
    parser.add_argument("--initial-validation", type=Path, required=True)
    parser.add_argument("--initial-greekmmlu", type=Path, required=True)
    parser.add_argument("--initial-per-document-root", type=Path, required=True)
    parser.add_argument("--initial-megatron", type=Path, required=True)
    parser.add_argument("--initial-checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--initial-hf-receipt", type=Path, required=True)
    parser.add_argument("--conversion-smoke", type=Path, required=True)
    parser.add_argument("--launch-environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = verify_code_bundle_receipt(args.code_bundle_receipt, args.code_root)
    recipe = read_json(args.recipe)
    selected = status(args.selected_profile, {"apertus_full_8b_selected_execution_profile_v1"})
    validated = validate(recipe, read_json(args.profiles), selected["selection"]["profile_id"])
    if validated["scientific_digest"] != selected["selection"]["scientific_digest"] or validated["scientific_digest"] != scientific_digest(recipe):
        raise ValueError("selected profile scientific digest drift")
    owner = status(args.owner_decisions, {"apertus_full_8b_owner_decisions_v1"})
    required_owner = (
        "D0_selection_confirmed_after_per_document_rerun_or_explicit_point_estimate_acceptance",
        "libduth_permission_evidence_conflict_is_reconciled_or_explicitly_accepted",
        "explicit_production_launch_authorization_is_received",
    )
    if any(owner.get("decisions", {}).get(name, {}).get("accepted") is not True for name in required_owner):
        raise ValueError("owner decision gate drift")
    visibility = status(args.hf_visibility, {"apertus_full_8b_hf_visibility_v1"})
    if visibility.get("public") is not True:
        raise ValueError("dataset v2 is not public")
    pool = status(args.pool_receipt, {"apertus_schedule_pool_corpus_v1"})
    packed = status(args.packed_receipt, {"apertus_packed_sequence_corpus_v1"})
    packed_integrity = status(args.packed_payload_integrity, {"apertus_full_8b_packed_payload_integrity_v1"})
    if packed_integrity.get("packed_receipt", {}).get("sha256") != sha256_file(args.packed_receipt):
        raise ValueError("packed payload integrity/receipt drift")
    if int(packed_integrity.get("manifest_count", -1)) != 512 or int(packed_integrity.get("payload_count", -1)) != 1536:
        raise ValueError("packed payload integrity inventory drift")
    schedule = status(args.schedule_manifest, {"apertus_data_order_schedules_v1"})
    validation = read_json(args.validation_manifest)
    if validation.get("schema_version") != "apertus_full_8b_validation_manifest_v1" or validation.get("status") != "frozen" or len(validation.get("panels", [])) != 13:
        raise ValueError("validation manifest drift")
    status(args.token_bytes_receipt, {"apertus_token_utf8_byte_lengths_v1"})
    benchmark = status(args.benchmark_receipt, {"apertus_full_8b_parallelism_benchmark_v1"})
    if benchmark.get("selected_profile") != selected["selection"]["profile_id"]:
        raise ValueError("benchmark/selected profile drift")
    graceful_smoke = status(args.graceful_stop_smoke, {"apertus_full_8b_graceful_stop_smoke_v1"})
    if graceful_smoke.get("profile_id") != selected["selection"]["profile_id"] or graceful_smoke.get("resume", {}).get("passed") is not True:
        raise ValueError("graceful-stop smoke/profile drift")
    initial_validation = status(args.initial_validation, {"apertus_full_8b_initial_validation_v1"})
    initial_validation_finite = len(initial_validation.get("panels", [])) == 13 and not any(
        not math.isfinite(float(row.get(metric, math.nan)))
        for row in initial_validation.get("panels", [])
        for metric in ("lm_loss", "bpb", "base_target_loss", "added_target_loss")
    )
    if not initial_validation_finite:
        raise ValueError("initial validation is incomplete or non-finite")
    validation_sha = sha256_file(args.validation_manifest)
    if initial_validation.get("validation_manifest_sha256") != validation_sha:
        raise ValueError("initial validation/manifest drift")
    initial_greek = status(args.initial_greekmmlu, {"apertus_full_8b_initial_greekmmlu_v1"})
    expected_geometry = {
        "rope_theta": float(recipe["model"]["rope"]["base"]),
        "max_position_embeddings": int(recipe["model"]["max_position_embeddings"]),
    }
    model_config = initial_greek.get("model_config", {})
    if any(model_config.get(name) != value for name, value in expected_geometry.items()):
        raise ValueError("iteration-zero GreekMMLU model geometry drift")
    initial_hf = status(args.initial_hf_receipt, {"apertus_full_8b_corrected_initial_hf_v1"})
    initial_hf_root = Path(initial_hf.get("model_root", "")).resolve()
    if (
        initial_hf_root != Path(initial_greek.get("model", "")).resolve()
        or initial_hf.get("geometry") != expected_geometry
        or initial_hf.get("config_changed_fields") != ["max_position_embeddings", "rope_theta"]
        or initial_hf.get("zero_tensor_and_logit_drift") is not True
        or initial_hf.get("non_config_files_hardlinked_to_source") is not True
        or initial_hf.get("source_roundtrip_verification", {}).get("sha256")
        != recipe["initialization"]["roundtrip_verification_file_sha256"]
    ):
        raise ValueError("corrected iteration-zero HF anchor drift")
    hf_rows = initial_hf.get("files", [])
    if int(initial_hf.get("file_count", -1)) != 10 or len(hf_rows) != 10:
        raise ValueError("corrected iteration-zero HF inventory drift")
    for row in hf_rows:
        path = initial_hf_root / row["relative_path"]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"corrected iteration-zero HF file drift: {path}")
    hf_canonical = json.dumps(hf_rows, separators=(",", ":"), sort_keys=True).encode()
    if hashlib.sha256(hf_canonical).hexdigest() != initial_hf.get("tree_sha256"):
        raise ValueError("corrected iteration-zero HF tree drift")
    conversion_smoke = status(args.conversion_smoke, {"apertus_full_8b_conversion_greekmmlu_smoke_v1"})
    if conversion_smoke.get("selected_profile_id") != selected["selection"]["profile_id"]:
        raise ValueError("conversion smoke/profile drift")
    environment = status(args.launch_environment, {"apertus_full_8b_launch_environment_v1"})
    if environment.get("nodes") != selected["selection"]["nodes"]:
        raise ValueError("scheduler snapshot/profile node drift")
    per_document = sorted(args.initial_per_document_root.glob("*.receipt.json"))
    if len(per_document) != 13:
        raise ValueError("initial per-document receipt count drift")
    expected_panels = {row["name"]: row for row in validation["panels"]}
    observed_panels = set()
    for path in per_document:
        value = status(path, {"apertus_per_document_validation_v1"})
        if int(value.get("aggregate", {}).get("documents", 0)) <= 0:
            raise ValueError(f"empty per-document receipt: {path}")
        panel = path.name.removesuffix(".receipt.json")
        if panel not in expected_panels or panel in observed_panels:
            raise ValueError(f"per-document panel drift: {panel}")
        expected_input = expected_panels[panel]["raw_jsonl"]
        observed_input = value.get("input", {})
        if (
            Path(observed_input.get("path", "")).resolve() != Path(expected_input["path"]).resolve()
            or observed_input.get("sha256") != expected_input["sha256"]
            or Path(value.get("model", "")).resolve() != Path(initial_greek["model"]).resolve()
        ):
            raise ValueError(f"per-document evidence binding drift: {panel}")
        observed_panels.add(panel)
    if observed_panels != set(expected_panels):
        raise ValueError("initial per-document panel coverage drift")
    production_init = Path(recipe["initialization"]["production_verification_path"])
    roundtrip_init = Path(recipe["initialization"]["roundtrip_verification_path"])
    if sha256_file(production_init) != recipe["initialization"]["production_verification_file_sha256"] or sha256_file(roundtrip_init) != recipe["initialization"]["roundtrip_verification_file_sha256"]:
        raise ValueError("Token Distillation initialization receipt drift")
    initial_root = args.initial_megatron.resolve()
    if initial_root != Path(recipe["initialization"]["megatron_tp2_checkpoint"]).resolve():
        raise ValueError("initial checkpoint path differs from the recipe")
    initial_tree = status(args.initial_checkpoint_receipt, {"apertus_full_8b_initial_checkpoint_tree_v1"})
    if Path(initial_tree.get("root", "")).resolve() != initial_root:
        raise ValueError("initial checkpoint receipt/root drift")
    rows = initial_tree.get("files", [])
    expected_initial_files = {
        "latest_checkpointed_iteration.txt",
        "release/mp_rank_00/model_optim_rng.pt",
        "release/mp_rank_01/model_optim_rng.pt",
    }
    if int(initial_tree.get("file_count", -1)) != 3 or {
        row.get("relative_path") for row in rows
    } != expected_initial_files:
        raise ValueError("initial checkpoint inventory drift")
    observed = []
    for row in rows:
        path = initial_root / row["relative_path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"initial checkpoint file drift: {path}")
        observed.append(row)
    canonical = json.dumps(observed, separators=(",", ":"), sort_keys=True).encode()
    if hashlib.sha256(canonical).hexdigest() != initial_tree.get("tree_sha256"):
        raise ValueError("initial checkpoint tree hash drift")
    control_restart = benchmark.get("restart", {}).get("control", {})
    actual_gates = {
        "frozen_pool_corpus_receipt_matches_recipe": pool["integer_79_20_1_geometry"]["active_tokens"] == recipe["data"]["planning_post_dedup_active_tokens"],
        "D0_selection_confirmed_after_per_document_rerun_or_explicit_point_estimate_acceptance": owner["decisions"]["D0_selection_confirmed_after_per_document_rerun_or_explicit_point_estimate_acceptance"]["accepted"] is True,
        "libduth_permission_evidence_conflict_is_reconciled_or_explicitly_accepted": owner["decisions"]["libduth_permission_evidence_conflict_is_reconciled_or_explicitly_accepted"]["accepted"] is True,
        "packed_schedule_receipt_matches_recipe": packed.get("status") == "completed" and schedule.get("status") == "completed" and packed_integrity.get("status") == "passed",
        "all_13_validation_panels_are_frozen": len(validation["panels"]) == 13,
        "tokenizer_and_initialization_hashes_pass": pool["tokenizer"]["tokenizer_json_sha256"] == recipe["tokenizer"]["tokenizer_json_sha256"] and initial_tree.get("status") == "frozen" and initial_hf.get("status") == "completed",
        "nan_checks_are_enabled": recipe["optimization"]["nan_and_inf_checks_enabled"] is True,
        "initial_validation_is_finite": initial_validation_finite and initial_validation.get("validation_manifest_sha256") == validation_sha,
        "two_update_train_and_resume_smoke_passes": control_restart.get("provenance", {}).get("passed") is True and control_restart.get("numerical", {}).get("passed") is True,
        "graceful_stop_and_resume_smoke_passes": graceful_smoke.get("status") == "passed" and graceful_smoke.get("resume", {}).get("passed") is True,
        "greekmmlu_checkpoint_conversion_and_eval_smoke_passes": conversion_smoke.get("status") == "passed" and model_config.get("rope_theta") == expected_geometry["rope_theta"] and model_config.get("max_position_embeddings") == expected_geometry["max_position_embeddings"],
        "initial_per_document_validation_all_13_panels_passes": observed_panels == set(expected_panels),
        "at_least_6TB_checkpoint_conversion_and_evaluation_space_is_available": environment["checks"]["storage_headroom_at_least_6TB"] is True,
        "fresh_scheduler_snapshot_is_recorded": environment["checks"]["normal_partition_snapshot_succeeded"] is True and environment["checks"]["test_only_prediction_succeeded"] is True,
        "explicit_production_launch_authorization_is_received": owner["decisions"]["explicit_production_launch_authorization_is_received"]["accepted"] is True,
    }
    gates = {name: bool(actual_gates.get(name, False)) for name in recipe["launch_gates"]}
    if not all(gates.values()):
        raise ValueError(f"launch gates failed: {[name for name, passed in gates.items() if not passed]}")
    evidence = {
        name: file_binding(path)
        for name, path in {
            "code_bundle": args.code_bundle_receipt,
            "recipe": args.recipe,
            "profiles": args.profiles,
            "selected_profile": args.selected_profile,
            "owner_decisions": args.owner_decisions,
            "hf_visibility": args.hf_visibility,
            "pool": args.pool_receipt,
            "packed": args.packed_receipt,
            "packed_payload_integrity": args.packed_payload_integrity,
            "schedule": args.schedule_manifest,
            "validation": args.validation_manifest,
            "token_bytes": args.token_bytes_receipt,
            "benchmark": args.benchmark_receipt,
            "graceful_stop_smoke": args.graceful_stop_smoke,
            "initial_validation": args.initial_validation,
            "initial_greekmmlu": args.initial_greekmmlu,
            "conversion_smoke": args.conversion_smoke,
            "launch_environment": args.launch_environment,
            "production_initialization": production_init,
            "roundtrip_initialization": roundtrip_init,
            "initial_checkpoint_tree": args.initial_checkpoint_receipt,
            "initial_hf_anchor": args.initial_hf_receipt,
        }.items()
    }
    evidence["initial_per_document"] = [file_binding(path) for path in per_document]
    payload = {
        "schema_version": "apertus_full_8b_launch_gate_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recipe_id": recipe["recipe_id"],
        "scientific_digest": scientific_digest(recipe),
        "selected_profile": selected["selection"],
        "gates": gates,
        "evidence": evidence,
        "checkpoint_averaging": False,
        "executing_code_bundle": {"root": str(args.code_root.resolve()), "tree_sha256": bundle["tree_sha256"]},
        "initialization_checkpoint": {"root": str(initial_root), "tree_sha256": initial_tree["tree_sha256"]},
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "gates": len(gates), "selected_profile": selected["selection"]["profile_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

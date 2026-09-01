#!/usr/bin/env python3
"""Select DP32 from an exact-recipe synchronous restart-parity proof.

This is deliberately not a DP32/DP64 benchmark.  It is the fail-closed path
used when DP64 is not being reconsidered and production should use the already
approved DP32 fallback without spending another 32-node benchmark allocation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract import (
    atomic_write_json,
    file_binding,
    read_json,
    verify_code_bundle_receipt,
)
from finalize_parallelism_benchmark import parse_log
from validate_execution_profile import validate


PROFILE_ID = "dp32_16node"
REQUIRED_PARITY_CHECKS = {
    "control_completed_to_162",
    "synchronous_checkpoint_save",
    "single_leaf_switch_placement",
    "control_zero_skipped_updates",
    "control_zero_nonfinite_updates",
    "first_restart_provenance",
    "second_restart_provenance",
    "first_restart_numerically_equivalent",
    "second_restart_numerically_equivalent",
    "independent_restarts_identical",
}


def require_binding(receipt_binding: dict, path: Path, label: str) -> dict:
    observed = file_binding(path)
    if receipt_binding != observed:
        raise ValueError(f"{label} binding drift")
    return observed


def require_training_receipt(
    path: Path,
    *,
    scientific_digest: str,
    start: int,
    end: int,
) -> dict:
    value = read_json(path)
    if (
        value.get("schema_version") != "apertus_full_8b_training_job_v1"
        or value.get("status") != "completed"
        or value.get("profile_id") != PROFILE_ID
        or value.get("scientific_digest") != scientific_digest
        or value.get("checkpoint_save_mode") != "synchronous"
        or value.get("start_iteration") != start
        or value.get("end_iteration") != end
        or value.get("allocation", {}).get("single_leaf_switch") is not True
        or len(value.get("allocation", {}).get("leaf_switches", [])) != 1
    ):
        raise ValueError(f"training receipt drift: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--parity-code-root", type=Path, required=True)
    parser.add_argument("--parity-code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--parity-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection_bundle = verify_code_bundle_receipt(
        args.code_bundle_receipt, args.code_root
    )
    parity_bundle = verify_code_bundle_receipt(
        args.parity_code_bundle_receipt, args.parity_code_root
    )
    recipe = read_json(args.recipe)
    profiles = read_json(args.profiles)
    selected = validate(recipe, profiles, PROFILE_ID)
    profile = profiles.get("profiles", {}).get(PROFILE_ID, {})
    if profile.get("status") != "proven_fallback":
        raise ValueError("DP32 profile is not the declared proven fallback")

    parity_path = args.parity_root / "checkpoint_parity_receipt.json"
    parity = read_json(parity_path)
    parity_checks = parity.get("checks", {})
    if (
        parity.get("schema_version")
        != "apertus_full_8b_checkpoint_parity_smoke_v1"
        or parity.get("status") != "passed"
        or parity.get("profile_id") != PROFILE_ID
        or parity.get("checkpoint_save_mode") != "synchronous"
        or set(parity_checks) != REQUIRED_PARITY_CHECKS
        or not all(parity_checks.values())
    ):
        raise ValueError("DP32 checkpoint-parity receipt did not pass every frozen check")

    thresholds = profiles["benchmark"]["promotion"]
    expected_restart_thresholds = {
        "gradient_atol": thresholds["restart_gradient_norm_atol"],
        "gradient_rtol": thresholds["restart_gradient_norm_rtol"],
    }
    if parity.get("thresholds") != expected_restart_thresholds:
        raise ValueError("checkpoint-parity restart threshold drift")

    schedule_path = args.stage_root / "schedules/schedule_manifest.json"
    control_log = (
        args.parity_root / "control_dp32/segments/updates_0_162/training.log"
    )
    require_binding(parity["inputs"]["profiles"], args.profiles, "profiles")
    require_binding(
        parity["inputs"]["schedule_manifest"], schedule_path, "schedule manifest"
    )
    require_binding(parity["inputs"]["control_log"], control_log, "control log")

    control_receipt_path = (
        args.parity_root
        / "control_dp32/segments/updates_0_162/training_job_receipt.json"
    )
    first_restart_receipt_path = (
        args.parity_root
        / "control_dp32/segments/updates_160_161/training_job_receipt.json"
    )
    second_restart_receipt_path = (
        args.parity_root
        / "control_dp32_repeat/segments/updates_160_161/training_job_receipt.json"
    )
    control_receipt = require_training_receipt(
        control_receipt_path,
        scientific_digest=selected["scientific_digest"],
        start=0,
        end=162,
    )
    require_training_receipt(
        first_restart_receipt_path,
        scientific_digest=selected["scientific_digest"],
        start=160,
        end=161,
    )
    require_training_receipt(
        second_restart_receipt_path,
        scientific_digest=selected["scientific_digest"],
        start=160,
        end=161,
    )
    require_binding(
        parity["restart"]["first"]["provenance"]["receipt"],
        first_restart_receipt_path,
        "first restart receipt",
    )
    require_binding(
        parity["restart"]["second"]["provenance"]["receipt"],
        second_restart_receipt_path,
        "second restart receipt",
    )

    control = parse_log(control_log, require_full=False)
    if control["observations"] < 130:
        raise ValueError("DP32 parity control has too few timed updates")
    if control["skipped"] != 0 or control["nonfinite"] != 0:
        raise ValueError("DP32 parity control is numerically unhealthy")

    checks = {
        "selection_code_bundle_verified": True,
        "parity_code_bundle_verified": True,
        "exact_recipe_and_profile_validated": True,
        "profile_is_declared_proven_fallback": True,
        "parity_receipt_passed_all_frozen_checks": True,
        "parity_restart_thresholds_match_profiles": True,
        "parity_inputs_bind_exact_profiles_schedule_and_control_log": True,
        "control_and_restart_receipts_bind_scientific_digest": True,
        "control_and_restart_receipts_use_synchronous_checkpoints": True,
        "control_and_restart_allocations_are_single_leaf": True,
        "control_has_zero_skipped_and_nonfinite_updates": True,
        "two_independent_restart_allocations_passed": bool(
            parity["restart"]["first"]["provenance"]["passed"]
            and parity["restart"]["first"]["numerical"]["passed"]
            and parity["restart"]["second"]["provenance"]["passed"]
            and parity["restart"]["second"]["numerical"]["passed"]
            and parity["restart"]["independent_identity"]["passed"]
        ),
    }
    if not all(checks.values()):
        raise ValueError("DP32 fallback selection checks failed")

    training_updates = int(profiles["scientific_invariants"]["training_updates"])
    payload = {
        "schema_version": "apertus_full_8b_dp32_fallback_selection_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_policy": "explicit_dp32_fallback_from_exact_recipe_sync_restart_parity_v1",
        "selected_profile": PROFILE_ID,
        "candidate_profile": "dp64_32node",
        "candidate_evaluated": False,
        "candidate_promoted": False,
        "fallback_control_viable": True,
        "scientific_digest": selected["scientific_digest"],
        "checks": checks,
        "performance": {
            "scope": "DP32 parity control only; no DP64 speedup claim",
            "control_median_step_seconds": control["median_step_seconds"],
            "control_p90_step_seconds": control["p90_step_seconds"],
            "observations": control["observations"],
            "selected_compute_hours": training_updates
            * control["median_step_seconds"]
            / 3600,
        },
        "checkpointing": {
            "save_mode": "synchronous",
            "async_save_forbidden_for_resumable_boundaries": True,
        },
        "parity": {
            "receipt": file_binding(parity_path),
            "checks": parity_checks,
            "thresholds": expected_restart_thresholds,
            "two_independent_restart_allocations_passed": True,
        },
        "inputs": {
            "selection_code_bundle": {
                "root": str(args.code_root.resolve()),
                "receipt": file_binding(args.code_bundle_receipt),
                "tree_sha256": selection_bundle["tree_sha256"],
            },
            "parity_code_bundle": {
                "root": str(args.parity_code_root.resolve()),
                "receipt": file_binding(args.parity_code_bundle_receipt),
                "tree_sha256": parity_bundle["tree_sha256"],
                "finalizer": file_binding(
                    args.parity_code_root
                    / "subprojects/07_full_8b_cpt/scripts/finalize_checkpoint_parity_smoke.py"
                ),
            },
            "recipe": file_binding(args.recipe),
            "profiles": file_binding(args.profiles),
            "schedule_manifest": file_binding(schedule_path),
            "control_log": control["binding"],
            "control_job_receipt": file_binding(control_receipt_path),
            "control_job_id": control_receipt.get("job_id"),
        },
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "selected_profile": PROFILE_ID,
                "candidate_evaluated": False,
                "checks": checks,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

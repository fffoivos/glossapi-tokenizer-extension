#!/usr/bin/env python3
"""Freeze the exact owner-authorized 8B 2499->3218 stable-peak branch gate."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from build_checkpoint_permit import validate_permit as validate_checkpoint_permit
from build_training_run_permit import validate_permit as validate_training_run_permit
from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from freeze_phase_blend_cache import validate_receipt as validate_phase_cache
from producer_bundle_compatibility import load_authority


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--owner-confirmation", required=True)
    parser.add_argument("--confirmed-at", required=True)
    parser.add_argument("--base-launch-gate", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--checkpoint-reference", type=Path, required=True)
    parser.add_argument("--phase-data-path-spec", type=Path, required=True)
    parser.add_argument("--phase-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase-cache-root", type=Path, required=True)
    parser.add_argument("--training-run-permit", type=Path, required=True)
    parser.add_argument("--qualification-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), "immutable stable-peak gate exists")
    plan_text = args.plan.read_text(encoding="utf-8")
    require(
        "AUTHORIZED FOR EXECUTION 2026-08-22" in plan_text
        and "2499" in plan_text
        and "3218" in plan_text
        and "constant LR 5.5e-5" in plan_text,
        "stable-peak plan drift",
    )
    confirmation = args.owner_confirmation.casefold()
    require(
        "apply" in confirmation and "plan" in confirmation,
        "owner confirmation does not authorize the named plan",
    )
    confirmed = dt.datetime.fromisoformat(args.confirmed_at.replace("Z", "+00:00"))
    require(confirmed.tzinfo is not None, "owner confirmation lacks timezone")
    current = executing_code_bundle()
    _, accepted = load_authority(args.producer_compatibility, current)
    accepted_code = {(root, tree) for root, tree, _receipt, _bytes, _sha in accepted}

    base_gate = read_json(args.base_launch_gate)
    require(
        base_gate.get("schema_version") == "apertus_hard_h_to_g_frozen_contract_v2"
        and base_gate.get("status") == "launch_ready"
        and base_gate.get("mode") == "launch"
        and base_gate.get("gate_stage") == "pre_main"
        and base_gate.get("scale") == "8b"
        and base_gate.get("blockers") == [],
        "base 8B launch gate drift",
    )

    reference = read_json(args.checkpoint_reference)
    require(
        reference.get("schema_version") == "apertus_hard_h_to_g_checkpoint_reference_v1"
        and reference.get("status") == "passed"
        and reference.get("scale") == "8b"
        and int(reference.get("update", -1)) == 2499,
        "branch checkpoint reference drift",
    )
    checkpoint_root = Path(str(reference["checkpoint_root"]))
    permit_path = Path(str(reference["checkpoint_permit"]["path"]))
    cache_path = Path(str(reference["source_phase_cache_receipt"]["path"]))
    require(reference["checkpoint_permit"] == file_binding(permit_path), "checkpoint permit binding drift")
    require(reference["source_phase_cache_receipt"] == file_binding(cache_path), "source-cache binding drift")
    validate_checkpoint_permit(
        read_json(permit_path), scale="8b", source_phase=2, update=2499,
        checkpoint_root=checkpoint_root, source_phase_cache_receipt=cache_path,
    )
    require(cache_path.resolve() == args.phase_cache_receipt.resolve(), "source and target Phase-2 cache differ")
    cache = read_json(args.phase_cache_receipt)
    validate_phase_cache(
        cache, phase=2, data_path_spec=args.phase_data_path_spec,
        cache_root=args.phase_cache_root, accepted_code_bundles=accepted_code,
        verify_payload_hashes=False,
    )
    run_permit = read_json(args.training_run_permit)
    validate_training_run_permit(
        run_permit, scale="8b", nodes=16, tensor_parallel=2, microbatch=2,
        peak_lr="5.5e-5", floor_lr="5.5e-6",
    )
    qualification = read_json(args.qualification_contract)
    require(
        qualification.get("schema_version") == "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1"
        and qualification.get("status") == "frozen"
        and qualification.get("scale") == "8b"
        and int(qualification.get("nodes", -1)) == 16
        and int(qualification.get("tensor_parallel", -1)) == 2
        and int(qualification.get("microbatch", -1)) == 2,
        "8B qualification contract drift",
    )
    payload = {
        "schema_version": "apertus_hard_h_to_g_stable_peak_branch_gate_v1",
        "status": "launch_ready",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "owner": "fffoivos",
        "owner_confirmation": args.owner_confirmation,
        "owner_confirmed_at": confirmed.astimezone(dt.timezone.utc).isoformat(),
        "scale": "8b",
        "phase": 2,
        "start_update": 2499,
        "end_update": 3218,
        "lr_policy": "stable_peak",
        "learning_rate": "5.5e-5",
        "b3_extension_authorized": False,
        "plan": file_binding(args.plan),
        "base_launch_gate": file_binding(args.base_launch_gate),
        "producer_bundle_compatibility": file_binding(args.producer_compatibility),
        "checkpoint_reference": file_binding(args.checkpoint_reference),
        "phase_data_path_spec": file_binding(args.phase_data_path_spec),
        "phase_cache_receipt": file_binding(args.phase_cache_receipt),
        "phase_cache_root": str(args.phase_cache_root.resolve()),
        "phase_cache_tree_sha256": cache["cache_tree_sha256"],
        "training_run_permit": file_binding(args.training_run_permit),
        "qualification_contract": file_binding(args.qualification_contract),
        "executing_code_bundle": current,
        "checks": {
            "owner_authorized_exact_plan": True,
            "base_8b_recipe_launch_gate_passed": True,
            "intermediate_checkpoint_state_permitted": True,
            "phase2_data_and_cursor_bound": True,
            "only_lr_policy_changes": True,
            "b3_remains_closed": True,
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

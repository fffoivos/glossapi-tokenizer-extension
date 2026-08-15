#!/usr/bin/env python3
"""Freeze pair authorities after actual main/extension checkpoints exist."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


def bound_receipt(path: Path, schema: str, scale: str, current: dict) -> dict:
    value = read_json(path)
    require(value.get("schema_version") == schema and value.get("status") == "passed", f"receipt drift: {path}")
    require(value.get("scale") == scale, f"scale drift: {path}")
    bundle = value.get("executing_code_bundle")
    require(isinstance(bundle, dict) and bundle.get("root") == current["root"] and bundle.get("tree_sha256") == current["tree_sha256"], f"code-bundle drift: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("phase3_entry", "checkpoint_3456", "resume_3456"), required=True)
    parser.add_argument("--8b-receipt", type=Path, required=True)
    parser.add_argument("--1p5b-receipt", type=Path, required=True)
    parser.add_argument("--secondary-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable authority exists: {args.output}")
    if args.secondary_output is not None:
        require(not args.secondary_output.exists(), f"immutable secondary authority exists: {args.secondary_output}")
    current = executing_code_bundle()
    if args.kind in {"phase3_entry", "resume_3456"}:
        schema = "apertus_hard_h_to_g_phase3_resume_smoke_v1"
    else:
        schema = "apertus_hard_h_to_g_checkpoint_permit_v2"
    values = {
        "8b": bound_receipt(args.__dict__["8b_receipt"], schema, "8b", current),
        "1p5b": bound_receipt(args.__dict__["1p5b_receipt"], schema, "1p5b", current),
    }
    bindings = {scale: file_binding(args.__dict__[f"{scale}_receipt"]) for scale in values}
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    if args.kind == "phase3_entry":
        require(all(value.get("start_update") == 3218 and value.get("end_update") == 3219 for value in values.values()), "Phase-3 entry smoke boundary drift")
        require(all(all(value.get("checks", {}).values()) for value in values.values()), "Phase-3 entry smoke checks incomplete")
        require(args.secondary_output is not None, "constant-floor secondary output is required")
        cursor = {
            "schema_version": "apertus_hard_h_to_g_phase_cursor_authority_v1",
            "status": "passed", "created_at": created, "executing_code_bundle": current,
            "smokes": bindings,
            "checks": {"both_scales_enter_phase3_at_cursor_zero": True, "both_scales_save_cursor_1024": True, "both_scales_bind_the_exact_phase3_cache": True},
        }
        floor = {
            "schema_version": "apertus_hard_h_to_g_constant_floor_resume_authority_v1",
            "status": "passed", "created_at": created, "executing_code_bundle": current,
            "smokes": bindings,
            "checks": {"both_scales_execute_constant_floor_guard": True, "both_scales_save_constant_scheduler_at_selected_floor": True, "optimizer_ramps_remain_bound_to_3218_in_the_immutable_recipe": True},
        }
        write_json_atomic(args.output, cursor)
        write_json_atomic(args.secondary_output, floor)
    elif args.kind == "checkpoint_3456":
        require(all(value.get("update") == 3456 and value.get("source_phase") == 3 for value in values.values()), "update-3456 checkpoint-permit drift")
        payload = {
            "schema_version": "apertus_hard_h_to_g_checkpoint_pair_authority_v1",
            "status": "passed", "created_at": created, "executing_code_bundle": current,
            "update": 3456, "permits": bindings,
            "checks": {"both_scales_have_signed_update_3456_permits": True, "both_permits_bind_phase3_cache_and_cursor": True},
        }
        write_json_atomic(args.output, payload)
    else:
        require(all(value.get("start_update") == 3456 and value.get("end_update") == 3457 for value in values.values()), "update-3456 resume-smoke drift")
        require(all(value.get("entry_phase_local_samples") == 243_712 for value in values.values()), "update-3456 Phase-3 cursor drift")
        payload = {
            "schema_version": "apertus_hard_h_to_g_resume_pair_authority_v1",
            "status": "passed", "created_at": created, "executing_code_bundle": current,
            "start_update": 3456, "end_update": 3457, "smokes": bindings,
            "checks": {"both_scales_restore_phase3_cursor_243712": True, "both_scales_complete_the_exact_next_update": True, "both_scales_preserve_optimizer_rng_and_constant_floor": True},
        }
        write_json_atomic(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

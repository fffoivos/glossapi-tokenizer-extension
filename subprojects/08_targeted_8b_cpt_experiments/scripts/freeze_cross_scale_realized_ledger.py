#!/usr/bin/env python3
"""Prove both endpoint checkpoints consumed the same frozen Phase-1/2 trajectory."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from freeze_phase_blend_cache import validate_receipt as validate_phase_cache


EXPECTED_CACHE_SAMPLES = {1: 2_315_264, 2: 979_968}


def validate_cache(path: Path, phase: int) -> dict[str, Any]:
    value = read_json(path)
    data_path = Path(str(value.get("data_path_spec", {}).get("path", "")))
    cache_root = Path(str(value.get("cache_root", "")))
    validate_phase_cache(value, phase=phase, data_path_spec=data_path, cache_root=cache_root)
    return value


def validate_endpoint_permit(
    path: Path,
    *,
    scale: str,
    phase2_binding: dict[str, Any],
    current_bundle: dict[str, Any],
) -> dict[str, Any]:
    permit = read_json(path)
    require(permit.get("schema_version") == "apertus_hard_h_to_g_checkpoint_permit_v2", f"{scale}: checkpoint permit schema drift")
    require(
        permit.get("status") == "passed"
        and permit.get("scale") == scale
        and permit.get("source_phase") == 2
        and permit.get("update") == 3218,
        f"{scale}: endpoint permit identity drift",
    )
    require(permit.get("source_phase_cache_receipt") == phase2_binding, f"{scale}: Phase-2 cache binding drift")
    bundle = permit.get("executing_code_bundle")
    require(
        isinstance(bundle, dict)
        and bundle.get("root") == current_bundle["root"]
        and bundle.get("tree_sha256") == current_bundle["tree_sha256"],
        f"{scale}: checkpoint permit code-bundle drift",
    )
    audit_binding = permit.get("checkpoint_audit")
    require(isinstance(audit_binding, dict), f"{scale}: checkpoint audit binding missing")
    audit_path = Path(str(audit_binding.get("path", "")))
    require(audit_path.is_file() and audit_binding == file_binding(audit_path), f"{scale}: checkpoint audit binding drift")
    audit = read_json(audit_path)
    require(
        audit.get("schema_version") == "apertus_hard_h_to_g_checkpoint_state_audit_v1"
        and audit.get("status") == "passed"
        and audit.get("scale") == scale
        and audit.get("source_phase") == 2
        and audit.get("update") == 3218,
        f"{scale}: checkpoint audit identity drift",
    )
    require(audit.get("source_phase_cache_receipt") == phase2_binding, f"{scale}: audit cache binding drift")
    require(
        audit.get("data_cursor") == {
            "global_consumed_samples": 3_295_232,
            "phase_local_consumed_samples": 979_968,
            "phase_start_update": 2261,
        },
        f"{scale}: endpoint data cursor drift",
    )
    return permit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase2-cache-receipt", type=Path, required=True)
    parser.add_argument("--realized-ledger-receipt", type=Path, required=True)
    parser.add_argument("--8b-checkpoint-permit", type=Path, required=True)
    parser.add_argument("--1p5b-checkpoint-permit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable cross-scale ledger authority exists: {args.output}")
    current_bundle = executing_code_bundle()
    validate_cache(args.phase1_cache_receipt, 1)
    validate_cache(args.phase2_cache_receipt, 2)
    cache_bindings = {
        1: file_binding(args.phase1_cache_receipt),
        2: file_binding(args.phase2_cache_receipt),
    }
    ledger = read_json(args.realized_ledger_receipt)
    require(ledger.get("schema_version") == "apertus_hard_h_to_g_realized_document_ledger_v1", "realized-ledger schema drift")
    require(ledger.get("status") == "passed", "realized-ledger did not pass")
    rows = ledger.get("cache_receipts")
    require(isinstance(rows, list) and len(rows) == 2, "realized-ledger cache set drift")
    observed = {
        int(row["phase"]): (row.get("receipt"), int(row.get("consumed_samples", -1)))
        for row in rows if isinstance(row, dict)
    }
    require(
        observed == {
            phase: (cache_bindings[phase], EXPECTED_CACHE_SAMPLES[phase])
            for phase in (1, 2)
        },
        "realized-ledger Phase-1/2 cache/sample binding drift",
    )
    trajectory_sha = str(ledger.get("realized_sample_trajectory_sha256", ""))
    require(len(trajectory_sha) == 64, "realized sample trajectory SHA-256 missing")
    ledger_bundle = ledger.get("executing_code_bundle")
    require(
        isinstance(ledger_bundle, dict)
        and ledger_bundle.get("root") == current_bundle["root"]
        and ledger_bundle.get("tree_sha256") == current_bundle["tree_sha256"],
        "realized-ledger code-bundle drift",
    )
    phase2_binding = cache_bindings[2]
    validate_endpoint_permit(
        args.__dict__["8b_checkpoint_permit"], scale="8b",
        phase2_binding=phase2_binding, current_bundle=current_bundle,
    )
    validate_endpoint_permit(
        args.__dict__["1p5b_checkpoint_permit"], scale="1p5b",
        phase2_binding=phase2_binding, current_bundle=current_bundle,
    )
    payload = {
        "schema_version": "apertus_hard_h_to_g_cross_scale_ledger_match_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": current_bundle,
        "phase_cache_receipts": {str(phase): binding for phase, binding in cache_bindings.items()},
        "realized_ledger_receipt": file_binding(args.realized_ledger_receipt),
        "realized_sample_trajectory_sha256": trajectory_sha,
        "endpoint_checkpoint_permits": {
            "8b": file_binding(args.__dict__["8b_checkpoint_permit"]),
            "1p5b": file_binding(args.__dict__["1p5b_checkpoint_permit"]),
        },
        "matched_through_update": 3218,
        "global_consumed_samples": 3_295_232,
        "phase_2_local_consumed_samples": 979_968,
        "invariants": {
            "both_scales_bind_the_same_phase_1_and_phase_2_caches": True,
            "both_scales_bind_the_same_exact_index_derived_trajectory": True,
            "both_checkpoint_audits_restore_the_same_phase_local_cursor": True,
            "quota_inference_used": False,
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

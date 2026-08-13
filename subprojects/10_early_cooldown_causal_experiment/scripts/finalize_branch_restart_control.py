#!/usr/bin/env python3
"""Gate paired peak-LR and cooldown probes run on the same allocation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

from contract_utils import atomic_json, file_binding, parse_iteration, read_json, require, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--intervention-log", type=Path, required=True)
    parser.add_argument("--intervention-checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    gate = contract["paired_same_allocation_gate"]
    iteration = int(gate["comparison_iteration"])
    reference = parse_iteration(args.reference_log, iteration)
    intervention = parse_iteration(args.intervention_log, iteration)
    exact = {
        name: reference.get(name) == intervention.get(name)
        for name in gate["exact_fields"]
    }
    checkpoint = args.intervention_checkpoint_root / f"iter_{iteration:07d}"
    source_receipt = read_json(args.source_checkpoint_receipt)
    expected_source_receipt = contract["parent"]["checkpoint_receipt"]
    require(args.source_checkpoint_receipt.resolve() == Path(expected_source_receipt["path"]).resolve(), "source checkpoint receipt path drift")
    require(sha256_file(args.source_checkpoint_receipt) == expected_source_receipt["sha256"], "source checkpoint receipt hash drift")
    source_files = source_receipt.get("source_files", [])
    checks = {
        "all_pre_step_fields_match_exactly": all(exact.values()),
        "reference_lr_is_parent_peak": reference.get("learning rate") == gate["reference_learning_rate"],
        "intervention_lr_is_positive_and_below_peak": (
            0 < intervention.get("learning rate", math.inf)
            < contract["training"]["learning_rate"]["peak"]
        ),
        "learning_rate_is_the_only_logged_difference": all(exact.values()) and reference.get("learning rate") != intervention.get("learning rate"),
        "intervention_checkpoint_is_complete": (checkpoint / ".metadata").is_file() and (checkpoint / "common.pt").is_file(),
        "source_receipt_covers_complete_checkpoint": (
            source_receipt.get("schema_version") == "megatron_exact_checkpoint_view_v1"
            and source_receipt.get("iteration") == gate["source_iteration"]
            and len(source_files) == 131
            and any(row.get("relative_path") == ".metadata" for row in source_files)
            and any(row.get("relative_path") == "common.pt" for row in source_files)
        ),
        "no_skipped_updates": reference.get("number of skipped iterations") == intervention.get("number of skipped iterations") == 0,
        "no_nan_updates": reference.get("number of nan iterations") == intervention.get("number of nan iterations") == 0,
    }
    receipt = {
        "schema_version": "apertus_full8_early_cooldown_paired_restart_control_v2",
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_iteration": int(gate["source_iteration"]),
        "comparison_iteration": iteration,
        "same_allocation_required": True,
        "checks": checks,
        "exact_fields": exact,
        "reference": reference,
        "intervention": intervention,
        "reference_log": file_binding(args.reference_log),
        "intervention_log": file_binding(args.intervention_log),
        "intervention_checkpoint_metadata": file_binding(checkpoint / ".metadata"),
        "source_checkpoint_receipt": file_binding(args.source_checkpoint_receipt),
        "code_bundle_receipt": file_binding(args.code_bundle_receipt),
    }
    atomic_json(args.output, receipt)
    require(receipt["status"] == "passed", f"paired same-allocation gate failed: {checks}")
    print(json.dumps({"ok": True, "iteration": iteration}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

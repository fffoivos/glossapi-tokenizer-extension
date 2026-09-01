#!/usr/bin/env python3
"""Gate a peak-LR/cooldown/peak-LR sandwich on the same allocation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from decimal import Decimal
from pathlib import Path

from contract_utils import atomic_json, file_binding, parse_iteration, read_json, require, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reference-before-log", type=Path, required=True)
    parser.add_argument("--intervention-log", type=Path, required=True)
    parser.add_argument("--reference-after-log", type=Path, required=True)
    parser.add_argument("--intervention-checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    gate = contract["sandwich_same_allocation_gate"]
    iteration = int(gate["comparison_iteration"])
    reference_before = parse_iteration(args.reference_before_log, iteration)
    intervention = parse_iteration(args.intervention_log, iteration)
    reference_after = parse_iteration(args.reference_after_log, iteration)
    exact = {
        name: reference_before.get(name) == intervention.get(name) == reference_after.get(name)
        for name in gate["exact_fields"]
    }
    gradient_field = gate["reproducibility_field"]
    before_gradient = Decimal(str(reference_before.get(gradient_field)))
    intervention_gradient = Decimal(str(intervention.get(gradient_field)))
    after_gradient = Decimal(str(reference_after.get(gradient_field)))
    gradient_low = min(before_gradient, after_gradient)
    gradient_high = max(before_gradient, after_gradient)
    gradient_spread = gradient_high - gradient_low
    maximum_spread = Decimal(str(gate["maximum_parent_control_display_spread"]))
    checkpoint = args.intervention_checkpoint_root / f"iter_{iteration:07d}"
    source_receipt = read_json(args.source_checkpoint_receipt)
    expected_source_receipt = contract["parent"]["checkpoint_receipt"]
    require(args.source_checkpoint_receipt.resolve() == Path(expected_source_receipt["path"]).resolve(), "source checkpoint receipt path drift")
    require(sha256_file(args.source_checkpoint_receipt) == expected_source_receipt["sha256"], "source checkpoint receipt hash drift")
    source_files = source_receipt.get("source_files", [])
    checks = {
        "all_non_gradient_pre_step_fields_match_exactly": all(exact.values()),
        "both_control_lrs_are_parent_peak": (
            reference_before.get("learning rate")
            == reference_after.get("learning rate")
            == gate["control_learning_rate"]
        ),
        "intervention_lr_is_positive_and_below_peak": (
            0 < intervention.get("learning rate", math.inf)
            < contract["training"]["learning_rate"]["peak"]
        ),
        "parent_control_gradient_spread_is_at_most_one_display_quantum": gradient_spread <= maximum_spread,
        "intervention_gradient_is_inside_parent_control_envelope": gradient_low <= intervention_gradient <= gradient_high,
        "learning_rate_is_the_only_scheduled_difference": (
            all(exact.values())
            and reference_before.get("learning rate") != intervention.get("learning rate")
            and reference_after.get("learning rate") != intervention.get("learning rate")
        ),
        "intervention_checkpoint_is_complete": (checkpoint / ".metadata").is_file() and (checkpoint / "common.pt").is_file(),
        "source_receipt_covers_complete_checkpoint": (
            source_receipt.get("schema_version") == "megatron_exact_checkpoint_view_v1"
            and source_receipt.get("iteration") == gate["source_iteration"]
            and len(source_files) == 131
            and any(row.get("relative_path") == ".metadata" for row in source_files)
            and any(row.get("relative_path") == "common.pt" for row in source_files)
        ),
        "no_skipped_updates": reference_before.get("number of skipped iterations") == intervention.get("number of skipped iterations") == reference_after.get("number of skipped iterations") == 0,
        "no_nan_updates": reference_before.get("number of nan iterations") == intervention.get("number of nan iterations") == reference_after.get("number of nan iterations") == 0,
    }
    receipt = {
        "schema_version": "apertus_full8_early_cooldown_sandwich_restart_control_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_iteration": int(gate["source_iteration"]),
        "comparison_iteration": iteration,
        "same_allocation_required": True,
        "checks": checks,
        "exact_fields": exact,
        "reference_before": reference_before,
        "intervention": intervention,
        "reference_after": reference_after,
        "gradient_reproducibility": {
            "field": gradient_field,
            "display_precision_decimals": gate["gradient_display_precision_decimals"],
            "before": float(before_gradient),
            "intervention": float(intervention_gradient),
            "after": float(after_gradient),
            "parent_control_low": float(gradient_low),
            "parent_control_high": float(gradient_high),
            "parent_control_spread": float(gradient_spread),
            "maximum_parent_control_spread": float(maximum_spread),
        },
        "reference_before_log": file_binding(args.reference_before_log),
        "intervention_log": file_binding(args.intervention_log),
        "reference_after_log": file_binding(args.reference_after_log),
        "intervention_checkpoint_metadata": file_binding(checkpoint / ".metadata"),
        "source_checkpoint_receipt": file_binding(args.source_checkpoint_receipt),
        "code_bundle_receipt": file_binding(args.code_bundle_receipt),
    }
    atomic_json(args.output, receipt)
    require(receipt["status"] == "passed", f"sandwich same-allocation gate failed: {checks}")
    print(json.dumps({"ok": True, "iteration": iteration}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

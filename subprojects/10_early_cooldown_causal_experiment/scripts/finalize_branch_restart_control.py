#!/usr/bin/env python3
"""Gate the early-cooldown first update against the verified replay checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

from contract_utils import atomic_json, file_binding, read_json, require
from finalize_one_update_control import parse_iteration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--parent-replay-receipt", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    gate = contract["branch_restart_gate"]
    replay = read_json(args.parent_replay_receipt)
    require(replay.get("schema_version") == "apertus_full8_early_cooldown_parent_replay_v1" and replay.get("status") == "passed" and all(replay.get("checks", {}).values()), "parent replay is not passed")
    iteration = int(gate["iteration"])
    observed = parse_iteration(args.control_log, iteration)
    expected = replay["rows"][str(iteration)]["observed"]
    exact = {name: observed.get(name) == expected[name] for name in gate["exact_fields"]}
    tolerance = gate["gradient_tolerance"]
    grad, expected_grad = observed.get("grad norm", math.inf), expected["grad norm"]
    delta = abs(grad - expected_grad)
    limit = float(tolerance["atol"]) + float(tolerance["rtol"]) * abs(expected_grad)
    checks = {
        "all_non_lr_logged_fields_match_replay": all(exact.values()),
        "gradient_norm_within_predeclared_tolerance": math.isfinite(grad) and delta <= limit,
        "learning_rate_below_parent_peak": 0 < observed.get("learning rate", math.inf) < contract["training"]["learning_rate"]["peak"],
        "no_skipped_updates": observed.get("number of skipped iterations") == 0,
        "no_nan_updates": observed.get("number of nan iterations") == 0,
    }
    receipt = {
        "schema_version": "apertus_full8_early_cooldown_branch_restart_control_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": iteration,
        "checks": checks,
        "exact_fields": exact,
        "expected_from_replay": expected,
        "observed": observed,
        "gradient_norm": {"absolute_delta": delta, "absolute_limit": limit, **tolerance},
        "control_log": file_binding(args.control_log),
        "parent_replay_receipt": file_binding(args.parent_replay_receipt),
        "code_bundle_receipt": file_binding(args.code_bundle_receipt),
    }
    atomic_json(args.output, receipt)
    require(receipt["status"] == "passed", f"branch restart control failed: {checks}")
    print(json.dumps({"ok": True, "iteration": iteration}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

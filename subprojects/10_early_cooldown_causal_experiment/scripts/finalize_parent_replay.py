#!/usr/bin/env python3
"""Verify that the 8,000->9,537 replay reproduces the frozen parent rows."""

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
    parser.add_argument("--replay-log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = read_json(args.contract)["parent_replay_gate"]
    tolerance = gate["gradient_tolerance"]
    rows, checks = {}, {}
    for iteration in gate["comparison_iterations"]:
        key = str(iteration)
        observed = parse_iteration(args.replay_log, iteration)
        expected = {name: float(value) for name, value in gate["expected"][key].items()}
        exact = {name: observed.get(name) == expected[name] for name in gate["exact_fields"]}
        grad, expected_grad = observed.get("grad norm", math.inf), expected["grad norm"]
        delta = abs(grad - expected_grad)
        limit = float(tolerance["atol"]) + float(tolerance["rtol"]) * abs(expected_grad)
        grad_ok = math.isfinite(grad) and delta <= limit
        checks[f"iteration_{iteration}_exact_fields"] = all(exact.values())
        checks[f"iteration_{iteration}_gradient_within_tolerance"] = grad_ok
        rows[key] = {
            "expected": expected,
            "observed": observed,
            "exact_fields": exact,
            "gradient_norm": {"absolute_delta": delta, "absolute_limit": limit, **tolerance},
        }
    checkpoint = args.checkpoint_root / "iter_0009536"
    checks["synchronous_iteration_9536_checkpoint_complete"] = (
        checkpoint.is_dir() and (checkpoint / ".metadata").is_file()
    )
    receipt = {
        "schema_version": "apertus_full8_early_cooldown_parent_replay_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_iteration": int(gate["source_iteration"]),
        "saved_iteration": int(gate["save_iteration"]),
        "last_control_iteration": max(gate["comparison_iterations"]),
        "checks": checks,
        "rows": rows,
        "replay_log": file_binding(args.replay_log),
        "checkpoint_metadata": file_binding(checkpoint / ".metadata") if (checkpoint / ".metadata").is_file() else None,
        "code_bundle_receipt": file_binding(args.code_bundle_receipt),
    }
    atomic_json(args.output, receipt)
    require(receipt["status"] == "passed", f"parent replay gate failed: {checks}")
    print(json.dumps({"ok": True, "saved_iteration": gate["save_iteration"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare the same-allocation update-9537 control with the frozen parent log."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

from contract_utils import atomic_json, file_binding, read_json, require


ITERATION = re.compile(r"iteration\s+(\d+)\s*/")
METRIC = re.compile(r"(?:^|\|)\s*([^|:]+?)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def parse_iteration(path: Path, iteration: int) -> dict[str, float]:
    matches = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        found = ITERATION.search(line)
        if found and int(found.group(1)) == iteration:
            matches.append({normalize(key): float(value) for key, value in METRIC.findall(line)})
    require(len(matches) == 1, f"expected exactly one logged iteration {iteration}, got {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    control = contract["one_update_control"]
    observed = parse_iteration(args.control_log, int(control["iteration"]))
    expected = {key: float(value) for key, value in control["expected"].items()}
    exact = {key: observed.get(key) == expected[key] for key in control["exact_fields"]}
    grad = observed.get("grad norm", math.inf)
    expected_grad = expected["grad norm"]
    tolerance = control["gradient_tolerance"]
    delta = abs(grad - expected_grad)
    limit = float(tolerance["atol"]) + float(tolerance["rtol"]) * abs(expected_grad)
    gradient_ok = math.isfinite(grad) and delta <= limit
    checks = {
        "all_exact_logged_fields": all(exact.values()),
        "gradient_norm_within_predeclared_tolerance": gradient_ok,
        "no_skipped_updates": observed.get("number of skipped iterations") == 0,
        "no_nan_updates": observed.get("number of nan iterations") == 0,
    }
    receipt = {
        "schema_version": "apertus_full8_early_cooldown_one_update_control_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": int(control["iteration"]),
        "checks": checks,
        "exact_fields": exact,
        "expected": expected,
        "observed": observed,
        "gradient_norm": {"absolute_delta": delta, "absolute_limit": limit, **tolerance},
        "control_log": file_binding(args.control_log),
        "code_bundle_receipt": file_binding(args.code_bundle_receipt),
    }
    atomic_json(args.output, receipt)
    require(receipt["status"] == "passed", f"one-update causal control failed: {checks}")
    print(json.dumps({"ok": True, "iteration": control["iteration"], "gradient_delta": delta}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

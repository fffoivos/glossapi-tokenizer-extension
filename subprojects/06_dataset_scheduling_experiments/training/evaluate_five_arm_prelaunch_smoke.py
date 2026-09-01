#!/usr/bin/env python3
"""Verify real-data five-arm initial-load and resume smoke outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.collect_validation_trajectory import parse_log


ARMS = ("D0_mixed", "D1_hard_h_to_g", "D2_hard_g_to_h", "D3_gradual_h_to_g", "D4_gradual_g_to_h")
METRIC = re.compile(r"(?:^|\|)\s*([^|:]+?)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation_manifest.read_text())
    panels = {row["name"] for row in validation["panels"]}
    if len(panels) != 13:
        raise ValueError("expected 13 validation panels")
    results = []
    all_checks = []
    for phase, endpoint in (("initial_to_64", 64), ("resume_64_to_128", 128)):
        for arm in ARMS:
            root = args.run_root / phase / arm
            log = root / "driver.log"
            state = json.loads((root / "process_state.json").read_text())
            losses, grads = [], []
            text = log.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                for key, raw in METRIC.findall(line):
                    key = " ".join(key.lower().split())
                    value = float(raw)
                    if key.endswith("lm loss"): losses.append(value)
                    elif key.endswith("grad norm"): grads.append(value)
            endpoint_rows = [row for row in parse_log(log) if int(row["iteration"]) == endpoint]
            observed_panels = {row["panel"] for row in endpoint_rows}
            checks = {
                "process_exit_zero": int(state.get("exit_code", -1)) == 0,
                "endpoint_checkpoint_complete": (root / "checkpoints" / f"iter_{endpoint:07d}" / ".metadata").is_file(),
                "at_least_50_finite_losses": len(losses) >= 50 and all(math.isfinite(value) for value in losses),
                "finite_gradient_norms": bool(grads) and all(math.isfinite(value) for value in grads),
                "all_validation_panels_at_endpoint": observed_panels == panels,
                "resume_log_binds_iteration_64": phase == "initial_to_64" or bool(re.search(r"(?:load|checkpoint)[^\n]{0,200}(?:iteration\s*)?64", text, re.I)),
            }
            all_checks.extend(checks.values())
            results.append({"phase": phase, "arm_id": arm, "endpoint": endpoint, "checks": checks, "loss_count": len(losses), "validation_panels": sorted(observed_panels), "driver_log": {"path": str(log.resolve()), "sha256": sha(log)}})
    payload = {
        "schema_version": "apertus_mini_five_arm_prelaunch_smoke_v1",
        "status": "passed" if all(all_checks) else "failed",
        "real_scheduled_data": True,
        "data_parallel_size_per_arm": 16,
        "concurrent_arms": 5,
        "initial_load_endpoint": 64,
        "resume_endpoint": 128,
        "results": results,
    }
    partial = Path(str(args.output) + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, args.output)
    print(json.dumps({"ok": all(all_checks), "checks": len(all_checks)}, sort_keys=True))
    return 0 if all(all_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail closed on incomplete, skipped, nonfinite, or under-evaluated training."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from contract import atomic_write_json, file_binding, read_json


ITERATION = re.compile(r"iteration\s+(\d+)\s*/", re.IGNORECASE)
METRIC = re.compile(r"(?:^|\|)\s*([^|:]+?)\s*:\s*([^|]+)")
VALIDATION = re.compile(r"validation loss at .*?\[([^\]]+)\]", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    iterations = []
    skipped = 0
    nonfinite = []
    for line in text.splitlines():
        match = ITERATION.search(line)
        if not match:
            continue
        iteration = int(match.group(1))
        if not args.start < iteration <= args.end:
            continue
        iterations.append(iteration)
        for raw_key, raw_value in METRIC.findall(line):
            key = " ".join(raw_key.lower().split())
            value = raw_value.strip().split()[0]
            if key == "number of skipped iterations":
                skipped = max(skipped, int(float(value)))
            if key in {"lm loss", "grad norm", "params norm"}:
                try:
                    if not math.isfinite(float(value)):
                        nonfinite.append({"iteration": iteration, "metric": key, "value": value})
                except ValueError:
                    nonfinite.append({"iteration": iteration, "metric": key, "value": value})
    if not iterations or min(iterations) != args.start + 1 or max(iterations) != args.end:
        raise ValueError(f"training iteration coverage drift: {min(iterations or [-1])}..{max(iterations or [-1])}")
    if skipped or nonfinite:
        raise ValueError(f"scientific failure: skipped={skipped} nonfinite={nonfinite[:3]}")
    manifest = read_json(args.validation_manifest)
    panels = [row["name"] for row in manifest.get("panels", [])]
    if len(panels) != 13 or len(set(panels)) != 13:
        raise ValueError("validation manifest panel drift")
    observed = {name: 0 for name in panels}
    for name in VALIDATION.findall(text):
        if name in observed:
            observed[name] += 1
    expected_regular = args.end // 25 - args.start // 25
    if any(count < expected_regular for count in observed.values()):
        raise ValueError(f"source validation cadence drift: expected at least {expected_regular}, observed {observed}")
    payload = {
        "schema_version": "apertus_full_8b_training_attempt_audit_v1",
        "status": "passed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "start_iteration": args.start,
        "end_iteration": args.end,
        "optimizer_updates": len(set(iterations)),
        "skipped_updates": skipped,
        "nonfinite_metrics": nonfinite,
        "minimum_expected_regular_validation_points_per_panel": expected_regular,
        "validation_points_by_panel": observed,
        "training_log": file_binding(args.log),
        "validation_manifest": file_binding(args.validation_manifest),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "updates": len(set(iterations)), "validation_min": min(observed.values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

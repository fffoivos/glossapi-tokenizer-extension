#!/usr/bin/env python3
"""Freeze the bounded one-successor allocation schedule from exact recipe boundaries."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import file_binding, read_json, require, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--allocation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable prequeue schedule exists: {args.output}")
    recipe = read_json(args.recipe)
    allocation = read_json(args.allocation_plan)
    boundaries = [int(value) for value in recipe["segments"]["boundaries"]]
    require(boundaries[0] == 0 and len(boundaries) == 3, "targeted recipe must contain two absolute segments")
    targets = []
    if args.experiment == "A":
        policy = allocation["experiment_a"]
        require(int(policy["conservative_segment_runtime_seconds"]) == 36_000, "source runtime budget drift")
        require(int(policy["maximum_hold_seconds"]) == 6_000, "maximum hold drift")
        require(int(policy["source_trigger_seconds"]) == 30_000, "source trigger drift")
        targets.append({
            "target_segment_id": 1,
            "source_minimum_train_seconds": 36_000,
            "source_trigger_minutes": 500,
            "minimum_train_seconds": 36_000,
            "maximum_hold_seconds": 6_000,
        })
    else:
        require(boundaries[1] == 9_536, "B parent boundary drift")
    payload = {
        "schema_version": "apertus_full_8b_prequeue_schedule_v1",
        "status": "approved",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": "measured_safe_idle_overlap_v1",
        "experiment": args.experiment,
        "recipe": file_binding(args.recipe),
        "allocation_plan": file_binding(args.allocation_plan),
        "allocation_seconds": 43_200,
        "allocation_reserve_seconds": 1_200,
        "measured_seconds_per_update": 11.05,
        "segment_boundaries": boundaries,
        "targets": targets,
        "invariants": {
            "at_most_one_delayed_successor": True,
            "debug_timer_job_used": False,
            "source_trigger_uses_conservative_wall_time_not_step_time": True,
            "holder_requires_signed_checkpoint_permit": True,
            "holder_requires_target_runtime_plus_reserve": True,
        },
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": True, "experiment": args.experiment, "targets": len(targets)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

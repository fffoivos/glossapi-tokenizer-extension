#!/usr/bin/env python3
"""Promote one receipt-bound execution profile by a predeclared efficiency rule."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from build_training_run_permit import PROFILE_CHECKS


def validate_candidate(
    path: Path,
    *,
    scale: str,
    allowed: dict[str, dict[str, Any]],
    current: dict[str, Any],
    minimum_updates: int,
    discard_updates: int,
) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema_version") == "apertus_hard_h_to_g_profile_benchmark_v1", f"profile benchmark schema drift: {path}")
    require(value.get("status") in {"passed", "rejected"} and value.get("scale") == scale, f"profile benchmark identity drift: {path}")
    profile = value.get("profile")
    require(isinstance(profile, dict), f"profile benchmark geometry missing: {path}")
    profile_id = str(profile.get("profile_id", ""))
    require(profile_id in allowed, f"candidate outside frozen grid: {profile_id}")
    frozen = allowed[profile_id]
    for key in ("nodes", "gpus_per_node", "tensor_parallel", "pipeline_parallel", "context_parallel", "data_parallel"):
        require(int(profile.get(key, -1)) == int(frozen[key]), f"{profile_id}: {key} drift")
    microbatch = int(profile.get("microbatch", 0))
    require(microbatch > 0 and 1024 % (int(profile["data_parallel"]) * microbatch) == 0, f"{profile_id}: invalid microbatch/global batch")
    measurement = value.get("measurement")
    require(isinstance(measurement, dict), f"{profile_id}: measurement missing")
    require(int(measurement.get("updates", -1)) >= minimum_updates, f"{profile_id}: benchmark too short")
    require(int(measurement.get("discarded_warmup_updates", -1)) == discard_updates, f"{profile_id}: warmup discard drift")
    for key in ("median_step_seconds", "p90_step_seconds", "tokens_per_gpu_hour"):
        require(float(measurement.get(key, 0)) > 0, f"{profile_id}: invalid {key}")
    checks = value.get("checks")
    require(isinstance(checks, dict) and set(checks) == set(PROFILE_CHECKS), f"{profile_id}: parity check set drift")
    require(all(isinstance(checks[name], bool) for name in PROFILE_CHECKS), f"{profile_id}: parity checks are not booleans")
    require((value["status"] == "passed") == all(checks[name] is True for name in PROFILE_CHECKS), f"{profile_id}: status/check result mismatch")
    evidence = value.get("evidence")
    require(isinstance(evidence, list) and evidence, f"{profile_id}: evidence missing")
    for row in evidence:
        evidence_path = Path(str(row.get("path", "")))
        require(evidence_path.is_file() and row == file_binding(evidence_path), f"{profile_id}: evidence binding drift")
    bundle = value.get("executing_code_bundle")
    require(isinstance(bundle, dict) and bundle.get("root") == current["root"] and bundle.get("tree_sha256") == current["tree_sha256"], f"{profile_id}: code-bundle drift")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable profile-promotion receipt exists: {args.output}")
    experiment = read_json(args.experiment)
    allocation = read_json(args.allocation)
    require(experiment.get("schema_version") == "apertus_hard_h_to_g_replication_v2", "experiment contract drift")
    require(allocation.get("schema_version") == "apertus_hard_h_to_g_allocation_v1", "allocation contract drift")
    selection_rule = experiment["profile_selection"]
    current = executing_code_bundle()
    if args.scale == "8b":
        rows = [allocation["profiles"]["8b"]]
    else:
        rows = allocation["profiles"]["1p5b_candidates"]
    allowed = {str(row["profile_id"]): row for row in rows}
    candidates = [
        validate_candidate(
            path, scale=args.scale, allowed=allowed, current=current,
            minimum_updates=int(selection_rule["minimum_updates"]),
            discard_updates=int(selection_rule["discard_warmup_updates"]),
        )
        for path in args.candidate
    ]
    require(len({row["profile"]["profile_id"] for row in candidates}) == len(candidates), "duplicate profile candidate")
    passing = [row for row in candidates if row["status"] == "passed"]
    if args.scale == "8b":
        require(len(candidates) == 1 and candidates[0]["profile"]["profile_id"] == "dp32_16node" and len(passing) == 1, "8B promotion requires the passing exact DP32 candidate")
    else:
        require(len(candidates) >= int(selection_rule["minimum_candidates"]), "too few 1.5B profile candidates")
        require(len(passing) >= int(selection_rule["minimum_passing_candidates"]), "too few passing 1.5B profile candidates")
    best_efficiency = max(float(row["measurement"]["tokens_per_gpu_hour"]) for row in passing)
    near = [
        row for row in passing
        if float(row["measurement"]["tokens_per_gpu_hour"])
        >= best_efficiency * (1 - float(selection_rule["near_tie_relative_fraction"]))
    ]
    selected = min(
        near,
        key=lambda row: (
            float(row["measurement"]["p90_step_seconds"]),
            int(row["profile"]["nodes"]),
            str(row["profile"]["profile_id"]),
        ),
    )
    selection = dict(selected["profile"])
    selection["global_batch_sequences"] = 1024
    payload = {
        "schema_version": "apertus_hard_h_to_g_profile_promotion_v1",
        "status": "promoted",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scale": args.scale,
        "executing_code_bundle": current,
        "experiment": file_binding(args.experiment),
        "allocation": file_binding(args.allocation),
        "candidate_receipts": [file_binding(path) for path in args.candidate],
        "candidate_metrics": {
            row["profile"]["profile_id"]: {
                "status": row["status"], "checks": row["checks"], **row["measurement"]
            } for row in candidates
        },
        "selection_rule": selection_rule,
        "selection": selection,
        "checks": {name: True for name in PROFILE_CHECKS},
        "evidence": [file_binding(path) for path in args.candidate],
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

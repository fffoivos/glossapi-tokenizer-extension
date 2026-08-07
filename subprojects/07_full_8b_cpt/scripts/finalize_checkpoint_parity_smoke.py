#!/usr/bin/env python3
"""Freeze a two-allocation DP32 synchronous-checkpoint restart proof."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json
from finalize_parallelism_benchmark import parse_log, restart_equivalence, restart_provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--control-repeat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profiles = read_json(args.profiles)
    thresholds = profiles["benchmark"]["promotion"]
    control_log = args.control_root / "segments/updates_0_162/training.log"
    restart_log = args.control_root / "segments/updates_160_161/training.log"
    repeat_log = args.control_repeat_root / "segments/updates_160_161/training.log"
    control = parse_log(control_log, require_full=False)
    restart = parse_log(restart_log, require_full=False)
    repeat = parse_log(repeat_log, require_full=False)
    control_job = read_json(
        args.control_root / "segments/updates_0_162/training_job_receipt.json"
    )
    scientific_digest = str(control_job.get("scientific_digest", ""))
    if not scientific_digest:
        raise ValueError("control scientific digest absent")
    restart_options = {
        "gradient_atol": thresholds["restart_gradient_norm_atol"],
        "gradient_rtol": thresholds["restart_gradient_norm_rtol"],
    }
    provenance = restart_provenance(
        args.control_root,
        profile_id="dp32_16node",
        scientific_digest=scientific_digest,
        iteration=160,
    )
    repeat_provenance = restart_provenance(
        args.control_repeat_root,
        profile_id="dp32_16node",
        scientific_digest=scientific_digest,
        iteration=160,
        checkpoint_root=args.control_root / "checkpoints",
    )
    numerical = restart_equivalence(control, restart, 161, **restart_options)
    repeat_numerical = restart_equivalence(control, repeat, 161, **restart_options)
    first_restart_row = restart["rows"].get(161)
    second_restart_row = repeat["rows"].get(161)
    repeat_identity = {
        "passed": first_restart_row is not None and first_restart_row == second_restart_row,
        "iteration": 161,
        "first": first_restart_row,
        "second": second_restart_row,
        "requirement": "exact equality of all logged loss, gradient-norm and parameter-norm fields",
    }
    checks = {
        "control_completed_to_162": (
            control_job.get("schema_version") == "apertus_full_8b_training_job_v1"
            and control_job.get("status") == "completed"
            and control_job.get("profile_id") == "dp32_16node"
            and control_job.get("scientific_digest") == scientific_digest
            and control_job.get("start_iteration") == 0
            and control_job.get("end_iteration") == 162
        ),
        "synchronous_checkpoint_save": control_job.get("checkpoint_save_mode") == "synchronous",
        "control_zero_skipped_updates": control["skipped"] == 0,
        "control_zero_nonfinite_updates": control["nonfinite"] == 0,
        "first_restart_provenance": provenance["passed"],
        "second_restart_provenance": repeat_provenance["passed"],
        "first_restart_numerically_equivalent": numerical["passed"],
        "second_restart_numerically_equivalent": repeat_numerical["passed"],
        "independent_restarts_identical": repeat_identity["passed"],
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "apertus_full_8b_checkpoint_parity_smoke_v1",
        "status": "passed" if passed else "failed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile_id": "dp32_16node",
        "checkpoint_iteration": 160,
        "comparison_iteration": 161,
        "checkpoint_save_mode": "synchronous",
        "checks": checks,
        "restart": {
            "first": {"provenance": provenance, "numerical": numerical},
            "second": {"provenance": repeat_provenance, "numerical": repeat_numerical},
            "independent_identity": repeat_identity,
        },
        "thresholds": restart_options,
        "inputs": {
            "profiles": file_binding(args.profiles),
            "schedule_manifest": file_binding(args.stage_root / "schedules/schedule_manifest.json"),
            "control_log": file_binding(control_log),
            "restart_log": file_binding(restart_log),
            "repeat_log": file_binding(repeat_log),
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": passed, "checks": checks}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

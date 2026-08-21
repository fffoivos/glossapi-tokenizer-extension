#!/usr/bin/env python3
"""Recover canonical gate context after a completed profile finalizer retry.

This is an evidence-only recovery path.  It refuses to run unless the profile
receipt is already passed, re-derives the live Slurm geometry, and reconstructs
the same three receipts normally written by the qualification adapter after
the benchmark subprocess returns.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from contract_utils import file_binding, read_json, require, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--contract-digest", required=True)
    parser.add_argument("--remaining-conservative-blocks", type=int, required=True)
    parser.add_argument("--checkpoint-reserve-seconds", type=int, required=True)
    parser.add_argument("--maximum-allocation-seconds", type=int, required=True)
    parser.add_argument("--recovery-code-root", type=Path, required=True)
    parser.add_argument("--recovery-code-receipt", type=Path, required=True)
    parser.add_argument("--base-code-root", type=Path, required=True)
    parser.add_argument("--base-code-receipt", type=Path, required=True)
    parser.add_argument("--context-recovery-script", type=Path, required=True)
    parser.add_argument("--issue-url", required=True)
    return parser.parse_args()


def scontrol_value(raw: str, key: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S+)", raw)
    return "" if match is None else match.group(1)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    args = parse_args()
    qualification_root = args.qualification_root.resolve()
    manifest = read_json(args.manifest.resolve())
    require(manifest.get("runtime", {}).get("status") == "candidate", "runtime is not a candidate")
    require(
        manifest.get("campaign_id") == args.campaign_id
        and manifest.get("contract_digest") == args.contract_digest,
        "campaign identity drift",
    )
    runtime = manifest["runtime"]
    profile_path = qualification_root / "profile_benchmark.json"
    preflight_path = qualification_root / "benchmark/preflight.json"
    profile = read_json(profile_path)
    preflight = read_json(preflight_path)
    require(profile.get("status") == "passed", "profile receipt is not passed")
    checks = profile.get("checks")
    require(isinstance(checks, dict) and checks and all(checks.values()), "profile checks did not all pass")
    require(preflight.get("status") == "passed", "benchmark preflight is not passed")
    require(
        preflight.get("executing_code_bundle", {}).get("root")
        == str(args.base_code_root.resolve()),
        "benchmark base code root drift",
    )

    measurement = profile.get("measurement")
    require(isinstance(measurement, dict), "profile measurement is missing")
    production_cadence_wall_seconds = int(measurement.get("production_cadence_wall_seconds", 0))
    require(production_cadence_wall_seconds > 0, "invalid production cadence")
    qualification_elapsed_seconds = (
        parse_time(str(profile["created_at"])) - parse_time(str(preflight["created_at"]))
    ).total_seconds()
    require(qualification_elapsed_seconds > 0, "invalid recovered qualification wall time")
    projected_training_seconds = (
        1.15 * args.remaining_conservative_blocks * production_cadence_wall_seconds
    )
    projected_total_seconds = (
        qualification_elapsed_seconds
        + projected_training_seconds
        + args.checkpoint_reserve_seconds
    )
    require(
        projected_total_seconds <= args.maximum_allocation_seconds,
        "candidate profile cannot finish the projected allocation budget",
    )

    job_id = os.environ.get("SLURM_JOB_ID", "")
    raw_job = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    slurm_checks = {
        "job_id_present": bool(re.fullmatch(r"\d+", job_id)),
        "normal_partition": os.environ.get("SLURM_JOB_PARTITION") == "normal",
        "exact_live_nodes": int(os.environ.get("SLURM_NNODES", "0"))
        == int(runtime["slurm"]["nodes"]),
        "exact_parallelism_geometry": int(runtime["parallelism"]["tensor"])
        * int(runtime["parallelism"]["pipeline"])
        * int(runtime["parallelism"]["context"])
        * int(runtime["parallelism"]["data"])
        == int(runtime["slurm"]["nodes"]) * int(runtime["slurm"]["gpus_per_node"]),
        "scontrol_job_id": scontrol_value(raw_job, "JobId") == job_id,
        "scontrol_partition": scontrol_value(raw_job, "Partition") == "normal",
        "scontrol_nodes": scontrol_value(raw_job, "NumNodes")
        in {str(runtime["slurm"]["nodes"]), f"{runtime['slurm']['nodes']}-{runtime['slurm']['nodes']}"},
    }
    require(all(slurm_checks.values()), f"live Slurm profile drift: {slurm_checks}")

    budget_path = qualification_root / "allocation_budget_projection.json"
    slurm_path = qualification_root / "slurm_profile.json"
    context_path = qualification_root / "context.json"
    recovery_path = qualification_root / "finalizer_recovery.json"
    write_json_atomic(
        budget_path,
        {
            "schema_version": "apertus_hard_h_to_g_first_allocation_budget_v1",
            "status": "passed",
            "campaign_id": args.campaign_id,
            "contract_digest": args.contract_digest,
            "qualification_elapsed_seconds": qualification_elapsed_seconds,
            "qualification_elapsed_source": "profile.created_at - preflight.created_at",
            "candidate_production_cadence_wall_seconds": production_cadence_wall_seconds,
            "remaining_conservative_blocks": args.remaining_conservative_blocks,
            "conservative_multiplier": 1.15,
            "projected_training_seconds": projected_training_seconds,
            "checkpoint_reserve_seconds": args.checkpoint_reserve_seconds,
            "maximum_allocation_seconds": args.maximum_allocation_seconds,
            "projected_total_seconds": projected_total_seconds,
        },
    )
    write_json_atomic(
        slurm_path,
        {
            "schema_version": "apertus_in_allocation_slurm_profile_v1",
            "status": "passed",
            "campaign_id": args.campaign_id,
            "contract_digest": args.contract_digest,
            "job_id": job_id,
            "runtime_profile_id": runtime["profile_id"],
            "checks": slurm_checks,
            "scontrol": raw_job,
        },
    )
    write_json_atomic(
        recovery_path,
        {
            "schema_version": "apertus_completed_profile_context_recovery_v1",
            "status": "passed",
            "campaign_id": args.campaign_id,
            "contract_digest": args.contract_digest,
            "reason": "profile finalizer initially rejected an explicitly compatible cache producer after GPU work completed",
            "issue_url": args.issue_url,
            "base_scientific_code": {
                "root": str(args.base_code_root.resolve()),
                "receipt": file_binding(args.base_code_receipt.resolve()),
            },
            "evidence_only_recovery_code": {
                "root": str(args.recovery_code_root.resolve()),
                "receipt": file_binding(args.recovery_code_receipt.resolve()),
            },
            "context_recovery_script": file_binding(
                args.context_recovery_script.resolve()
            ),
            "profile_measurement": file_binding(profile_path),
            "statement": "No training, model, optimizer, data, tokenizer, schedule, or checkpoint artifact was changed by this recovery.",
        },
    )
    write_json_atomic(
        context_path,
        {
            "schema_version": "apertus_gate_context_v1",
            "status": "passed",
            "campaign_id": args.campaign_id,
            "contract_digest": args.contract_digest,
            "evidence": {
                "profile_measurement": file_binding(profile_path),
                "slurm_profile": file_binding(slurm_path),
                "allocation_budget": file_binding(budget_path),
            },
        },
    )
    print(context_path)
    print(recovery_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

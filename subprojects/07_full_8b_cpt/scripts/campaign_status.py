#!/usr/bin/env python3
"""Report exact 8B campaign progress and a receipt-aware remaining-time range."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def slurm_state(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", ",".join(job_ids), "--format=JobIDRaw,State", "--parsable2"],
        text=True,
        capture_output=True,
        check=False,
    )
    states: dict[str, str] = {}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            job, separator, state = line.partition("|")
            if separator and job in job_ids:
                states[job] = state.partition("|")[0]
    return states


def receipt_count(root: Path, pattern: str) -> int:
    return sum(1 for path in root.glob(pattern) if path.is_file()) if root.exists() else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--eta-config", type=Path, required=True)
    args = parser.parse_args()
    recipe = read_json(args.recipe)
    eta = read_json(args.eta_config)
    graph_path = args.run_root / "submissions" / "launch_graph.json"
    graph = read_json(graph_path) if graph_path.is_file() else {"segments": []}
    segments = graph.get("segments", [])
    job_ids = [str(row[key]) for row in segments for key in ("train_job", "checkpoint_gate_job")]
    states = slurm_state(job_ids)
    completed_update = 0
    segment_rows = []
    for row in segments:
        gate_state = states.get(str(row["checkpoint_gate_job"]), "UNKNOWN")
        train_state = states.get(str(row["train_job"]), "UNKNOWN")
        if gate_state.startswith("COMPLETED"):
            completed_update = max(completed_update, int(row["end"]))
        segment_rows.append({**row, "train_state": train_state, "gate_state": gate_state})
    total = int(recipe["batch_and_parallelism"]["training_updates"])
    remaining = total - completed_update
    wall = eta["training"]["wall_seconds_per_update"]
    checkpoint_evaluations = args.run_root / "checkpoint_evaluations"
    terminal_perdoc = receipt_count(
        checkpoint_evaluations / "iter_0019248" / "per_document", "*.receipt.json"
    )
    result = {
        "schema_version": "apertus_full_8b_campaign_status_v1",
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_root": str(args.run_root),
        "progress": {
            "completed_checkpoint_gated_updates": completed_update,
            "total_updates": total,
            "remaining_updates": remaining,
            "completed_tokens": completed_update * int(recipe["batch_and_parallelism"]["global_batch_tokens"]),
        },
        "remaining_training_core_hours": {
            key: remaining * float(value) / 3600 for key, value in wall.items()
        },
        "evaluation_receipts": {
            "greekmmlu_completed": receipt_count(
                checkpoint_evaluations, "iter_*/exact_checkpoint_native_greekmmlu_receipt.json"
            ),
            "greekmmlu_required": len(recipe["evaluation"]["greekmmlu"]["checkpoint_updates"]) - 1,
            "terminal_per_document_completed": terminal_perdoc,
            "terminal_per_document_required": 13,
        },
        "segments": segment_rows,
        "warning": "remaining core hours exclude future queue, recovery, and unfinished evaluation tail; reforecast after every gate or failure",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

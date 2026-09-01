#!/usr/bin/env python3
"""Freeze pending watcher waves into bounded multi-node debug batches."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from campaign_contract import (
    AUTHORITATIVE_EVALUATION_DTYPE,
    evaluation_namespace,
    read_json,
    scoped_evaluation_root,
    sha256_file,
)


def slurm_state(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-j", job_id, "-X", "-n", "-o", "State"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = [line.strip().split()[0].split("+")[0] for line in result.stdout.splitlines() if line.strip()]
    return rows[0] if rows else "UNKNOWN"


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--segment-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--waves-per-batch", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.waves_per_batch <= 32:
        raise ValueError("waves per batch must be between one and thirty-two")

    namespace = evaluation_namespace()
    state_root = scoped_evaluation_root(args.run_root, "evaluation_watch") / f"segment_{args.segment_id}"
    rows: list[dict[str, Any]] = []
    for state_path in sorted(state_root.glob("iteration_*.json")):
        state = read_json(state_path)
        if state.get("status") != "submitted" or not state.get("attempts"):
            continue
        if state.get("evaluation_namespace") != namespace:
            raise ValueError(f"evaluation namespace drift: {state_path}")
        attempt = state["attempts"][-1]
        job_id = str(attempt["job_id"])
        job_state = slurm_state(job_id)
        if job_state != "PENDING":
            continue
        wave_manifest = Path(attempt["wave_manifest"]).resolve()
        wave_output_root = Path(attempt["wave_output_root"]).resolve()
        wave = read_json(wave_manifest)
        if wave.get("schema_version") != "apertus_mini_greekmmlu_wave_v1" or wave.get("status") != "frozen":
            raise ValueError(f"wave manifest drift: {wave_manifest}")
        rows.append(
            {
                "iteration": int(state["iteration"]),
                "attempt": int(attempt["attempt"]),
                "cancelled_job_id": job_id,
                "state_path": str(state_path.resolve()),
                "state_sha256": sha256_file(state_path),
                "wave_manifest": str(wave_manifest),
                "wave_manifest_sha256": sha256_file(wave_manifest),
                "wave_output_root": str(wave_output_root),
            }
        )
    if not rows:
        raise ValueError("no pending evaluation waves to batch")

    manifest_sha = sha256_file(args.campaign_manifest.resolve())
    outputs = []
    for index, offset in enumerate(range(0, len(rows), args.waves_per_batch)):
        batch_rows = rows[offset : offset + args.waves_per_batch]
        output = args.output_root / f"batch_{index:02d}.json"
        atomic_write(
            output,
            {
                "schema_version": "apertus_mini_greekmmlu_backlog_batch_v1",
                "status": "frozen",
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "segment_id": args.segment_id,
                "evaluation_namespace": namespace,
                "authoritative_evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
                "campaign_manifest": {
                    "path": str(args.campaign_manifest.resolve()),
                    "sha256": manifest_sha,
                },
                "scientific_execution_unchanged": True,
                "mutation": "Slurm packing only: four existing one-node waves share one four-node allocation",
                "wave_count": len(batch_rows),
                "waves": batch_rows,
            },
        )
        outputs.append(str(output.resolve()))
    print(json.dumps({"ok": True, "wave_count": len(rows), "batches": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

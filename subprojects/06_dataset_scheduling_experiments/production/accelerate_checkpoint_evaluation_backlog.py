#!/usr/bin/env python3
"""Move one watcher-bound short evaluation at a time to Clariden debug."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from campaign_contract import evaluation_namespace, scoped_evaluation_root


ACTIVE = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def job_fields(text: str) -> dict[str, str]:
    return dict(re.findall(r"(?:^| )(\w+)=([^ ]*)", text))


def show_job(job_id: str) -> tuple[str, dict[str, str]] | None:
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    text = result.stdout.strip()
    return text, job_fields(text)


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def watcher_jobs(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    namespace = evaluation_namespace()
    for state_path in sorted(
        scoped_evaluation_root(run_root, "evaluation_watch").glob(
            "segment_*/iteration_*.json"
        )
    ):
        state = read_json(state_path)
        if state.get("evaluation_namespace") != namespace:
            raise ValueError(f"watch-state namespace drift: {state_path}")
        attempts = state.get("attempts", [])
        if state.get("status") != "submitted" or not attempts:
            continue
        attempt = attempts[-1]
        observed = show_job(str(attempt["job_id"]))
        if observed is None:
            continue
        raw, fields = observed
        rows.append(
            {
                "iteration": int(state["iteration"]),
                "attempt": int(attempt["attempt"]),
                "job_id": str(attempt["job_id"]),
                "state_path": state_path,
                "state_sha256": sha256_file(state_path),
                "raw": raw,
                "fields": fields,
            }
        )
    return rows


def validate_candidate(row: dict[str, Any]) -> None:
    fields = row["fields"]
    required = {
        "JobName": "mini_greekmmlu_wave",
        "JobState": "PENDING",
        "Partition": "normal",
        "Account": "a0140",
        "QOS": "normal",
        "TimeLimit": "00:30:00",
        "NumNodes": "1-1",
    }
    drift = {key: (fields.get(key), value) for key, value in required.items() if fields.get(key) != value}
    if drift:
        raise ValueError(f"refusing evaluation job with scheduling-contract drift: {drift}")
    if "gres/gpu=4" not in fields.get("ReqTRES", ""):
        raise ValueError("refusing evaluation job without exactly the expected GPU request")


def promote_once(
    run_root: Path,
    campaign_manifest: Path,
    *,
    dry_run: bool,
    max_active_debug_evaluations: int,
) -> dict[str, Any]:
    if not 1 <= max_active_debug_evaluations <= 4:
        raise ValueError("max active debug evaluations must be between one and four")
    jobs = watcher_jobs(run_root)
    debug = [
        row
        for row in jobs
        if row["fields"].get("Partition") == "debug"
        and row["fields"].get("JobState") in ACTIVE
    ]
    if len(debug) >= max_active_debug_evaluations:
        return {
            "action": "wait",
            "active_debug_evaluations": len(debug),
            "max_active_debug_evaluations": max_active_debug_evaluations,
            "debug_job_ids": [row["job_id"] for row in debug],
        }
    candidates = [
        row
        for row in jobs
        if row["fields"].get("Partition") == "normal"
        and row["fields"].get("JobState") == "PENDING"
    ]
    if not candidates:
        return {"action": "wait", "reason": "no_pending_normal_evaluation"}
    row = min(candidates, key=lambda value: (value["iteration"], value["attempt"]))
    validate_candidate(row)
    if dry_run:
        return {
            "action": "would_promote",
            "job_id": row["job_id"],
            "iteration": row["iteration"],
            "attempt": row["attempt"],
        }
    result = subprocess.run(
        ["scontrol", "update", f"JobId={row['job_id']}", "Partition=debug"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return {
            "action": "raced",
            "job_id": row["job_id"],
            "stderr": result.stderr.strip(),
        }
    observed = show_job(row["job_id"])
    if observed is None or observed[1].get("Partition") != "debug":
        raise RuntimeError("scontrol accepted update but debug partition binding is absent")
    after_raw, after_fields = observed
    receipt = (
        run_root
        / "orchestration"
        / "scheduling_requeues"
        / f"iteration_{row['iteration']:07d}_attempt_{row['attempt']}_job_{row['job_id']}.json"
    )
    campaign_manifest = campaign_manifest.resolve()
    payload = {
        "schema_version": "apertus_mini_evaluation_job_partition_recovery_v1",
        "status": "submitted",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "job_id": row["job_id"],
        "iteration": row["iteration"],
        "attempt": row["attempt"],
        "from_partition": "normal",
        "to_partition": "debug",
        "active_debug_evaluations_before_promotion": len(debug),
        "max_active_debug_evaluations": max_active_debug_evaluations,
        "scientific_execution_unchanged": True,
        "allowed_mutation": "pending Slurm partition only",
        "campaign_manifest": {
            "path": str(campaign_manifest),
            "sha256": sha256_file(campaign_manifest),
        },
        "watch_state_snapshot": {
            "path": str(row["state_path"].resolve()),
            "sha256_at_requeue": row["state_sha256"],
        },
        "before": row["raw"],
        "after": after_raw,
        "after_state": after_fields.get("JobState"),
    }
    atomic_write(receipt, payload)
    return {
        "action": "promoted",
        "job_id": row["job_id"],
        "iteration": row["iteration"],
        "attempt": row["attempt"],
        "receipt": str(receipt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=86_000)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-active-debug-evaluations", type=int, default=1)
    args = parser.parse_args()
    begin = time.monotonic()
    while True:
        result = promote_once(
            args.run_root.resolve(),
            args.campaign_manifest.resolve(),
            dry_run=args.dry_run,
            max_active_debug_evaluations=args.max_active_debug_evaluations,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.once or args.dry_run:
            return 0
        if time.monotonic() - begin >= args.max_seconds:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

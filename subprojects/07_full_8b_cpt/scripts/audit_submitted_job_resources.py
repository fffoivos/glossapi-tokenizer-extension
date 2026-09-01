#!/usr/bin/env python3
"""Fail closed when submitted full-8B jobs use the wrong Clariden resources."""

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


FIELDS = re.compile(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)")
NODE_COUNT = re.compile(r"^(\d+)(?:-(\d+))?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_binding(root: Path, receipt: Path, kind: str) -> dict:
    root = root.resolve()
    receipt = receipt.resolve()
    value = json.loads(receipt.read_text(encoding="utf-8"))
    checks = {
        "receipt_is_frozen": value.get("status") == "frozen",
        "receipt_kind_matches": value.get("kind") == kind,
        "receipt_root_matches": Path(value.get("root", "")).resolve() == root,
        "tree_sha_is_present": bool(value.get("tree_sha256")),
    }
    if not root.is_dir() or not all(checks.values()):
        raise ValueError({"bundle_kind": kind, "checks": checks})
    return {
        "root": str(root),
        "receipt": str(receipt),
        "receipt_sha256": sha256_file(receipt),
        "tree_sha256": value["tree_sha256"],
        "checks": checks,
    }


def parse_time(value: str) -> int:
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    parts = list(map(int, value.split(":")))
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        raise ValueError(value)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_exact_nodes(value: str) -> int:
    """Accept Slurm's exact pending range (``16-16``), reject a real range."""

    match = NODE_COUNT.fullmatch(value)
    if not match:
        raise ValueError(f"invalid NumNodes: {value}")
    lower = int(match.group(1))
    upper = int(match.group(2) or lower)
    if lower != upper:
        raise ValueError(f"NumNodes is a range, not an exact allocation: {value}")
    return lower


def read_job(job_id: str, *, attempts: int = 5, delay_seconds: float = 1.0) -> dict[str, str]:
    last_error = ""
    for attempt in range(attempts):
        result = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            text=True, capture_output=True, check=False,
        )
        rows = FIELDS.findall(result.stdout.strip())
        if result.returncode == 0 and rows:
            return dict(rows)
        last_error = result.stderr.strip() or result.stdout.strip()
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)
    raise ValueError(f"unable to read scontrol job {job_id}: {last_error}")


def audit(role: str, job_id: str, row: dict[str, str]) -> dict:
    partition = row.get("Partition")
    nodes = parse_exact_nodes(row.get("NumNodes", "0"))
    seconds = parse_time(row.get("TimeLimit", "0:00"))
    switches = row.get("ReqSwitch") or row.get("Switches", "")
    if role == "train":
        checks = {
            "partition_is_normal": partition == "normal",
            "nodes_are_16": nodes == 16,
            "time_is_12_hours": seconds == 12 * 3600,
            "one_leaf_switch_requested": bool(re.fullmatch(r"1(?:@[\d:]+)?", switches)),
        }
    elif role == "per_document_continuation":
        checks = {
            "partition_is_debug": partition == "debug",
            "nodes_are_one": nodes == 1,
            "time_is_85_minutes": seconds == 85 * 60,
            "node_minutes_fit_debug_qos": nodes * seconds <= 90 * 60,
            "four_gpus_requested": "gres/gpu=4" in row.get("ReqTRES", ""),
            "full_node_cpus_requested": row.get("NumCPUs") == "288",
        }
    else:
        checks = {
            "partition_is_debug": partition == "debug",
            "nodes_fit_debug": 1 <= nodes <= 4,
            "time_fits_debug": 0 < seconds <= 90 * 60,
            # Clariden's a0140 debug QOS caps the product of requested nodes
            # and wall time at 90 node-minutes per job.  Checking wall time
            # alone allowed an impossible four-node, 90-minute request to be
            # certified and left pending forever.
            "node_minutes_fit_debug_qos": nodes * seconds <= 90 * 60,
        }
    if not all(checks.values()):
        raise ValueError({"role": role, "job_id": job_id, "checks": checks, "scontrol": row})
    return {
        "role": role, "job_id": job_id, "partition": partition,
        "nodes": nodes, "time_limit_seconds": seconds,
        "state": row.get("JobState"), "switches": switches, "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--job", action="append", default=[], metavar="ROLE=JOB_ID")
    parser.add_argument("--scientific-root", type=Path, required=True)
    parser.add_argument("--scientific-receipt", type=Path, required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--ops-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs: list[tuple[str, str]] = []
    if args.manifest:
        value = json.loads(args.manifest.read_text(encoding="utf-8"))
        if value.get("schema_version") != "apertus_full_8b_resource_aware_launch_graph_v1" or value.get("status") != "submitted":
            raise ValueError("launch graph is not submitted")
        jobs.extend((str(row["role"]), str(row["job_id"])) for row in value["jobs"])
    for raw in args.job:
        role, separator, job_id = raw.partition("=")
        if not separator or not role or not job_id:
            raise ValueError(f"invalid --job: {raw}")
        jobs.append((role, job_id))
    if not jobs:
        raise ValueError("no jobs to audit")
    observed = [audit(role, job_id, read_job(job_id)) for role, job_id in jobs]
    payload = {
        "schema_version": "apertus_full_8b_allocation_routing_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "jobs": observed,
        "policy": {
            "train": "normal_16node_12h_single_leaf",
            "auxiliary": "debug_1to4node_at_most_90_total_node_minutes",
        },
        "scientific_bundle": bundle_binding(
            args.scientific_root, args.scientific_receipt, "scientific"
        ),
        "operational_bundle": bundle_binding(
            args.ops_root, args.ops_receipt, "efficiency"
        ),
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "jobs": len(observed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

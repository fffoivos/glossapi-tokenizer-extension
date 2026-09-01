#!/usr/bin/env python3
"""Capture storage and scheduler evidence immediately before production launch."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json


def run(command: list[str]) -> dict:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--leaf-switch", required=True)
    parser.add_argument("--leaf-exclusion", required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=6_000_000_000_000)
    args = parser.parse_args()
    profile = read_json(args.selected_profile)
    nodes = int(profile["selection"]["nodes"])
    storage = shutil.disk_usage(args.storage_path)
    scheduler = run(["sinfo", "-h", "-p", "normal", "-o", "%a|%D|%t|%l"])
    test_only = run([
        "sbatch", "--test-only", "--uenv-passthrough=ignore",
        "--account=a0140", "--partition=normal", f"--nodes={nodes}",
        "--switches=1", f"--exclude={args.leaf_exclusion}",
        "--exclusive", "--mem=450G", "--time=12:00:00", "--wrap=true",
    ])
    checks = {
        "storage_headroom_at_least_6TB": storage.free >= args.minimum_free_bytes,
        "normal_partition_snapshot_succeeded": scheduler["returncode"] == 0,
        "test_only_prediction_succeeded": test_only["returncode"] == 0,
        "leaf_switch_is_explicit": args.leaf_switch.startswith("group"),
        "leaf_exclusion_is_nonempty": bool(args.leaf_exclusion),
    }
    payload = {
        "schema_version": "apertus_full_8b_launch_environment_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selected_profile": file_binding(args.selected_profile),
        "nodes": nodes,
        "storage": {"path": str(args.storage_path.resolve()), "total_bytes": storage.total, "used_bytes": storage.used, "free_bytes": storage.free, "minimum_free_bytes": args.minimum_free_bytes},
        "scheduler": scheduler,
        "placement": {
            "leaf_switch": args.leaf_switch,
            "excluded_nodes": args.leaf_exclusion,
            "policy": "scheduler exclusion confines every training allocation to one Clariden Level=0 leaf switch",
        },
        "test_only": {**test_only, "is_prediction_not_reservation": True},
        "checks": checks,
    }
    atomic_write_json(args.output, payload)
    if not all(checks.values()):
        raise RuntimeError(f"launch environment failed: {checks}")
    print(json.dumps({"ok": True, "nodes": nodes, "free_bytes": storage.free, "test_only": test_only["stdout"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

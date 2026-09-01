#!/usr/bin/env python3
"""Freeze the newest checkpoint common to all five arms after an infra failure."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from campaign_contract import ARMS, atomic_write_json, read_json, sha256_file
from freeze_segment_checkpoint import tree_receipt


def checkpoint_iterations(root: Path) -> set[int]:
    values: set[int] = set()
    for path in root.glob("iter_*"):
        match = re.fullmatch(r"iter_(\d{7})", path.name)
        if match and path.is_dir() and (path / ".metadata").is_file():
            values.add(int(match.group(1)))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--segment-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--segment-attempt", type=int, required=True)
    parser.add_argument("--start-iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = read_json(args.campaign_manifest)
    segment = campaign["segments"][args.segment_id]
    nominal_start = int(segment["start_iteration"])
    end = int(segment["end_iteration"])
    start = int(args.start_iteration)
    if not nominal_start <= start < end:
        raise ValueError("recovery start outside segment")
    segment_root = (
        args.run_root
        / "segments"
        / f"segment_{args.segment_id}"
        / f"attempt_{args.segment_attempt}"
    )
    execution = read_json(segment_root / "segment_state.json")
    if execution.get("status") != "failed":
        raise ValueError("recovery receipt requires a failed aggregate segment")

    roots = {arm: segment_root / arm / "checkpoints" for arm in ARMS}
    common: set[int] | None = None
    for root in roots.values():
        values = checkpoint_iterations(root)
        common = values if common is None else common & values
    candidates = sorted(value for value in (common or set()) if start < value <= end)
    if not candidates:
        raise RuntimeError("failed attempt produced no newer common five-arm checkpoint")
    iteration = candidates[-1]

    arms = []
    for arm in ARMS:
        driver_log = segment_root / arm / "driver.out"
        if not driver_log.is_file():
            raise ValueError(f"missing driver log for {arm}")
        text = driver_log.read_text(encoding="utf-8", errors="replace")
        if re.search(r"skipped iterations\s*:\s*[1-9]", text, re.IGNORECASE):
            raise ValueError(f"nonzero skipped iteration detected for {arm}")
        if re.search(r"(?:lm loss|grad norm)\s*:\s*(?:nan|inf)", text, re.IGNORECASE):
            raise ValueError(f"non-finite training diagnostic detected for {arm}")
        checkpoint_root = roots[arm]
        arms.append(
            {
                "arm_id": arm,
                "checkpoint_root": str(checkpoint_root.resolve()),
                "checkpoint": tree_receipt(checkpoint_root / f"iter_{iteration:07d}"),
                "driver_log_sha256": sha256_file(driver_log),
            }
        )
    payload = {
        "schema_version": "apertus_mini_segment_checkpoint_receipt_v1",
        "status": "passed",
        "purpose": "common_infrastructure_failure_recovery",
        "segment_id": args.segment_id,
        "segment_attempt": args.segment_attempt,
        "iteration": iteration,
        "campaign_manifest": str(args.campaign_manifest.resolve()),
        "campaign_manifest_sha256": sha256_file(args.campaign_manifest),
        "failed_execution_state": str((segment_root / "segment_state.json").resolve()),
        "arms": arms,
        "full_state_required": ["model", "optimizer", "scheduler", "rng", "data_cursor"],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "iteration": iteration, "arms": len(arms)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

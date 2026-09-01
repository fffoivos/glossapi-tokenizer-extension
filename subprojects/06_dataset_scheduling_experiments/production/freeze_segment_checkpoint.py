#!/usr/bin/env python3
"""Freeze all five exact segment-end checkpoints into one resume receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from campaign_contract import ARMS, atomic_write_json, read_json, sha256_file


def tree_receipt(root: Path) -> dict:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files or not (root / ".metadata").is_file():
        raise ValueError(f"incomplete distributed checkpoint: {root}")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "root": str(root.resolve()),
        "files": files,
        "bytes": sum(int(row["bytes"]) for row in files),
        "tree_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--segment-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--segment-attempt", type=int, default=0)
    parser.add_argument("--iteration", type=int)
    parser.add_argument(
        "--failed-attempt-recovery",
        action="store_true",
        help="Freeze an earlier common checkpoint from a failed attempt for audited recovery.",
    )
    parser.add_argument(
        "--numerical-failure-recovery",
        action="store_true",
        help=(
            "Freeze an earlier clean checkpoint from an attempt whose processes "
            "exited zero but whose later training diagnostics became non-finite."
        ),
    )
    parser.add_argument(
        "--load-view-root",
        type=Path,
        help="New root for exact-iteration checkpoint views used by recovery loads.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = read_json(args.campaign_manifest)
    segment = campaign["segments"][args.segment_id]
    iteration = int(segment["end_iteration"]) if args.iteration is None else int(args.iteration)
    if not int(segment["start_iteration"]) < iteration <= int(segment["end_iteration"]):
        raise ValueError("checkpoint receipt iteration outside segment")
    segment_root = args.run_root / "segments" / f"segment_{args.segment_id}" / f"attempt_{args.segment_attempt}"
    if args.failed_attempt_recovery and args.numerical_failure_recovery:
        raise ValueError("failed-attempt and numerical-failure recovery are exclusive")
    recovery_mode = args.failed_attempt_recovery or args.numerical_failure_recovery
    execution = read_json(segment_root / "segment_state.json")
    expected_execution_status = "failed" if args.failed_attempt_recovery else "completed"
    if execution.get("status") != expected_execution_status:
        raise ValueError(
            "aggregate segment execution status drift: "
            f"expected {expected_execution_status}, got {execution.get('status')}"
        )
    if recovery_mode and args.iteration is None:
        raise ValueError("recovery requires an explicit common iteration")
    if recovery_mode != (args.load_view_root is not None):
        raise ValueError(
            "recovery mode and an exact load-view root must be supplied together"
        )
    arms = []
    for arm in ARMS:
        checkpoint_root = segment_root / arm / "checkpoints"
        latest = checkpoint_root / "latest_checkpointed_iteration.txt"
        if not latest.is_file():
            raise ValueError(f"latest checkpoint marker missing for {arm}")
        latest_iteration = int(latest.read_text().strip())
        if recovery_mode:
            if latest_iteration < iteration:
                raise ValueError(f"recovery checkpoint is newer than latest marker for {arm}")
        elif latest_iteration != iteration:
            raise ValueError(f"latest checkpoint marker drift for {arm}")
        driver_log = segment_root / arm / "driver.out"
        text = driver_log.read_text(encoding="utf-8", errors="replace")
        diagnostic_rows = [
            (int(match.group(1)), match.group(0))
            for match in re.finditer(
                r"iteration\s+(\d+)/\s*\d+\s*\|[^\n]+", text, re.IGNORECASE
            )
        ]
        clean_prefix = "\n".join(
            row for row_iteration, row in diagnostic_rows if row_iteration <= iteration
        )
        later_rows = "\n".join(
            row for row_iteration, row in diagnostic_rows if row_iteration > iteration
        )
        if re.search(r"skipped iterations\s*:\s*[1-9]", clean_prefix, re.IGNORECASE):
            raise ValueError(f"nonzero skipped iteration detected through checkpoint for {arm}")
        if re.search(
            r"(?:lm loss|grad norm)\s*:\s*(?:nan|inf)", clean_prefix, re.IGNORECASE
        ):
            raise ValueError(f"non-finite training diagnostic through checkpoint for {arm}")
        if args.numerical_failure_recovery and not re.search(
            r"(?:lm loss|grad norm)\s*:\s*(?:nan|inf)", later_rows, re.IGNORECASE
        ):
            raise ValueError(f"numerical recovery lacks a later non-finite diagnostic for {arm}")
        if not recovery_mode and re.search(
            r"(?:lm loss|grad norm)\s*:\s*(?:nan|inf)", later_rows, re.IGNORECASE
        ):
            raise ValueError(f"non-finite training diagnostic detected for {arm}")
        tree = tree_receipt(checkpoint_root / f"iter_{iteration:07d}")
        arms.append(
            {
                "arm_id": arm,
                "checkpoint_root": str(checkpoint_root.resolve()),
                "checkpoint": tree,
                "driver_log_sha256": sha256_file(driver_log),
                "latest_checkpointed_iteration": latest_iteration,
            }
        )
    if recovery_mode:
        view_root = args.load_view_root.resolve()
        if view_root.exists():
            raise FileExistsError(f"recovery load-view root already exists: {view_root}")
        view_root.mkdir(parents=True)
        for row in arms:
            arm = row["arm_id"]
            source_checkpoint_root = Path(row["checkpoint_root"]).resolve()
            source_iteration = source_checkpoint_root / f"iter_{iteration:07d}"
            arm_view = view_root / arm
            arm_view.mkdir()
            os.symlink(source_iteration, arm_view / source_iteration.name)
            (arm_view / "latest_checkpointed_iteration.txt").write_text(
                f"{iteration}\n", encoding="utf-8"
            )
            row["checkpoint_source_root"] = str(source_checkpoint_root)
            row["checkpoint_root"] = str(arm_view.resolve())
            row["load_view"] = {
                "root": str(arm_view.resolve()),
                "latest_checkpointed_iteration": iteration,
                "iteration_target": str(source_iteration.resolve()),
            }
    payload = {
        "schema_version": "apertus_mini_segment_checkpoint_receipt_v1",
        "status": "passed",
        "segment_id": args.segment_id,
        "segment_attempt": args.segment_attempt,
        "iteration": iteration,
        "campaign_manifest": str(args.campaign_manifest.resolve()),
        "campaign_manifest_sha256": sha256_file(args.campaign_manifest),
        "arms": arms,
        "full_state_required": ["model", "optimizer", "scheduler", "rng", "data_cursor"],
        "recovery_context": {
            "failed_attempt_recovery": args.failed_attempt_recovery,
            "numerical_failure_recovery": args.numerical_failure_recovery,
            "segment_state": str((segment_root / "segment_state.json").resolve()),
            "segment_state_sha256": sha256_file(segment_root / "segment_state.json"),
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "iteration": iteration, "arms": len(arms)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

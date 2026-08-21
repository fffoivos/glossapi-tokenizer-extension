#!/usr/bin/env python3
"""Classify a post-checkpoint torchrun teardown without hiding worker failures.

Experiment-side workaround for
https://github.com/fffoivos/apertus-cscs-efficiency/issues/128.

The bounded qualification workloads intentionally stop at an exact update.
Across multiple nodes, torchrun can return nonzero after every optimizer update
and distributed-checkpoint write has completed because the rendezvous server
closes while peer elastic agents are still leaving.  This verifier accepts only
that narrow evidence shape.  Restart parity remains the authoritative proof
that the resulting checkpoint is usable and numerically continuous.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path


ITERATION = re.compile(
    r"iteration\s+(?P<iteration>\d+)/.*?"
    r"number of skipped iterations:\s*(?P<skipped>\d+).*?"
    r"number of nan iterations:\s*(?P<nonfinite>\d+)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-update", type=int, required=True)
    parser.add_argument("--launcher-returncode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(args.launcher_returncode != 0, "classifier requires a nonzero launcher exit")
    require(args.expected_update > 0, "expected update must be positive")
    require(args.log.is_file(), "launcher log is missing")
    require(not args.output.exists(), "immutable teardown receipt already exists")

    text = args.log.read_text(encoding="utf-8", errors="replace")
    rows = {
        int(match.group("iteration")): {
            "skipped": int(match.group("skipped")),
            "nonfinite": int(match.group("nonfinite")),
        }
        for match in ITERATION.finditer(text)
    }
    require(args.expected_update in rows, "expected final optimizer row is missing")
    final = rows[args.expected_update]
    require(final == {"skipped": 0, "nonfinite": 0}, "final optimizer row is not finite")
    require(
        re.search(
            rf"successfully saved checkpoint from iteration\s+{args.expected_update}\b",
            text,
        )
        is not None,
        "expected successful checkpoint log row is missing",
    )
    require(
        "RendezvousConnectionError" in text
        and "Failed to recv, got 0 bytes" in text
        and "connection to the C10d store has failed" in text,
        "launcher failure is not the allowlisted post-checkpoint rendezvous teardown",
    )
    forbidden = (
        "CUDA out of memory",
        "OutOfMemoryError",
        "ChildFailedError",
        "NCCL watchdog",
        "Detected mismatch between collectives",
        "Segmentation fault",
    )
    present_forbidden = [marker for marker in forbidden if marker in text]
    require(not present_forbidden, f"non-teardown failure marker present: {present_forbidden}")

    iteration_dir = args.checkpoint_root / f"iter_{args.expected_update:07d}"
    metadata = iteration_dir / ".metadata"
    latest = args.checkpoint_root / "latest_checkpointed_iteration.txt"
    require(metadata.is_file(), "expected distributed checkpoint metadata is missing")
    require(latest.is_file(), "latest checkpoint cursor is missing")
    require(latest.read_text(encoding="utf-8").strip() == str(args.expected_update), "latest checkpoint cursor drift")

    payload = {
        "schema_version": "apertus_intentional_torchrun_teardown_v1",
        "status": "accepted_post_checkpoint_teardown",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "issue": "https://github.com/fffoivos/apertus-cscs-efficiency/issues/128",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "launcher_returncode": args.launcher_returncode,
        "expected_update": args.expected_update,
        "checks": {
            "expected_optimizer_row_present": True,
            "zero_skipped_updates": True,
            "zero_nonfinite_updates": True,
            "successful_checkpoint_log_present": True,
            "distributed_checkpoint_metadata_present": True,
            "latest_checkpoint_cursor_exact": True,
            "allowlisted_rendezvous_teardown_present": True,
            "forbidden_failure_markers_absent": True,
            "restart_parity_still_required": True,
        },
        "evidence": {
            "launcher_log": binding(args.log),
            "checkpoint_metadata": binding(metadata),
            "latest_checkpoint_cursor": binding(latest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

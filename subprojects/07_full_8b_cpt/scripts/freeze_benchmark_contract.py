#!/usr/bin/env python3
"""Freeze the exact D0 prefix shared by the 16- and 32-node benchmarks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json


def prefix_binding(path: Path, size: int) -> dict:
    resolved = path.resolve()
    with resolved.open("rb") as handle:
        payload = handle.read(size)
    if len(payload) != size:
        raise ValueError(f"short benchmark prefix: {resolved}")
    return {
        "path": str(resolved),
        "prefix_bytes": size,
        "prefix_sha256": hashlib.sha256(payload).hexdigest(),
        "full_file_bytes": resolved.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--goldfish-implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=288)
    parser.add_argument("--global-batch-sequences", type=int, default=1024)
    args = parser.parse_args()
    schedule = read_json(args.schedule_manifest)
    if schedule.get("schema_version") != "apertus_data_order_schedules_v1" or schedule.get("status") != "completed":
        raise ValueError("schedule is not complete")
    arms = {row["arm_id"]: row for row in schedule["arms"]}
    d0 = arms.get("D0_mixed")
    if d0 is None:
        raise ValueError("D0 schedule absent")
    sequences = args.updates * args.global_batch_sequences
    sequence_ids = Path(d0["sequence_ids"]["path"])
    active_tokens = Path(d0["active_tokens"]["path"])
    payload = {
        "schema_version": "apertus_full_8b_parallelism_benchmark_contract_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "schedule_manifest": file_binding(args.schedule_manifest),
        "arm_id": "D0_mixed",
        "updates": args.updates,
        "global_batch_sequences": args.global_batch_sequences,
        "sequences": sequences,
        "sequence_ids": prefix_binding(sequence_ids, sequences * 8),
        "active_tokens": prefix_binding(active_tokens, sequences * 2),
        "goldfish": {
            "k": 50,
            "h": 50,
            "mask_identity_proof": "same frozen sequence payloads and labels under the same deterministic Goldfish implementation",
            "implementation": file_binding(args.goldfish_implementation),
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "sequences": sequences, "sequence_prefix_sha256": payload["sequence_ids"]["prefix_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

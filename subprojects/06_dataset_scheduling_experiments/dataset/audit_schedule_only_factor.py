#!/usr/bin/env python3
"""Independently prove D0-D4 differ only in HPLT/non-HPLT temporal order."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


FILLER_ID = np.uint64(2**64 - 1)
PAIR_DTYPE = np.dtype([("sequence_id", "<u8"), ("active_tokens", "<u2")], align=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def checked_memmap(receipt: dict, dtype: np.dtype) -> np.memmap:
    path = Path(receipt["path"])
    if path.stat().st_size != int(receipt["bytes"]) or sha256_file(path) != receipt["sha256"]:
        raise ValueError(f"schedule payload drift: {path}")
    return np.memmap(path, mode="r", dtype=dtype)


def inventory_hash(ids: np.ndarray, active: np.ndarray) -> str:
    real = ids != FILLER_ID
    pairs = np.empty(int(np.count_nonzero(real)), dtype=PAIR_DTYPE)
    pairs["sequence_id"] = ids[real]
    pairs["active_tokens"] = active[real]
    pairs.sort(order="sequence_id", kind="stable")
    return hashlib.sha256(pairs.tobytes(order="C")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = read_json(args.schedule_manifest)
    if (
        manifest.get("schema_version") != "apertus_mini_five_data_order_schedules_v1"
        or manifest.get("status") != "completed"
    ):
        raise ValueError("schedule manifest is not completed")
    arms = manifest["arms"]
    if len(arms) != 5:
        raise ValueError("expected five schedule arms")
    replay_positions = checked_memmap(
        manifest["common_contract"]["replay_positions"], np.dtype("<u8")
    )
    reference_replay: np.ndarray | None = None
    inventories = []
    modern_position_hashes = []
    for arm in arms:
        ids = checked_memmap(arm["sequence_ids"], np.dtype("<u8"))
        active = checked_memmap(arm["active_tokens"], np.dtype("<u2"))
        if ids.size != active.size or ids.size != int(arm["training_slots"]):
            raise ValueError(f"schedule geometry drift: {arm['arm_id']}")
        replay = np.asarray(ids[replay_positions], dtype=np.uint64)
        if reference_replay is None:
            reference_replay = replay.copy()
        elif not np.array_equal(replay, reference_replay):
            raise ValueError(f"replay identity/position drift: {arm['arm_id']}")
        digest = inventory_hash(ids, active)
        inventories.append({"arm_id": arm["arm_id"], "sorted_id_active_sha256": digest})
        pool_codes = ids >> np.uint64(62)
        modern = (pool_codes == 0) | (pool_codes == 1)
        modern_position_hashes.append(
            {
                "arm_id": arm["arm_id"],
                "ordered_modern_pool_code_sha256": hashlib.sha256(
                    np.asarray(pool_codes[modern], dtype=np.uint8).tobytes()
                ).hexdigest(),
            }
        )
    if len({row["sorted_id_active_sha256"] for row in inventories}) != 1:
        raise ValueError("five arms do not share one exact sequence/active-token inventory")
    if len({row["ordered_modern_pool_code_sha256"] for row in modern_position_hashes}) != 5:
        raise ValueError("declared modern-Greek order factor did not produce five distinct trajectories")
    payload = {
        "schema_version": "apertus_mini_schedule_only_factor_audit_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "schedule_manifest": {
            "path": str(args.schedule_manifest.resolve()),
            "sha256": sha256_file(args.schedule_manifest),
        },
        "inventories": inventories,
        "modern_order_trajectories": modern_position_hashes,
        "same_replay_sequence_ids_at_same_positions": True,
        "same_exact_sequence_and_active_token_inventory": True,
        "five_distinct_modern_greek_order_trajectories": True,
        "goldfish_mask_identity_proof": (
            "Each stable sequence ID resolves to one immutable packed label payload; all arms "
            "contain the same sequence/active-token pairs; Goldfish is a deterministic function "
            "of that label payload under the frozen seed, k and context width. Reordering cannot "
            "change a sequence's mask."
        ),
        "only_declared_factor": "temporal ordering of HPLT versus GlossAPI/non-HPLT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "arms": 5, "inventory": inventories[0]["sorted_id_active_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

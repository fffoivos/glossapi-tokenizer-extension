#!/usr/bin/env python3
"""Build an exact-inventory schedule with a balanced 1,024-step prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from build_five_schedules import FILLER_ID, POOL_KEYS, SEQUENCE_DTYPE, sha256_file, smooth_categories


PREFIX_UPDATES = 1_024
GLOBAL_BATCH = 512
PREFIX_SLOTS = PREFIX_UPDATES * GLOBAL_BATCH
PREFIX_COUNTS = {"H": 207_094, "G": 207_094, "F": 104_858, "O": 5_242}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sum(PREFIX_COUNTS.values()) != PREFIX_SLOTS:
        raise AssertionError("prefix quota arithmetic drift")
    packed = read_json(args.packed_receipt)
    if packed.get("schema_version") != "apertus_mini_packed_sequence_corpus_v1" or packed.get("status") != "completed":
        raise ValueError("packed corpus receipt is incomplete")
    streams = {}
    for short, pool in POOL_KEYS.items():
        receipt = packed["pools"][pool]["sequence_catalog"]
        path = Path(receipt["path"])
        if not path.is_file() or path.stat().st_size != int(receipt["bytes"]) or sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"sequence catalog drift: {path}")
        streams[short] = np.memmap(path, mode="r", dtype=SEQUENCE_DTYPE)
        if streams[short].size < PREFIX_COUNTS[short]:
            raise ValueError(f"insufficient {short} sequences for stability prefix")
    prefix_categories = smooth_categories(PREFIX_COUNTS, ("H", "G", "F", "O"))
    remaining = {key: int(stream.size) - PREFIX_COUNTS[key] for key, stream in streams.items()}
    tail_categories = smooth_categories(remaining, ("H", "G", "F", "O"))
    categories = np.concatenate((prefix_categories, tail_categories))
    ids = np.empty(categories.size, dtype=np.uint64)
    active = np.empty(categories.size, dtype=np.uint16)
    for key, stream in streams.items():
        prefix_positions = np.flatnonzero(prefix_categories == key.encode())
        tail_positions = PREFIX_SLOTS + np.flatnonzero(tail_categories == key.encode())
        ids[prefix_positions] = stream["sequence_id"][: PREFIX_COUNTS[key]]
        active[prefix_positions] = stream["active_tokens"][: PREFIX_COUNTS[key]]
        ids[tail_positions] = stream["sequence_id"][PREFIX_COUNTS[key] :]
        active[tail_positions] = stream["active_tokens"][PREFIX_COUNTS[key] :]
    pad_slots = (-ids.size) % GLOBAL_BATCH
    real_sequences = int(ids.size)
    if pad_slots:
        ids = np.concatenate((ids, np.full(pad_slots, FILLER_ID, dtype=np.uint64)))
        active = np.concatenate((active, np.zeros(pad_slots, dtype=np.uint16)))
    args.output_dir.mkdir(parents=True)
    ids_path = args.output_dir / "stability_balanced.sequence_ids.u64"
    active_path = args.output_dir / "stability_balanced.active_tokens.u16"
    ids.tofile(ids_path)
    active.tofile(active_path)
    prefix_active = {
        key: int(active[:PREFIX_SLOTS][prefix_categories == key.encode()].astype(np.uint64).sum())
        for key in PREFIX_COUNTS
    }
    inventory = hashlib.sha256()
    for key in ("H", "G", "F", "O"):
        inventory.update(streams[key]["sequence_id"].tobytes())
        inventory.update(streams[key]["active_tokens"].tobytes())
    payload = {
        "schema_version": "apertus_mini_five_data_order_schedules_v1",
        "status": "completed",
        "purpose": "common_peak_lr_stability_smoke_only",
        "packed_corpus_receipt": {"path": str(args.packed_receipt.resolve()), "sha256": sha256_file(args.packed_receipt)},
        "common_contract": {
            "arms": ["stability_balanced"],
            "global_batch_sequences": GLOBAL_BATCH,
            "same_exact_sequence_multiset": True,
            "same_replay_sequence_ids_at_same_global_positions": True,
            "same_per_sequence_active_token_count": True,
            "canonical_sequence_inventory_sha256": inventory.hexdigest(),
            "prefix_updates": PREFIX_UPDATES,
            "prefix_sequence_counts": PREFIX_COUNTS,
            "prefix_active_tokens": prefix_active,
            "prefix_modern_mix": "HPLT and non-HPLT equal by scheduled sequence count; replay stationary at 20% foreign plus 1% Old Greek by sequence quota",
        },
        "arms": [
            {
                "arm_id": "stability_balanced",
                "sequence_ids": {"path": str(ids_path.resolve()), "sha256": sha256_file(ids_path), "bytes": ids_path.stat().st_size},
                "active_tokens": {"path": str(active_path.resolve()), "sha256": sha256_file(active_path), "bytes": active_path.stat().st_size},
                "training_slots": int(ids.size),
                "real_sequences": real_sequences,
                "pad_only_filler_slots": pad_slots,
                "optimizer_updates": int(ids.size // GLOBAL_BATCH),
            }
        ],
    }
    out = args.output_dir / "schedule_manifest.json"
    partial = Path(str(out) + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, out)
    print(json.dumps({"ok": True, "prefix_updates": PREFIX_UPDATES, "training_slots": int(ids.size)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

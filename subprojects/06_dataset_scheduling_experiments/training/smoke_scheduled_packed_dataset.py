#!/usr/bin/env python3
"""Smoke the Megatron schedule adapter on two arms sharing one payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from scheduled_packed_dataset import ScheduledPackedGPTDataset


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_receipt(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def fixture(root: Path) -> Path:
    text = np.arange(4097, dtype=np.int32) % 1000
    text[0] = 2
    text[100] = 2
    text[4001:] = 10
    binary = root / "bucket.bin"
    active = root / "bucket.active.u16"
    text.tofile(binary)
    np.asarray([4000], dtype=np.uint16).tofile(active)
    bucket_manifest = root / "bucket.manifest.json"
    bucket_manifest.write_text(
        json.dumps(
            {
                "schema_version": "apertus_mini_fixed_sequence_bucket_v1",
                "status": "completed",
                "pool_code": 0,
                "bucket": 0,
                "pad_token_id": 10,
                "sequence_count": 1,
                "stored_tokens_per_sequence": 4097,
                "sequence_length": 4096,
                "outputs": {
                    "bin": file_receipt(binary),
                    "active_counts": file_receipt(active),
                },
            }
        )
        + "\n"
    )
    packed = root / "packed.json"
    packed.write_text(
        json.dumps(
            {
                "schema_version": "apertus_mini_packed_sequence_corpus_v1",
                "status": "completed",
                "packing_task_manifests": [
                    {"manifest_path": str(bucket_manifest), "pool": "hplt", "bucket": 0}
                ],
            }
        )
        + "\n"
    )
    sequence_id = np.uint64(0)
    arms = []
    for arm in ("D0_mixed", "D1_hard_h_to_g"):
        ids = root / f"{arm}.ids"
        counts = root / f"{arm}.active"
        np.asarray([sequence_id, np.uint64(2**64 - 1)], dtype=np.uint64).tofile(ids)
        np.asarray([4000, 0], dtype=np.uint16).tofile(counts)
        arms.append(
            {
                "arm_id": arm,
                "training_slots": 2,
                "sequence_ids": file_receipt(ids),
                "active_tokens": file_receipt(counts),
            }
        )
    schedule = root / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "schema_version": "apertus_mini_five_data_order_schedules_v1",
                "status": "completed",
                "common_contract": {
                    "same_exact_sequence_multiset": True,
                    "same_replay_sequence_ids_at_same_global_positions": True,
                },
                "packed_corpus_receipt": {"path": str(packed), "sha256": sha(packed)},
                "arms": arms,
            }
        )
        + "\n"
    )
    return schedule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    import torch

    with tempfile.TemporaryDirectory() as temporary:
        schedule = fixture(Path(temporary))
        outputs = []
        for arm in ("D0_mixed", "D1_hard_h_to_g"):
            dataset = ScheduledPackedGPTDataset(
                schedule,
                arm,
                eod_token_id=2,
                reset_position_ids=True,
                reset_attention_mask=True,
                eod_mask_loss=True,
                create_attention_mask=False,
                goldfish_loss=True,
                goldfish_k=50,
                goldfish_h=50,
            )
            sample = dataset[0]
            filler = dataset[1]
            assert sample["tokens"].shape == sample["labels"].shape == (4096,)
            assert int(sample["loss_mask"][4000:].sum()) == 0
            assert int(filler["loss_mask"].sum()) == 0
            assert sample["position_ids"][101].item() == 0
            outputs.append(sample)
        for key in ("tokens", "labels", "loss_mask", "position_ids"):
            assert torch.equal(outputs[0][key], outputs[1][key]), key
    print(json.dumps({"ok": True, "same_payload_and_goldfish_mask_across_arms": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


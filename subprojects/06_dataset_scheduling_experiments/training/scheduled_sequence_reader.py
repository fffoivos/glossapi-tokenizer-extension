#!/usr/bin/env python3
"""Read immutable packed rows through a receipt-bound sequence-ID schedule.

The schedule changes only the order in which stable sequence payloads are
visited.  It never reconstructs or repacks text, which keeps document masks,
Goldfish inputs, replay placement, and resume cursors comparable across arms.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FILLER_ID = np.uint64(2**64 - 1)
STORED_TOKENS = 4097
TARGET_TOKENS = 4096
POOL_SHIFT = 62
BUCKET_SHIFT = 55
BUCKET_MASK = 0x7F
ROW_MASK = (1 << BUCKET_SHIFT) - 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def checked_file(receipt: dict[str, Any], *, hash_file: bool = True) -> Path:
    path = Path(receipt["path"]).resolve()
    if not path.is_file() or path.stat().st_size != int(receipt["bytes"]):
        raise ValueError(f"receipt-bound file size drift: {path}")
    if hash_file and sha256_file(path) != receipt["sha256"]:
        raise ValueError(f"receipt-bound file checksum drift: {path}")
    return path


@dataclass(frozen=True)
class SequencePayload:
    sequence_id: int
    tokens: np.ndarray
    active_tokens: int
    filler: bool


class _PackedBucket:
    def __init__(self, manifest_path: Path):
        manifest = read_json(manifest_path)
        if (
            manifest.get("schema_version")
            not in {
                "apertus_mini_fixed_sequence_bucket_v1",
                "apertus_fixed_sequence_bucket_v1",
            }
            or manifest.get("status") != "completed"
            or int(manifest.get("stored_tokens_per_sequence", -1)) != STORED_TOKENS
            or int(manifest.get("sequence_length", -1)) != TARGET_TOKENS
        ):
            raise ValueError(f"unsupported packed bucket manifest: {manifest_path}")
        self.pool_code = int(manifest["pool_code"])
        self.bucket = int(manifest["bucket"])
        self.pad_token_id = int(manifest["pad_token_id"])
        self.sequence_count = int(manifest["sequence_count"])
        self.bin_path = checked_file(manifest["outputs"]["bin"], hash_file=False)
        self.active_path = checked_file(
            manifest["outputs"]["active_counts"], hash_file=True
        )
        expected_bytes = self.sequence_count * STORED_TOKENS * np.dtype(np.int32).itemsize
        if self.bin_path.stat().st_size != expected_bytes:
            raise ValueError(f"packed binary geometry drift: {self.bin_path}")
        self._tokens: np.memmap | None = None
        self._active: np.memmap | None = None

    def _open(self) -> None:
        if self._tokens is None:
            self._tokens = np.memmap(
                self.bin_path,
                mode="r",
                dtype=np.int32,
                shape=(self.sequence_count, STORED_TOKENS),
            )
            self._active = np.memmap(
                self.active_path,
                mode="r",
                dtype=np.uint16,
                shape=(self.sequence_count,),
            )

    def get(self, row: int) -> tuple[np.ndarray, int]:
        if row < 0 or row >= self.sequence_count:
            raise IndexError(f"packed row outside bucket: {row}/{self.sequence_count}")
        self._open()
        assert self._tokens is not None and self._active is not None
        return np.array(self._tokens[row], dtype=np.int64, copy=True), int(self._active[row])


class ScheduledSequenceReader:
    """Resolve one schedule slot to its immutable 4097-token payload."""

    def __init__(self, schedule_manifest: Path, arm_id: str):
        schedule_manifest = schedule_manifest.resolve()
        schedule = read_json(schedule_manifest)
        schema = schedule.get("schema_version")
        legacy_schemas = {
            "apertus_mini_five_data_order_schedules_v1",
            "apertus_data_order_schedules_v1",
        }
        if schema in legacy_schemas:
            if (
                schedule.get("status") != "completed"
                or schedule.get("common_contract", {}).get("same_exact_sequence_multiset")
                is not True
                or schedule.get("common_contract", {}).get(
                    "same_replay_sequence_ids_at_same_global_positions"
                )
                is not True
            ):
                raise ValueError("schedule manifest is not a completed five-arm receipt")
            arms = {row["arm_id"]: row for row in schedule["arms"]}
            if arm_id not in arms:
                raise KeyError(f"unknown schedule arm: {arm_id}")
            arm = arms[arm_id]
            ids_receipt = arm["sequence_ids"]
            active_receipt = arm["active_tokens"]
            length = int(arm["training_slots"])
            packed_receipts = [schedule["packed_corpus_receipt"]]
        elif schema == "apertus_cpt_stationary_replay_schedule_v1":
            if (
                schedule.get("status") != "completed"
                or schedule.get("checks", {}).get("every_selected_sequence_scheduled_once")
                is not True
                or schedule.get("checks", {}).get("payload_not_rewritten") is not True
            ):
                raise ValueError("stationary schedule is not a completed immutable receipt")
            schedule_id = schedule.get("schedule_id")
            if arm_id != schedule_id:
                raise KeyError(f"unknown schedule arm: {arm_id}")
            ids_receipt = schedule["sequence_ids"]
            active_receipt = schedule["active_tokens_per_sequence"]
            length = int(schedule["sequence_count"])
            packed_receipts = schedule.get("packed_corpus_receipts")
            if not isinstance(packed_receipts, list) or not packed_receipts:
                raise ValueError("stationary schedule omits packed corpus receipts")
        else:
            raise ValueError(f"unsupported schedule schema: {schema}")

        ids_path = checked_file(ids_receipt)
        active_path = checked_file(active_receipt)
        self.arm_id = arm_id
        self.schedule_manifest = schedule_manifest
        self.length = length
        if ids_path.stat().st_size != self.length * 8:
            raise ValueError("sequence-ID schedule byte geometry drift")
        if active_path.stat().st_size != self.length * 2:
            raise ValueError("active-token schedule byte geometry drift")
        self.sequence_ids = np.memmap(
            ids_path, mode="r", dtype=np.uint64, shape=(self.length,)
        )
        self.active_tokens = np.memmap(
            active_path, mode="r", dtype=np.uint16, shape=(self.length,)
        )

        self.buckets: dict[tuple[int, int], _PackedBucket] = {}
        pad_ids = set()
        for packed_receipt_info in packed_receipts:
            packed_path = Path(packed_receipt_info["path"]).resolve()
            if sha256_file(packed_path) != packed_receipt_info["sha256"]:
                raise ValueError("packed corpus receipt drift")
            packed = read_json(packed_path)
            if (
                packed.get("schema_version")
                not in {
                    "apertus_mini_packed_sequence_corpus_v1",
                    "apertus_packed_sequence_corpus_v1",
                }
                or packed.get("status") != "completed"
            ):
                raise ValueError("packed corpus is not completed")
            for row in packed["packing_task_manifests"]:
                manifest_path = Path(row["manifest_path"]).resolve()
                expected_manifest_sha = row.get("manifest_sha256")
                if expected_manifest_sha and sha256_file(manifest_path) != expected_manifest_sha:
                    raise ValueError(f"packed bucket manifest drift: {manifest_path}")
                bucket = _PackedBucket(manifest_path)
                key = (bucket.pool_code, bucket.bucket)
                if key in self.buckets:
                    raise ValueError(f"duplicate packed bucket: {key}")
                self.buckets[key] = bucket
                pad_ids.add(bucket.pad_token_id)
        if len(pad_ids) != 1:
            raise ValueError(f"packed pad-token drift: {sorted(pad_ids)}")
        self.pad_token_id = next(iter(pad_ids))

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def decode_sequence_id(sequence_id: int) -> tuple[int, int, int]:
        return (
            sequence_id >> POOL_SHIFT,
            (sequence_id >> BUCKET_SHIFT) & BUCKET_MASK,
            sequence_id & ROW_MASK,
        )

    def __getitem__(self, index: int) -> SequencePayload:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        sequence_id = int(self.sequence_ids[index])
        scheduled_active = int(self.active_tokens[index])
        if sequence_id == int(FILLER_ID):
            if scheduled_active != 0:
                raise ValueError("filler schedule slot carries active loss tokens")
            return SequencePayload(
                sequence_id=sequence_id,
                tokens=np.zeros(STORED_TOKENS, dtype=np.int64),
                active_tokens=0,
                filler=True,
            )
        pool, bucket, row = self.decode_sequence_id(sequence_id)
        source = self.buckets.get((pool, bucket))
        if source is None:
            raise KeyError(f"schedule points to missing packed bucket: {(pool, bucket)}")
        tokens, packed_active = source.get(row)
        if scheduled_active <= 0 or scheduled_active > packed_active:
            raise ValueError(
                f"invalid scheduled active-token cap for {sequence_id}: "
                f"{scheduled_active} not in [1, {packed_active}]"
            )
        return SequencePayload(
            sequence_id=sequence_id,
            tokens=tokens,
            active_tokens=scheduled_active,
            filler=False,
        )

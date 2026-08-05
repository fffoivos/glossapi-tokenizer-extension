#!/usr/bin/env python3
"""Pack one identity-ordered catalog bucket into immutable 4097-token rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CATALOG_DTYPE = np.dtype(
    [
        ("pool", "u1"),
        ("task_index", "<u4"),
        ("document_index", "<u4"),
        ("tokens", "<u4"),
        ("identity", "V16"),
        ("order", "V16"),
    ],
    align=False,
)
INDEX_HEADER = b"MMIDIDX\x00\x00"


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


def write_index(path: Path, lengths: list[int]) -> None:
    """Write the Megatron mmap index without an external bridge dependency."""

    values = np.asarray(lengths, dtype=np.int32)
    pointers = np.empty(len(values), dtype=np.int64)
    offset = 0
    for index, length in enumerate(values):
        pointers[index] = offset
        offset += int(length) * 4
    documents = np.arange(len(values) + 1, dtype=np.int64)
    with path.open("wb") as handle:
        handle.write(INDEX_HEADER)
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", 4))
        handle.write(struct.pack("<Q", len(values)))
        handle.write(struct.pack("<Q", len(documents)))
        handle.write(values.tobytes(order="C"))
        handle.write(pointers.tobytes(order="C"))
        handle.write(documents.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class SourceShard:
    def __init__(self, prefix: Path):
        self.binary = np.memmap(Path(str(prefix) + ".bin"), mode="r", dtype=np.int32)
        idx_path = Path(str(prefix) + ".idx")
        with idx_path.open("rb") as handle:
            if handle.read(9) != INDEX_HEADER:
                raise ValueError(f"invalid index header: {idx_path}")
            if struct.unpack("<Q", handle.read(8))[0] != 1:
                raise ValueError(f"invalid index version: {idx_path}")
            if struct.unpack("<B", handle.read(1))[0] != 4:
                raise ValueError(f"invalid index dtype: {idx_path}")
            sequences = struct.unpack("<Q", handle.read(8))[0]
            handle.read(8)
            offset = handle.tell()
        raw = np.memmap(idx_path, mode="r", dtype=np.uint8)
        self._idx_raw = raw
        self.lengths = np.frombuffer(raw, dtype=np.int32, count=sequences, offset=offset)
        self.pointers = np.frombuffer(
            raw, dtype=np.int64, count=sequences, offset=offset + sequences * 4
        )

    def get(self, document_index: int, expected_tokens: int) -> np.ndarray:
        length = int(self.lengths[document_index])
        if length != expected_tokens:
            raise ValueError("catalog/source index token-count drift")
        start = int(self.pointers[document_index]) // 4
        return self.binary[start : start + length]


def output_prefix(stage: Path, value: str) -> Path:
    return (stage / "megatron" / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packing-plan", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = read_json(args.packing_plan)
    plan_schema = plan.get("schema_version")
    if plan_schema not in {
        "apertus_mini_fixed_sequence_packing_plan_v1",
        "apertus_fixed_sequence_packing_plan_v1",
    }:
        raise ValueError("unsupported packing plan")
    generic_schema = plan_schema == "apertus_fixed_sequence_packing_plan_v1"
    if not 0 <= args.task_index < len(plan["tasks"]):
        print(json.dumps({"ok": True, "skipped": True, "task_index": args.task_index}))
        return 0
    task = plan["tasks"][args.task_index]
    if int(task["task_index"]) != args.task_index:
        raise ValueError("packing task index drift")

    prefix = output_prefix(args.stage_root, task["output_prefix"])
    manifest_path = Path(str(prefix) + ".manifest.json")
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if (
            manifest.get("status") == "completed"
            and manifest.get("packing_plan_sha256") == sha256_file(args.packing_plan)
            and int(manifest.get("task_index", -1)) == args.task_index
        ):
            for receipt in manifest["outputs"].values():
                path = Path(receipt["path"])
                if not path.is_file() or path.stat().st_size != int(receipt["bytes"]):
                    raise ValueError(f"resumed packed payload drift: {path}")
            print(json.dumps({"ok": True, "resumed": True, "task_index": args.task_index}))
            return 0
        raise ValueError(f"existing packed manifest binding drift: {manifest_path}")

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    desired = min(8192, hard)
    if soft < desired:
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))

    input_receipt = read_json(args.input_receipt)
    source_tasks = {int(row["task_index"]): row for row in input_receipt["tasks"]}
    catalog_receipt = task["catalog"]
    catalog_path = Path(catalog_receipt["path"])
    if (
        not catalog_path.is_file()
        or catalog_path.stat().st_size != int(catalog_receipt["bytes"])
    ):
        raise ValueError("sorted training catalog drift")
    catalog = np.memmap(catalog_path, mode="r", dtype=CATALOG_DTYPE)[
        int(task["catalog_row_start"]) : int(task["catalog_row_end"])
    ]
    if catalog.size != int(task["selected_document_rows"]):
        raise ValueError("selected catalog row accounting drift")
    if np.any(catalog["pool"] != int(task["pool_code"])):
        raise ValueError("selected catalog contains a different pool")

    from tokenizers import Tokenizer

    tokenizer_root = Path(input_receipt["tokenizer"]["root"])
    tokenizer = Tokenizer.from_file(str(tokenizer_root / "tokenizer.json"))
    tokenizer_config = read_json(tokenizer_root / "tokenizer_config.json")
    eos_value = tokenizer_config["eos_token"]
    eos_token = eos_value.get("content") if isinstance(eos_value, dict) else eos_value
    eos_id = tokenizer.token_to_id(str(eos_token))
    if eos_id is None:
        raise ValueError("overlay tokenizer has no EOS token")
    pad_id = int(plan["geometry"]["pad_token_id"])

    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = Path(str(prefix) + ".bin")
    idx_path = Path(str(prefix) + ".idx")
    active_path = Path(str(prefix) + ".active.u16")
    for path in (bin_path, idx_path, active_path):
        if path.exists():
            raise FileExistsError(f"refusing unreceipted packed payload: {path}")
    bin_tmp = Path(str(bin_path) + ".partial")
    idx_tmp = Path(str(idx_path) + ".partial")
    active_tmp = Path(str(active_path) + ".partial")
    for path in (bin_tmp, idx_tmp, active_tmp):
        path.unlink(missing_ok=True)

    shards: dict[int, SourceShard] = {}
    target_remaining = int(task["target_active_tokens"])
    target_buffer = np.empty(4096, dtype=np.int32)
    fill = 0
    context_token = int(eos_id)
    active_counts: list[int] = []
    consumed_source_tokens = 0
    selected_documents_touched = 0
    with bin_tmp.open("wb") as output:
        for row in catalog:
            if target_remaining == 0:
                break
            source_task_index = int(row["task_index"])
            if source_task_index not in shards:
                source_task = source_tasks[source_task_index]
                shards[source_task_index] = SourceShard(
                    output_prefix(args.stage_root, source_task["output_prefix"])
                )
            values = shards[source_task_index].get(
                int(row["document_index"]), int(row["tokens"])
            )
            take_document = min(int(values.size), target_remaining)
            offset = 0
            selected_documents_touched += 1
            while offset < take_document:
                take = min(4096 - fill, take_document - offset)
                target_buffer[fill : fill + take] = values[offset : offset + take]
                fill += take
                offset += take
                consumed_source_tokens += take
                target_remaining -= take
                if fill == 4096:
                    output.write(np.asarray([context_token], dtype=np.int32).tobytes())
                    output.write(target_buffer.tobytes(order="C"))
                    context_token = int(target_buffer[-1])
                    active_counts.append(4096)
                    fill = 0
        if target_remaining != 0:
            raise ValueError("selected catalog rows did not satisfy the active-token target")
        if fill:
            target_buffer[fill:] = pad_id
            output.write(np.asarray([context_token], dtype=np.int32).tobytes())
            output.write(target_buffer.tobytes(order="C"))
            active_counts.append(fill)
        output.flush()
        os.fsync(output.fileno())
    os.replace(bin_tmp, bin_path)
    write_index(idx_tmp, [4097] * len(active_counts))
    os.replace(idx_tmp, idx_path)
    np.asarray(active_counts, dtype=np.uint16).tofile(active_tmp)
    os.replace(active_tmp, active_path)
    if sum(active_counts) != int(task["target_active_tokens"]):
        raise RuntimeError("packed active-token accounting drift")
    if bin_path.stat().st_size != len(active_counts) * 4097 * 4:
        raise RuntimeError("packed binary byte accounting drift")

    manifest = {
        "schema_version": (
            "apertus_fixed_sequence_bucket_v1"
            if generic_schema
            else "apertus_mini_fixed_sequence_bucket_v1"
        ),
        "status": "completed",
        "task_index": args.task_index,
        "pool": task["pool"],
        "pool_code": task["pool_code"],
        "bucket": task["bucket"],
        "packing_plan": str(args.packing_plan.resolve()),
        "packing_plan_sha256": sha256_file(args.packing_plan),
        "source_catalog": catalog_receipt,
        "catalog_row_start": task["catalog_row_start"],
        "catalog_row_end": task["catalog_row_end"],
        "selected_documents": task["selected_document_rows"],
        "selected_documents_touched": selected_documents_touched,
        "selected_source_tokens": task["selected_source_tokens"],
        "consumed_source_tokens": consumed_source_tokens,
        "discarded_tail_tokens_in_last_selected_document": task[
            "discarded_tail_tokens_in_last_selected_document"
        ],
        "sequence_length": 4096,
        "stored_tokens_per_sequence": 4097,
        "sequence_count": len(active_counts),
        "active_tokens": sum(active_counts),
        "padding_targets": len(active_counts) * 4096 - sum(active_counts),
        "prefix_context_token_id": int(eos_id),
        "pad_token_id": pad_id,
        "outputs": {
            "bin": {"path": str(bin_path), "bytes": bin_path.stat().st_size, "sha256": sha256_file(bin_path)},
            "idx": {"path": str(idx_path), "bytes": idx_path.stat().st_size, "sha256": sha256_file(idx_path)},
            "active_counts": {
                "path": str(active_path),
                "bytes": active_path.stat().st_size,
                "sha256": sha256_file(active_path),
                "dtype": "uint16_little_endian",
            },
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "task_index": args.task_index,
                "sequences": len(active_counts),
                "active_tokens": sum(active_counts),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

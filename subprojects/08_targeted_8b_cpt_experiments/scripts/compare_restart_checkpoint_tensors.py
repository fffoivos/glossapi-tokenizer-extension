#!/usr/bin/env python3
"""Compare logical tensor chunks in two PyTorch DCP checkpoints.

The comparison uses each checkpoint's DCP storage index, reads the matching
serialized chunk, deserializes it with PyTorch, and compares the resulting
object.  It can operate on a deterministic stratified sample or on the full
checkpoint.  Run it on a Clariden debug node, never on a login node.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Hashable

import torch
from torch.distributed.checkpoint import FileSystemReader


def index_key(index: Any) -> tuple[str, tuple[int, ...], int | None]:
    offset = tuple(int(value) for value in (index.offset or ()))
    return str(index.fqn), offset, index.index


def category(fqn: str) -> str:
    lowered = fqn.lower()
    if "optimizer" in lowered or "optim" in lowered:
        return "optimizer"
    if "rng" in lowered:
        return "rng"
    if "model" in lowered:
        return "model"
    return fqn.split(".", 1)[0]


def deterministic_sample(keys: list[Hashable], count: int) -> list[Hashable]:
    if count <= 0 or count >= len(keys):
        return keys
    if count == 1:
        return [keys[0]]
    positions = {round(index * (len(keys) - 1) / (count - 1)) for index in range(count)}
    return [keys[position] for position in sorted(positions)]


def select_keys(keys: list[Hashable], maximum: int) -> list[Hashable]:
    if maximum <= 0 or maximum >= len(keys):
        return keys
    grouped: dict[str, list[Hashable]] = defaultdict(list)
    for key in keys:
        grouped[category(key[0])].append(key)

    # Give every present category representation, then fill proportionally.
    selected: set[Hashable] = set()
    categories = sorted(grouped)
    base = max(1, maximum // max(1, len(categories)))
    for name in categories:
        selected.update(deterministic_sample(grouped[name], min(base, len(grouped[name]))))
    if len(selected) < maximum:
        remaining = [key for key in keys if key not in selected]
        selected.update(deterministic_sample(remaining, maximum - len(selected)))
    return sorted(selected)[:maximum]


def read_object(root: Path, storage: Any) -> Any:
    path = root / storage.relative_path
    with path.open("rb") as handle:
        handle.seek(int(storage.offset))
        payload = handle.read(int(storage.length))
    if len(payload) != int(storage.length):
        raise ValueError(f"short read for {path}: {len(payload)} != {storage.length}")
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def compare_values(left: Any, right: Any, path: str, differences: list[dict[str, Any]]) -> None:
    if type(left) is not type(right):
        differences.append(
            {"path": path, "kind": "type", "left": type(left).__name__, "right": type(right).__name__}
        )
        return
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            differences.append(
                {
                    "path": path,
                    "kind": "tensor_structure",
                    "left_shape": list(left.shape),
                    "right_shape": list(right.shape),
                    "left_dtype": str(left.dtype),
                    "right_dtype": str(right.dtype),
                }
            )
            return
        if not torch.equal(left, right):
            row: dict[str, Any] = {
                "path": path,
                "kind": "tensor_value",
                "shape": list(left.shape),
                "dtype": str(left.dtype),
                "different_elements": int(torch.count_nonzero(left != right).item()),
            }
            if left.is_floating_point():
                delta = (left.to(torch.float64) - right.to(torch.float64)).abs()
                row["max_abs_delta"] = float(delta.max().item())
                row["mean_abs_delta"] = float(delta.mean().item())
            differences.append(row)
        return
    if isinstance(left, dict):
        if set(left) != set(right):
            differences.append({"path": path, "kind": "dict_keys"})
        for key in sorted(set(left) & set(right), key=str):
            compare_values(left[key], right[key], f"{path}.{key}", differences)
        return
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            differences.append(
                {"path": path, "kind": "sequence_length", "left": len(left), "right": len(right)}
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_values(left_item, right_item, f"{path}[{index}]", differences)
        return
    try:
        equal = left == right or (
            isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right)
        )
    except (TypeError, ValueError):
        equal = repr(left) == repr(right)
    if not equal:
        differences.append(
            {"path": path, "kind": "value", "left": repr(left)[:300], "right": repr(right)[:300]}
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=128, help="0 compares every storage entry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-debug", action="store_true")
    args = parser.parse_args()

    if args.require_debug and os.environ.get("SLURM_JOB_PARTITION") != "debug":
        raise SystemExit("tensor comparison must run in the Clariden debug partition")

    control_metadata = FileSystemReader(args.control).read_metadata()
    resumed_metadata = FileSystemReader(args.resumed).read_metadata()
    control_storage = {index_key(index): value for index, value in control_metadata.storage_data.items()}
    resumed_storage = {index_key(index): value for index, value in resumed_metadata.storage_data.items()}
    if set(control_storage) != set(resumed_storage):
        raise SystemExit("DCP storage index sets differ")

    all_keys = sorted(control_storage)
    selected = select_keys(all_keys, args.max_entries)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    compared_bytes = 0
    equal_entries = 0
    category_counts: Counter[str] = Counter()
    category_differences: Counter[str] = Counter()

    for position, key in enumerate(selected, start=1):
        control_info = control_storage[key]
        resumed_info = resumed_storage[key]
        fqn, offset, storage_index = key
        local_differences: list[dict[str, Any]] = []
        left = read_object(args.control, control_info)
        right = read_object(args.resumed, resumed_info)
        compare_values(left, right, fqn, local_differences)
        name = category(fqn)
        category_counts[name] += 1
        compared_bytes += int(control_info.length) + int(resumed_info.length)
        if local_differences:
            category_differences[name] += 1
            rows.append(
                {
                    "fqn": fqn,
                    "offset": list(offset),
                    "storage_index": storage_index,
                    "control_storage": {
                        "path": str(control_info.relative_path),
                        "offset": int(control_info.offset),
                        "length": int(control_info.length),
                    },
                    "resumed_storage": {
                        "path": str(resumed_info.relative_path),
                        "offset": int(resumed_info.offset),
                        "length": int(resumed_info.length),
                    },
                    "differences": local_differences,
                }
            )
        else:
            equal_entries += 1
        del left, right
        gc.collect()
        if position % 100 == 0 or position == len(selected):
            print(
                f"progress={position}/{len(selected)} differences={len(rows)} "
                f"bytes={compared_bytes}",
                flush=True,
            )

    payload = {
        "schema_version": "targeted_restart_checkpoint_tensor_diagnostic_v1",
        "control": str(args.control.resolve()),
        "resumed": str(args.resumed.resolve()),
        "storage_entries_total": len(all_keys),
        "storage_entries_compared": len(selected),
        "full_checkpoint_compared": len(selected) == len(all_keys),
        "logical_bytes_read": compared_bytes,
        "equal_entries": equal_entries,
        "different_entries": len(rows),
        "category_entries_compared": dict(sorted(category_counts.items())),
        "category_entries_different": dict(sorted(category_differences.items())),
        "tensor_payloads_equal_in_scope": not rows,
        "differences": rows,
        "elapsed_seconds": time.monotonic() - started,
        "acceptance_effect": "diagnostic_only; does_not_relax_or_replace_restart_gate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

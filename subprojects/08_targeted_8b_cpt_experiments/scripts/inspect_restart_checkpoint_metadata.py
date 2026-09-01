#!/usr/bin/env python3
"""Inspect cheap restart-checkpoint evidence without loading distributed tensors.

This intentionally limits itself to the small ``common.pt`` payload and DCP
metadata.  It is suitable for a one-node Clariden debug allocation and does
not replace a tensor-level checkpoint comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader


def scalar_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    return left == right


def compare_objects(left: Any, right: Any, path: str, differences: list[dict[str, Any]]) -> None:
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
                row["max_abs_delta"] = float((left - right).abs().max().item())
            differences.append(row)
        return

    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            differences.append(
                {
                    "path": path,
                    "kind": "dict_keys",
                    "left_only": sorted(map(str, left_keys - right_keys)),
                    "right_only": sorted(map(str, right_keys - left_keys)),
                }
            )
        for key in sorted(left_keys & right_keys, key=str):
            compare_objects(left[key], right[key], f"{path}.{key}", differences)
        return

    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            differences.append(
                {"path": path, "kind": "sequence_length", "left": len(left), "right": len(right)}
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_objects(left_item, right_item, f"{path}[{index}]", differences)
        return

    if hasattr(left, "__dict__") and hasattr(right, "__dict__"):
        compare_objects(vars(left), vars(right), f"{path}.__dict__", differences)
        return

    try:
        equal = scalar_equal(left, right)
    except (TypeError, ValueError):
        equal = repr(left) == repr(right)
    if not equal:
        differences.append(
            {"path": path, "kind": "value", "left": repr(left)[:500], "right": repr(right)[:500]}
        )


def metadata_signature(checkpoint: Path) -> dict[str, Any]:
    metadata = FileSystemReader(checkpoint).read_metadata()
    state_rows: dict[str, Any] = {}
    for key, value in sorted(metadata.state_dict_metadata.items()):
        if is_dataclass(value):
            state_rows[key] = asdict(value)
        else:
            state_rows[key] = repr(value)

    storage_rows: dict[str, dict[str, Any]] = {}
    for index, value in metadata.storage_data.items():
        index_key = repr(index)
        storage_rows[index_key] = {
            "relative_path": str(value.relative_path),
            "offset": int(value.offset),
            "length": int(value.length),
        }

    return {
        "state_dict_metadata": state_rows,
        "storage_data": storage_rows,
        "planner_data": repr(metadata.planner_data),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control_common = torch.load(args.control / "common.pt", map_location="cpu", weights_only=False)
    resumed_common = torch.load(args.resumed / "common.pt", map_location="cpu", weights_only=False)
    common_differences: list[dict[str, Any]] = []
    compare_objects(control_common, resumed_common, "common", common_differences)

    control_metadata = metadata_signature(args.control)
    resumed_metadata = metadata_signature(args.resumed)
    control_state = control_metadata["state_dict_metadata"]
    resumed_state = resumed_metadata["state_dict_metadata"]
    control_storage = control_metadata["storage_data"]
    resumed_storage = resumed_metadata["storage_data"]

    payload = {
        "schema_version": "targeted_restart_checkpoint_metadata_diagnostic_v1",
        "control": str(args.control.resolve()),
        "resumed": str(args.resumed.resolve()),
        "common_payload_equal": not common_differences,
        "common_difference_count": len(common_differences),
        "common_differences": common_differences,
        "state_dict_metadata_equal": control_state == resumed_state,
        "state_dict_key_count_control": len(control_state),
        "state_dict_key_count_resumed": len(resumed_state),
        "storage_index_sets_equal": set(control_storage) == set(resumed_storage),
        "storage_index_count_control": len(control_storage),
        "storage_index_count_resumed": len(resumed_storage),
        "storage_lengths_equal": {
            key: value["length"] for key, value in control_storage.items()
        }
        == {key: value["length"] for key, value in resumed_storage.items()},
        "storage_layout_equal": control_storage == resumed_storage,
        "planner_data_equal": control_metadata["planner_data"] == resumed_metadata["planner_data"],
        "scope": "small_common_payload_and_dcp_metadata_only",
        "tensor_payloads_compared": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

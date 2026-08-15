#!/usr/bin/env python3
"""Inspect a Megatron distributed-checkpoint schema without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader


def walk(value: Any, path: str, rows: list[dict[str, Any]], depth: int = 0) -> None:
    if depth > 8:
        rows.append({"path": path, "type": type(value).__name__, "truncated": True})
        return
    if isinstance(value, dict):
        rows.append({"path": path, "type": "dict", "length": len(value)})
        for key in sorted(value, key=str):
            walk(value[key], f"{path}.{key}", rows, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        rows.append({"path": path, "type": type(value).__name__, "length": len(value)})
        for index, item in enumerate(value[:16]):
            walk(item, f"{path}[{index}]", rows, depth + 1)
        if len(value) > 16:
            rows.append({"path": f"{path}[16:]", "type": "truncated", "length": len(value) - 16})
        return
    if isinstance(value, torch.Tensor):
        rows.append({
            "path": path,
            "type": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        })
        return
    if isinstance(value, (str, int, float, bool)) or value is None:
        rows.append({"path": path, "type": type(value).__name__, "value": repr(value)[:300]})
        return
    rows.append({"path": path, "type": type(value).__name__, "repr": repr(value)[:300]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    common_path = root / "common.pt"
    common = torch.load(common_path, map_location="cpu", weights_only=False)
    common_rows: list[dict[str, Any]] = []
    walk(common, "common", common_rows)
    metadata = FileSystemReader(root).read_metadata()
    state_rows = []
    for key, value in sorted(metadata.state_dict_metadata.items()):
        row: dict[str, Any] = {"key": key, "type": type(value).__name__}
        if is_dataclass(value):
            raw = asdict(value)
            row["metadata_keys"] = sorted(raw)
            if "size" in raw:
                row["size"] = list(raw["size"])
            if "properties" in raw:
                row["properties"] = repr(raw["properties"])
        state_rows.append(row)
    storage_files: dict[str, dict[str, int]] = {}
    for value in metadata.storage_data.values():
        relative = str(value.relative_path)
        row = storage_files.setdefault(relative, {"chunks": 0, "maximum_end": 0})
        row["chunks"] += 1
        row["maximum_end"] = max(row["maximum_end"], int(value.offset) + int(value.length))
    payload = {
        "checkpoint": str(root),
        "common_bytes": common_path.stat().st_size,
        "common_schema": common_rows,
        "state_dict_key_count": len(state_rows),
        "state_dict_metadata": state_rows,
        "planner_data_type": type(metadata.planner_data).__name__,
        "planner_data_repr": repr(metadata.planner_data)[:5000],
        "storage_files": storage_files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(root),
        "common_rows": len(common_rows),
        "state_dict_keys": len(state_rows),
        "storage_files": len(storage_files),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

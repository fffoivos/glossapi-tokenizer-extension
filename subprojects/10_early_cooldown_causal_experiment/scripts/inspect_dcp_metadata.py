#!/usr/bin/env python3
"""Summarize distributed-checkpoint metadata without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader


def category(key: str) -> str:
    lowered = key.lower()
    for name in (
        "model",
        "optimizer",
        "opt_param_scheduler",
        "rng",
        "random",
        "iteration",
        "num_floating_point_operations_so_far",
    ):
        if name in lowered:
            return name
    return key.split(".", 1)[0]


def metadata_description(value: object) -> dict[str, object]:
    description: dict[str, object] = {"type": type(value).__name__}
    if hasattr(value, "size"):
        description["size"] = list(value.size)  # type: ignore[attr-defined]
    if hasattr(value, "properties"):
        properties = value.properties  # type: ignore[attr-defined]
        description["properties"] = {
            "dtype": str(properties.dtype),
            "layout": str(properties.layout),
            "requires_grad": bool(properties.requires_grad),
            "memory_format": str(properties.memory_format),
            "pin_memory": bool(properties.pin_memory),
        }
    if hasattr(value, "chunks"):
        description["chunks"] = [
            {"offsets": list(chunk.offsets), "sizes": list(chunk.sizes)}
            for chunk in value.chunks  # type: ignore[attr-defined]
        ]
    return description


def storage_description(metadata: object, key: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for index, info in metadata.storage_data.items():  # type: ignore[attr-defined]
        if index.fqn != key:
            continue
        entries.append(
            {
                "offset": list(index.offset) if index.offset is not None else None,
                "index": index.index,
                "relative_path": info.relative_path,
                "byte_offset": info.offset,
                "byte_length": info.length,
            }
        )
    return sorted(entries, key=lambda item: (str(item["relative_path"]), int(item["byte_offset"])))


def file_manifest(root: Path) -> list[dict[str, object]]:
    files = []
    for path in sorted(root.glob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append({"name": path.name, "bytes": stat.st_size})
    return files


def summarize(root: Path) -> dict[str, object]:
    metadata = FileSystemReader(root).read_metadata()
    keys = sorted(metadata.state_dict_metadata)
    counts = Counter(category(key) for key in keys)
    scalar_keys = [
        key
        for key, value in metadata.state_dict_metadata.items()
        if type(value).__name__ == "BytesStorageMetadata"
    ]
    return {
        "root": str(root.resolve()),
        "state_dict_keys": len(keys),
        "storage_entries": len(metadata.storage_data),
        "category_counts": dict(sorted(counts.items())),
        "scalar_or_bytes_keys": sorted(scalar_keys),
        "all_keys": keys,
        "logical_metadata": {
            key: metadata_description(metadata.state_dict_metadata[key]) for key in keys
        },
        "control_storage": {
            key: storage_description(metadata, key)
            for key in keys
            if key.startswith("rng_state/") or key.startswith("rerun_state_machine_state/")
        },
        "files": file_manifest(root),
    }


def compare(checkpoints: list[dict[str, object]]) -> dict[str, object] | None:
    if len(checkpoints) != 2:
        return None
    left, right = checkpoints
    left_keys = set(left["all_keys"])  # type: ignore[arg-type]
    right_keys = set(right["all_keys"])  # type: ignore[arg-type]
    left_meta = left["logical_metadata"]  # type: ignore[assignment]
    right_meta = right["logical_metadata"]  # type: ignore[assignment]
    common = sorted(left_keys & right_keys)
    differing_metadata = [key for key in common if left_meta[key] != right_meta[key]]
    left_files = {item["name"]: item["bytes"] for item in left["files"]}  # type: ignore[index]
    right_files = {item["name"]: item["bytes"] for item in right["files"]}  # type: ignore[index]
    return {
        "left_only_keys": sorted(left_keys - right_keys),
        "right_only_keys": sorted(right_keys - left_keys),
        "differing_logical_metadata_keys": differing_metadata,
        "same_file_names": set(left_files) == set(right_files),
        "different_file_sizes": {
            name: {"left": left_files.get(name), "right": right_files.get(name)}
            for name in sorted(set(left_files) | set(right_files))
            if left_files.get(name) != right_files.get(name)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"schema_version": "apertus_dcp_metadata_inspection_v2", "checkpoints": []}
    result["checkpoints"] = [summarize(root) for root in args.roots]
    result["comparison"] = compare(result["checkpoints"])
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

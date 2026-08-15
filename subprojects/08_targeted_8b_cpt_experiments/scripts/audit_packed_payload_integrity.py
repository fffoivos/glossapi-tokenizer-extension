#!/usr/bin/env python3
"""Hash every packed payload and reconcile it with the frozen manifests."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from contract_utils import file_binding, read_json, require, sha256_file, write_json_atomic


def verify_payload(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["path"])
    require(path.is_file(), f"packed payload missing: {path}")
    require(path.stat().st_size == int(row["bytes"]), f"packed payload size drift: {path}")
    observed = sha256_file(path)
    require(observed == row["sha256"], f"packed payload hash drift: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": observed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable packed-integrity receipt exists: {args.output}")
    require(args.workers >= 1, "--workers must be positive")
    packed = read_json(args.packed_receipt)
    require(packed.get("schema_version") == "apertus_packed_sequence_corpus_v1", "packed schema drift")
    require(packed.get("status") == "completed", "packed corpus is incomplete")
    manifest_rows = packed.get("packing_task_manifests", [])
    require(manifest_rows, "packed receipt has no task manifests")
    manifests: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    task_indices: list[int] = []
    for binding in manifest_rows:
        path = Path(binding["manifest_path"])
        require(path.is_file(), f"packing manifest missing: {path}")
        require(sha256_file(path) == binding["manifest_sha256"], f"packing manifest hash drift: {path}")
        manifest = read_json(path)
        require(manifest.get("schema_version") == "apertus_fixed_sequence_bucket_v1", f"packing manifest schema drift: {path}")
        require(manifest.get("status") == "completed", f"packing task incomplete: {path}")
        task_indices.append(int(manifest["task_index"]))
        manifests.append(file_binding(path))
        outputs = manifest.get("outputs", {})
        require(set(outputs) == {"bin", "idx", "active_counts"}, f"packing output set drift: {path}")
        payload_rows.extend(dict(value) for value in outputs.values())
    require(sorted(task_indices) == list(range(len(task_indices))), "packing task indices are not complete and contiguous")
    paths = [str(Path(row["path"]).resolve()) for row in payload_rows]
    require(len(paths) == len(set(paths)), "duplicate packed payload path")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        payloads = list(executor.map(verify_payload, payload_rows))
    payloads.sort(key=lambda row: row["path"])
    value = {
        "schema_version": "targeted_8b_packed_payload_integrity_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "packed_receipt": file_binding(args.packed_receipt),
        "manifest_count": len(manifests),
        "payload_count": len(payloads),
        "payload_bytes": sum(int(row["bytes"]) for row in payloads),
        "manifests": manifests,
        "payloads": payloads,
        "checks": {
            "every_manifest_hash_matches": True,
            "every_payload_size_and_sha256_matches": True,
            "task_indices_complete_and_contiguous": True,
            "payload_paths_unique": True,
        },
    }
    write_json_atomic(args.output, value)
    print(json.dumps({"ok": True, "manifests": len(manifests), "payloads": len(payloads)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

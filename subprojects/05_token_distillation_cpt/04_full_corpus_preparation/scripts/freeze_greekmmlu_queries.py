#!/usr/bin/env python3
"""Freeze a checksum-bound manifest for an already exported GreekMMLU query file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from finalization_io import sha256_file, utc_now, write_json_atomic


HEX_REVISION = re.compile(r"^[0-9a-f]{7,64}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="dascim/GreekMMLU")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-config", default="All")
    parser.add_argument("--required-split", action="append", required=True)
    parser.add_argument("--default-split", default="")
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    if not HEX_REVISION.fullmatch(args.dataset_revision):
        raise ValueError("--dataset-revision must be an immutable hexadecimal commit")
    if not args.queries_jsonl.is_file():
        raise FileNotFoundError(args.queries_jsonl)
    rows = 0
    observed: set[str] = set()
    with args.queries_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{args.queries_jsonl}:{line_number}: row is not an object")
            if str(row.get("benchmark", "greekmmlu")).casefold() != "greekmmlu":
                continue
            split = str(row.get("split") or args.default_split)
            if not split:
                raise ValueError(f"{args.queries_jsonl}:{line_number}: no split and --default-split is empty")
            observed.add(split)
            rows += 1
    required = set(args.required_split)
    if not rows or required - observed:
        raise ValueError(f"query split coverage is incomplete; required={sorted(required)}, observed={sorted(observed)}")
    payload = {
        "schema_version": "greekmmlu_query_manifest_v1",
        "created_at": utc_now(),
        "benchmark_id": "greekmmlu",
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_revision": args.dataset_revision,
        "dataset_config": args.dataset_config,
        "required_splits": sorted(required),
        "default_split": args.default_split,
        "queries_path": args.queries_jsonl.name,
        "queries_sha256": sha256_file(args.queries_jsonl),
        "query_rows": rows,
        "observed_splits": sorted(observed),
        "registry_path": str(args.registry.resolve()) if args.registry else None,
        "registry_sha256": sha256_file(args.registry) if args.registry else None,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": True, "output": str(args.output), "rows": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

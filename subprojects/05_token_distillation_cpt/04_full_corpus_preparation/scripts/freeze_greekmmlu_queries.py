#!/usr/bin/env python3
"""Verify and freeze revision-bound GreekMMLU evaluation queries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from finalization_io import sha256_file, utc_now, write_json_atomic


HEX_REVISION = re.compile(r"^[0-9a-f]{40}$")
REGISTRY_SCHEMA = "native-greek-eval-registry-v1"
QUERY_SCHEMA = "greek-mcq-decontam-query-v1"


def registry_greekmmlu(path: Path) -> tuple[dict, dict]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("--registry does not use native-greek-eval-registry-v1")
    matches = [
        row
        for row in registry.get("benchmarks", [])
        if isinstance(row, dict) and row.get("id") == "greekmmlu"
    ]
    if len(matches) != 1:
        raise ValueError("--registry must contain exactly one GreekMMLU benchmark")
    return registry, matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="dascim/GreekMMLU")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-config", default="All")
    parser.add_argument("--required-split", action="append", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--builder-summary", type=Path, required=True)
    parser.add_argument(
        "--verify-hf-revision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not HEX_REVISION.fullmatch(args.dataset_revision):
        raise ValueError("--dataset-revision must be a full 40-hex immutable commit")
    if not args.queries_jsonl.is_file():
        raise FileNotFoundError(args.queries_jsonl)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {args.output}")
    if not args.registry.is_file() or not args.builder_summary.is_file():
        raise FileNotFoundError("--registry and --builder-summary must both exist")
    _, registry_spec = registry_greekmmlu(args.registry)
    expected_registry = {
        "source_type": "hf_dataset",
        "source": args.dataset_repo_id,
        "revision": args.dataset_revision,
        "config": args.dataset_config,
    }
    for key, expected in expected_registry.items():
        if registry_spec.get(key) != expected:
            raise ValueError(
                f"--registry GreekMMLU {key} drift: {registry_spec.get(key)!r} != {expected!r}"
            )
    required = set(args.required_split)
    if not required or any(not split for split in required):
        raise ValueError("--required-split must contain non-empty unique values")
    registry_split = registry_spec.get("split")
    if not isinstance(registry_split, str) or required != {registry_split}:
        raise ValueError(
            f"--required-split must exactly match the registry split {registry_split!r}"
        )
    rows = 0
    observed: set[str] = set()
    seen_ids: set[tuple[str, str]] = set()
    with args.queries_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{args.queries_jsonl}:{line_number}: row is not an object")
            if str(row.get("benchmark", "")).casefold() != "greekmmlu":
                continue
            if row.get("schema") != QUERY_SCHEMA:
                raise ValueError(f"{args.queries_jsonl}:{line_number}: unsupported query schema")
            if row.get("dataset_repo_id") != args.dataset_repo_id:
                raise ValueError(f"{args.queries_jsonl}:{line_number}: dataset_repo_id drift")
            if row.get("dataset_revision") != args.dataset_revision:
                raise ValueError(f"{args.queries_jsonl}:{line_number}: dataset_revision drift")
            if row.get("dataset_config") != args.dataset_config:
                raise ValueError(f"{args.queries_jsonl}:{line_number}: dataset_config drift")
            split = str(row.get("split") or "")
            if not split:
                raise ValueError(f"{args.queries_jsonl}:{line_number}: split provenance is missing")
            item_id = str(row.get("example_id") or "")
            if not item_id or (split, item_id) in seen_ids:
                raise ValueError(f"{args.queries_jsonl}:{line_number}: duplicate/empty example identity")
            seen_ids.add((split, item_id))
            observed.add(split)
            rows += 1
    if not rows or required != observed:
        raise ValueError(f"query split coverage is incomplete; required={sorted(required)}, observed={sorted(observed)}")
    summary = json.loads(args.builder_summary.read_text(encoding="utf-8"))
    if summary.get("schema") != "greek-mcq-decontam-query-summary-v1":
        raise ValueError("--builder-summary schema is unsupported")
    if summary.get("registry_sha256") != sha256_file(args.registry):
        raise ValueError("--builder-summary is bound to a different registry")
    if summary.get("output_jsonl_sha256") != sha256_file(args.queries_jsonl):
        raise ValueError("--builder-summary is bound to different query bytes")
    benchmark = summary.get("benchmarks", {}).get("greekmmlu", {})
    expected_summary = {
        "source": args.dataset_repo_id,
        "config": args.dataset_config,
        "revision": args.dataset_revision,
        "split_loaded": next(iter(required)) if len(required) == 1 else None,
        "items": rows,
    }
    for key, expected in expected_summary.items():
        if expected is not None and benchmark.get(key) != expected:
            raise ValueError(f"--builder-summary GreekMMLU {key} drift: {benchmark.get(key)!r}")
    if args.verify_hf_revision:
        from huggingface_hub import HfApi

        resolved = HfApi().dataset_info(args.dataset_repo_id, revision=args.dataset_revision).sha
        if resolved != args.dataset_revision:
            raise ValueError("Hugging Face did not resolve the requested immutable revision exactly")
    payload = {
        "schema_version": "greekmmlu_query_manifest_v1",
        "created_at": utc_now(),
        "benchmark_id": "greekmmlu",
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_revision": args.dataset_revision,
        "dataset_config": args.dataset_config,
        "required_splits": sorted(required),
        "default_split": "",
        "queries_path": args.queries_jsonl.name,
        "queries_sha256": sha256_file(args.queries_jsonl),
        "query_rows": rows,
        "observed_splits": sorted(observed),
        "registry_path": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "registry_schema": REGISTRY_SCHEMA,
        "registry_benchmark_spec": registry_spec,
        "builder_summary_path": str(args.builder_summary.resolve()),
        "builder_summary_sha256": sha256_file(args.builder_summary),
        "builder_summary_schema": summary["schema"],
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": True, "output": str(args.output), "rows": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

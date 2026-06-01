#!/usr/bin/env python3
"""Validate the 04 Vanilla CPT dataset smoke artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_BUCKETS = {
    "greek": 0.70,
    "replay": 0.24,
    "code": 0.04,
    "math": 0.02,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--mix-manifest", required=True, type=Path)
    parser.add_argument("--preprocess-manifest", required=True, type=Path)
    parser.add_argument("--data-prefix", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--bucket-tolerance", type=float, default=0.035)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_token_ids(bin_path: Path) -> dict:
    data = np.memmap(bin_path, mode="r", dtype=np.int32)
    if data.size == 0:
        raise ValueError(f"empty Megatron .bin: {bin_path}")
    chunk = 16_000_000
    min_id = None
    max_id = None
    for start in range(0, data.size, chunk):
        arr = data[start : start + chunk]
        local_min = int(arr.min())
        local_max = int(arr.max())
        min_id = local_min if min_id is None else min(min_id, local_min)
        max_id = local_max if max_id is None else max(max_id, local_max)
    return {"tokens": int(data.size), "min_id": int(min_id), "max_id": int(max_id)}


def main() -> None:
    args = parse_args()
    recipe = read_json(args.recipe)
    mix = read_json(args.mix_manifest)
    preprocess = read_json(args.preprocess_manifest)

    errors: list[str] = []
    greek_sources = [
        source["name"]
        for source in recipe["sources"]
        if source.get("bucket") == "greek"
    ]
    if greek_sources != ["greek_hplt_clean60"]:
        errors.append(f"expected only greek_hplt_clean60, got {greek_sources}")

    for bucket, expected in EXPECTED_BUCKETS.items():
        observed = mix.get("per_bucket", {}).get(bucket, {}).get("effective_weight")
        if observed is None:
            errors.append(f"missing bucket {bucket} in mix manifest")
            continue
        if abs(float(observed) - expected) > args.bucket_tolerance:
            errors.append(
                f"bucket {bucket} effective weight {observed:.5f} outside "
                f"expected {expected:.5f} +/- {args.bucket_tolerance:.5f}"
            )

    bin_path = args.data_prefix.with_suffix(args.data_prefix.suffix + ".bin")
    idx_path = args.data_prefix.with_suffix(args.data_prefix.suffix + ".idx")
    if not bin_path.is_file():
        errors.append(f"missing bin file: {bin_path}")
    if not idx_path.is_file():
        errors.append(f"missing idx file: {idx_path}")

    token_scan = scan_token_ids(bin_path) if bin_path.is_file() else {}
    if token_scan:
        if token_scan["min_id"] < 0:
            errors.append(f"negative token id found: {token_scan['min_id']}")
        if token_scan["max_id"] >= args.vocab_size:
            errors.append(
                f"max token id {token_scan['max_id']} exceeds base vocab "
                f"size {args.vocab_size}"
            )

    report = {
        "ok": not errors,
        "errors": errors,
        "recipe": str(args.recipe),
        "mix_manifest": str(args.mix_manifest),
        "preprocess_manifest": str(args.preprocess_manifest),
        "data_prefix": str(args.data_prefix),
        "expected_buckets": EXPECTED_BUCKETS,
        "actual_buckets": mix.get("per_bucket", {}),
        "greek_sources": greek_sources,
        "preprocess": preprocess,
        "token_scan": token_scan,
        "document_boundary_flags_for_training": [
            "--reset-attention-mask",
            "--reset-position-ids",
            "--eod-mask-loss",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

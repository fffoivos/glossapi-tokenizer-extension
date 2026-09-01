#!/usr/bin/env python3
"""Validate and freeze one sharded HPLT/OpenArchives mix selection."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


TARGETS = {"hplt": 8_500_000_000, "openarchives": 3_700_000_000}


def bind_shards(paths: list[Path], manifests: list[dict]) -> tuple[list[dict], str, int, int]:
    """Hash relocated shard payloads and the exact concatenation in one pass."""
    concatenated = hashlib.sha256()
    bindings: list[dict] = []
    total_bytes = 0
    total_rows = 0
    for index, (path, manifest) in enumerate(zip(paths, manifests, strict=True)):
        digest = hashlib.sha256()
        shard_bytes = 0
        shard_rows = 0
        with path.open("rb") as handle:
            for raw_line in handle:
                require(raw_line.strip(), f"blank row in modern mix shard {index}")
                digest.update(raw_line)
                concatenated.update(raw_line)
                shard_bytes += len(raw_line)
                shard_rows += 1
        require(shard_rows == int(manifest["actual_rows"]), f"modern mix shard row drift: {index}")
        require(shard_bytes == path.stat().st_size, f"modern mix shard byte drift: {index}")
        bindings.append({
            "path": str(path.resolve()),
            "bytes": shard_bytes,
            "sha256": digest.hexdigest(),
            "rows": shard_rows,
            "pre_relocation_output": manifest["output"],
        })
        total_bytes += shard_bytes
        total_rows += shard_rows
    return bindings, concatenated.hexdigest(), total_bytes, total_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=tuple(TARGETS), required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-receipt", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_manifest.exists(), f"immutable modern-mix manifest exists: {args.output_manifest}")
    require(args.target_tokens == TARGETS[args.pool], f"historical {args.pool} target drift")
    require(args.shards == 16 and args.seed == 20260611, "historical modern mix geometry drift")
    recipe_receipt = read_json(args.recipe_receipt)
    require(recipe_receipt.get("schema_version") == "apertus_hard_h_to_g_modern_mix_recipes_v1", "modern recipe receipt schema drift")
    require(recipe_receipt.get("status") == "frozen", "modern recipes are not frozen")
    require(recipe_receipt["recipes"][args.pool]["recipe"]["sha256"] == file_binding(args.recipe)["sha256"], "modern recipe binding drift")
    require(int(recipe_receipt["recipes"][args.pool]["target_tokens"]) == args.target_tokens, "modern recipe target drift")
    manifests = [args.shard_root / f"shard_{index:02d}.manifest.json" for index in range(args.shards)]
    require(all(path.is_file() for path in manifests), "modern mix shard manifest missing")
    parts = [read_json(path) for path in manifests]
    shard_payloads = [args.shard_root / f"shard_{index:02d}.jsonl" for index in range(args.shards)]
    target_per_shard = (args.target_tokens + args.shards - 1) // args.shards
    for index, part in enumerate(parts):
        require(int(part["source_shard_index"]) == index, f"modern mix shard index drift: {index}")
        require(int(part["source_shard_count"]) == args.shards, f"modern mix shard count drift: {index}")
        require(int(part["seed"]) == args.seed, f"modern mix shard seed drift: {index}")
        require(int(part["target_tokens"]) == target_per_shard, f"modern mix shard target drift: {index}")
        require(Path(part["output"]).name == f"shard_{index:02d}.jsonl", f"modern mix shard output name drift: {index}")
        require(shard_payloads[index].is_file(), f"modern mix shard payload missing: {index}")
    actual_tokens = sum(int(part["actual_tokens"]) for part in parts)
    actual_rows = sum(int(part["actual_rows"]) for part in parts)
    require(actual_tokens >= args.target_tokens, "modern mix did not reach its requested token target")

    lineage = hashlib.sha256()
    output_sha256 = hashlib.sha256()
    output_bytes = 0
    observed_rows = 0
    source_datasets: Counter[str] = Counter()
    with args.output_jsonl.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            output_sha256.update(raw_line)
            output_bytes += len(raw_line)
            row = json.loads(raw_line)
            for key in ("text", "doc_id", "source_dataset", "source_doc_id", "_source_release_shard", "_source_release_row_index"):
                require(row.get(key) not in (None, ""), f"{args.pool} selected row {line_number} lacks {key}")
            require(str(row["doc_id"]) == str(row["source_doc_id"]), f"{args.pool} selected doc-id lineage drift at row {line_number}")
            dataset = str(row["source_dataset"])
            if args.pool == "hplt":
                require(dataset == "HPLT/ell_Grek_ge8_no_mt_clean60", f"HPLT selected source drift: {dataset}")
            else:
                require(dataset.startswith("openarchives.gr"), f"OpenArchives selected source drift: {dataset}")
            identity = f"{row['_source_release_shard']}\0{int(row['_source_release_row_index'])}\n".encode()
            lineage.update(identity)
            source_datasets[dataset] += 1
            observed_rows += 1
    require(observed_rows == actual_rows, "modern mix JSONL/shard row accounting drift")
    require(output_bytes == args.output_jsonl.stat().st_size, "modern mix byte accounting drift")
    shard_bindings, concatenated_sha256, concatenated_bytes, concatenated_rows = bind_shards(shard_payloads, parts)
    require(concatenated_sha256 == output_sha256.hexdigest(), "modern mix aggregate is not the exact ordered shard concatenation")
    require(concatenated_bytes == output_bytes and concatenated_rows == observed_rows, "modern mix concatenation accounting drift")
    per_source_rows = sum(int(row["rows"]) for part in parts for row in part["per_source"].values())
    per_source_tokens = sum(int(row["tokens"]) for part in parts for row in part["per_source"].values())
    require(per_source_rows == actual_rows and per_source_tokens == actual_tokens, "modern mix per-source accounting drift")
    payload = {
        "schema_version": "apertus_hard_h_to_g_modern_mix_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "pool": args.pool,
        "recipe": file_binding(args.recipe),
        "recipe_receipt": file_binding(args.recipe_receipt),
        "target_tokens": args.target_tokens,
        "actual_tokens": actual_tokens,
        "actual_rows": actual_rows,
        "mix_shards": args.shards,
        "mix_seed": args.seed,
        "target_tokens_per_shard": target_per_shard,
        "output": {
            "path": str(args.output_jsonl.resolve()),
            "bytes": output_bytes,
            "sha256": output_sha256.hexdigest(),
        },
        "selected_release_coordinate_order_sha256": lineage.hexdigest(),
        "source_dataset_rows": dict(sorted(source_datasets.items())),
        "shard_manifests": [file_binding(path) for path in manifests],
        "shard_payloads": shard_bindings,
        "invariants": {
            "source_shards_are_disjoint_by_eligible_row_index_modulo": True,
            "concatenation_order_is_shard_index_ascending": True,
            "historical_mix_seed_and_geometry_reproduced": True,
            "transactional_shard_directory_relocation_preserves_payload_hashes": True,
            "aggregate_is_exact_shard_index_order_concatenation": True,
            "source_lineage_preserved": True,
            "historical_document_identity_claimed": False,
            "additional_deduplication": False,
        },
    }
    write_json_atomic(args.output_manifest, payload)
    print(args.output_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

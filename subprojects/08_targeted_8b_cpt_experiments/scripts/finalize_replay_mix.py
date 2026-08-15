#!/usr/bin/env python3
"""Freeze the aggregate manifest for the sharded 5B-token replay selection."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


def bind_shards(paths: list[Path], manifests: list[dict]) -> tuple[list[dict], str, int, int]:
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
                require(raw_line.strip(), f"blank row in replay mix shard {index}")
                digest.update(raw_line)
                concatenated.update(raw_line)
                shard_bytes += len(raw_line)
                shard_rows += 1
        require(shard_rows == int(manifest["actual_rows"]), f"replay shard row drift: {index}")
        require(shard_bytes == path.stat().st_size, f"replay shard byte drift: {index}")
        bindings.append({
            "path": str(path.resolve()), "bytes": shard_bytes,
            "sha256": digest.hexdigest(), "rows": shard_rows,
        })
        total_bytes += shard_bytes
        total_rows += shard_rows
    return bindings, concatenated.hexdigest(), total_bytes, total_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_manifest.exists(), f"immutable replay manifest exists: {args.output_manifest}")
    inventory = read_json(args.source_inventory)
    require(inventory.get("schema_version") == "apertus_hard_h_to_g_replay_source_inventory_v1", "replay inventory schema drift")
    require(inventory.get("status") == "passed", "replay source inventory did not pass")
    require(inventory["recipe"]["sha256"] == file_binding(args.recipe)["sha256"], "replay recipe/inventory drift")
    manifests = [args.shard_root / f"shard_{index:02d}.manifest.json" for index in range(args.shards)]
    require(all(path.is_file() for path in manifests), "replay shard manifest missing")
    parts = [read_json(path) for path in manifests]
    shard_payloads = [args.shard_root / f"shard_{index:02d}.jsonl" for index in range(args.shards)]
    require(args.shards == 16, "historical replay shard count drift")
    require(args.seed == 20260611, "historical replay mix seed drift")
    target_per_shard = (args.target_tokens + args.shards - 1) // args.shards
    for index, part in enumerate(parts):
        require(int(part["source_shard_index"]) == index, f"replay shard index drift: {index}")
        require(int(part["source_shard_count"]) == args.shards, f"replay shard count drift: {index}")
        require(int(part["seed"]) == args.seed, f"replay shard seed drift: {index}")
        require(int(part["target_tokens"]) == target_per_shard, f"replay shard target drift: {index}")
        require(Path(part["output"]).resolve() == (args.shard_root / f"shard_{index:02d}.jsonl").resolve(), f"replay shard output drift: {index}")
        require(shard_payloads[index].is_file(), f"replay shard payload missing: {index}")
    actual_tokens = sum(int(part["actual_tokens"]) for part in parts)
    actual_rows = sum(int(part["actual_rows"]) for part in parts)
    require(actual_tokens >= args.target_tokens, "replay mix did not reach the requested token target")
    per_source: dict[str, Counter[str]] = {}
    per_bucket: dict[str, Counter[str]] = {}
    for part in parts:
        for name, row in part["per_source"].items():
            bucket = per_source.setdefault(name, Counter())
            bucket["rows"] += int(row["rows"])
            bucket["tokens"] += int(row["tokens"])
        for name, row in part["per_bucket"].items():
            bucket = per_bucket.setdefault(name, Counter())
            bucket["rows"] += int(row["rows"])
            bucket["tokens"] += int(row["tokens"])
    require(sum(row["rows"] for row in per_source.values()) == actual_rows, "replay per-source row accounting drift")
    require(sum(row["tokens"] for row in per_source.values()) == actual_tokens, "replay per-source token accounting drift")
    require(set(per_source) == set(inventory["sources"]), "replay selected-source inventory drift")
    output_binding = file_binding(args.output_jsonl)
    shard_bindings, concatenated_sha256, concatenated_bytes, concatenated_rows = bind_shards(shard_payloads, parts)
    require(concatenated_sha256 == output_binding["sha256"], "replay aggregate is not the exact ordered shard concatenation")
    require(concatenated_bytes == output_binding["bytes"] and concatenated_rows == actual_rows, "replay concatenation accounting drift")
    payload = {
        "schema_version": "apertus_hard_h_to_g_replay_mix_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "recipe": file_binding(args.recipe),
        "source_inventory": file_binding(args.source_inventory),
        "target_tokens": args.target_tokens,
        "actual_tokens": actual_tokens,
        "actual_rows": actual_rows,
        "mix_shards": args.shards,
        "mix_seed": args.seed,
        "target_tokens_per_shard": target_per_shard,
        "output": str(args.output_jsonl.resolve()),
        "output_binding": output_binding,
        "shard_manifests": [file_binding(path) for path in manifests],
        "shard_payloads": shard_bindings,
        "per_source": {
            name: {
                "rows": row["rows"],
                "tokens": row["tokens"],
                "effective_weight": row["tokens"] / actual_tokens,
            }
            for name, row in sorted(per_source.items())
        },
        "per_bucket": {
            name: {
                "rows": row["rows"],
                "tokens": row["tokens"],
                "effective_weight": row["tokens"] / actual_tokens,
            }
            for name, row in sorted(per_bucket.items())
        },
        "invariants": {
            "source_shards_are_disjoint_by_eligible_row_index_modulo": True,
            "concatenation_order_is_shard_index_ascending": True,
            "historical_bucket_preserving_token_fair_scheduler": True,
            "historical_16_way_modulo_sharding": args.shards == 16,
            "historical_mix_seed_reproduced": args.seed == 20260611,
            "historical_document_identity_claimed": False,
            "additional_deduplication": False,
            "aggregate_is_exact_shard_index_order_concatenation": True,
        },
    }
    write_json_atomic(args.output_manifest, payload)
    print(args.output_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

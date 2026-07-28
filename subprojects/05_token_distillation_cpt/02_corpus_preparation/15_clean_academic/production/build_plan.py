#!/usr/bin/env python3
"""Build stable row-group work units for every academic source shard."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from .contracts import PLAN_SCHEMA, atomic_write_json, unit_id, validate_contract

CHARS_PER_SECOND_PER_PROCESS = 197_860.0


def run(args: argparse.Namespace) -> dict:
    contract, contract_sha = validate_contract(args.contract)
    release_data = Path(contract["release_root"]) / "data"
    manifest_path = (
        Path(contract["release_root"]) / "manifests" / "deduplicated_manifest.json"
    )
    import json

    manifest = json.loads(manifest_path.read_text())
    manifest_files = {Path(entry["path"]).name: entry for entry in manifest["files"]}
    sources = contract["policy"]["analysis_sources"]
    units: list[dict] = []
    total_rows = 0
    for rank_text, source_policy in sorted(
        sources.items(), key=lambda item: int(item[0])
    ):
        rank = int(rank_text)
        shard = release_data / f"{rank:06d}.parquet"
        manifest_entry = manifest_files.get(shard.name)
        if manifest_entry is None:
            raise ValueError(f"{shard.name} is absent from the release manifest")
        if shard.is_symlink() or not shard.is_file():
            raise ValueError(f"missing or symlinked source shard: {shard}")
        parquet = pq.ParquetFile(shard)
        if parquet.metadata.num_rows != source_policy["expected_rows"]:
            raise ValueError(
                f"rank {rank} rows {parquet.metadata.num_rows} != "
                f"{source_policy['expected_rows']}"
            )
        batch = next(
            parquet.iter_batches(batch_size=256, columns=["text", "source_dataset"])
        )
        source_values = set(batch.column("source_dataset").to_pylist())
        if source_values != {source_policy["source_dataset"]}:
            raise ValueError(f"rank {rank} source mismatch: {source_values}")
        texts = batch.column("text").to_pylist()
        mean_chars = sum(len(text or "") for text in texts) / max(1, len(texts))
        start = 0
        rows = 0
        for row_group in range(parquet.metadata.num_row_groups):
            rows += parquet.metadata.row_group(row_group).num_rows
            if rows * mean_chars >= args.target_chars:
                end = row_group + 1
                units.append(
                    _unit(
                        rank,
                        shard,
                        source_policy,
                        manifest_entry,
                        start,
                        end,
                        rows,
                        mean_chars,
                    )
                )
                start, rows = end, 0
        if rows:
            units.append(
                _unit(
                    rank,
                    shard,
                    source_policy,
                    manifest_entry,
                    start,
                    parquet.metadata.num_row_groups,
                    rows,
                    mean_chars,
                )
            )
        total_rows += parquet.metadata.num_rows
    if total_rows != contract["policy"]["expected_analysis_rows"]:
        raise ValueError(f"analysis rows {total_rows} != expected")
    apply_units = [unit for unit in units if unit["apply"]]
    apply_rows = sum(unit["rows"] for unit in apply_units)
    if apply_rows != contract["policy"]["expected_apply_rows"]:
        raise ValueError(
            f"apply rows {apply_rows} != "
            f"{contract['policy']['expected_apply_rows']}"
        )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "run_id": contract["run_id"],
        "contract_sha256": contract_sha,
        "target_chars": args.target_chars,
        "chars_per_second_per_process": CHARS_PER_SECOND_PER_PROCESS,
        "rows": total_rows,
        "apply_rows": apply_rows,
        "apply_units": len(apply_units),
        "units": units,
    }
    output = Path(args.output or Path(contract["run_root"]) / "work_plan.json")
    atomic_write_json(output, plan)
    print(f"{output}: {len(units)} units, {total_rows} rows")
    return plan


def _unit(
    rank: int,
    shard: Path,
    policy: dict,
    manifest_entry: dict,
    start: int,
    end: int,
    rows: int,
    mean_chars: float,
) -> dict:
    estimated_chars = round(rows * mean_chars)
    return {
        "unit_id": unit_id(rank, start, end),
        "rank": rank,
        "source_dataset": policy["source_dataset"],
        "apply": policy["apply"],
        "source_path": str(shard),
        "source_bytes": shard.stat().st_size,
        "source_sha256": manifest_entry["sha256"],
        "row_group_start": start,
        "row_group_end": end,
        "rows": rows,
        "estimated_chars": estimated_chars,
        "estimated_seconds": estimated_chars / CHARS_PER_SECOND_PER_PROCESS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output")
    parser.add_argument("--target-chars", type=int, default=220_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

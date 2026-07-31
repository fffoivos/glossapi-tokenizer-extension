#!/usr/bin/env python3
"""Verify every shard, prove phase disjointness, and freeze both data blends."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from collections import defaultdict
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
PROBE_ROOT = HERE.parent
SHARED = PROBE_ROOT.parent / "05_training_dataset_bridge" / "scripts"
sys.path.insert(0, str(SHARED))

from bridge_common import (  # noqa: E402
    canonical_sha256,
    iter_index_lengths,
    read_json,
    sha256_file,
    task_output_prefix,
    utc_now,
    write_json_atomic,
)


getcontext().prec = 60


def _validate_shard(
    *, task: Mapping[str, Any], stage_root: Path, input_sha: str, heldout_sha: str
) -> dict[str, Any]:
    prefix = task_output_prefix(stage_root, task)
    manifest_path = Path(str(prefix) + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    expected = {
        "schema_version": "full_cpt_megatron_shard_v1",
        "status": "completed",
        "task_id": task["task_id"],
        "task_sha256": canonical_sha256(task),
        "input_receipt_sha256": input_sha,
        "heldout_manifest_sha256": heldout_sha,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"shard manifest drift ({name}): {manifest_path}")
    for label, suffix in {
        "bin": ".bin",
        "idx": ".idx",
        "dropped_ledger": ".dropped.jsonl",
        "retained_ledger": ".retained.jsonl",
    }.items():
        receipt = manifest.get("outputs", {}).get(label, {})
        path = Path(str(receipt.get("path", "")))
        if path.resolve() != Path(str(prefix) + suffix).resolve():
            raise ValueError(f"shard output path drift: {path}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(receipt.get("bytes", -1))
            or sha256_file(path) != receipt.get("sha256")
        ):
            raise ValueError(f"shard output payload drift: {path}")
    sequences, document_entries, tokens = iter_index_lengths(
        Path(manifest["outputs"]["idx"]["path"])
    )
    counts = manifest["counts"]
    if (sequences, document_entries, tokens) != (
        int(counts["documents"]),
        int(counts["document_index_entries"]),
        int(counts["tokens"]),
    ):
        raise ValueError(f"indexed-dataset accounting drift: {manifest_path}")
    if int(manifest["outputs"]["bin"]["bytes"]) != tokens * 4:
        raise ValueError(f"indexed-dataset byte accounting drift: {manifest_path}")
    return {**manifest, "task": dict(task), "manifest_path": str(manifest_path)}


def _prove_disjoint(shards: list[Mapping[str, Any]], database: Path) -> dict[str, Any]:
    database.parent.mkdir(parents=True, exist_ok=True)
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, phase INTEGER NOT NULL)")
    by_phase: dict[int, int] = defaultdict(int)
    try:
        for shard in sorted(shards, key=lambda row: int(row["task"]["task_index"])):
            phase = int(shard["task"]["phase_partition"]["phase"])
            ledger = Path(shard["outputs"]["retained_ledger"]["path"])
            batch: list[tuple[str, int]] = []
            with ledger.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    doc_id = str(value.get("doc_id", ""))
                    if not doc_id:
                        raise ValueError(f"empty retained doc_id: {ledger}:{line_number}")
                    batch.append((doc_id, phase))
                    if len(batch) >= 10000:
                        try:
                            connection.executemany("INSERT INTO documents VALUES (?, ?)", batch)
                        except sqlite3.IntegrityError as error:
                            raise ValueError(f"document assigned more than once: {ledger}") from error
                        by_phase[phase] += len(batch)
                        batch.clear()
                if batch:
                    try:
                        connection.executemany("INSERT INTO documents VALUES (?, ?)", batch)
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"document assigned more than once: {ledger}") from error
                    by_phase[phase] += len(batch)
            connection.commit()
        unique = int(connection.execute("SELECT count(*) FROM documents").fetchone()[0])
    finally:
        connection.close()
    observed = sum(by_phase.values())
    if unique != observed:
        raise ValueError("phase-disjoint identity accounting did not reconcile")
    return {
        "algorithm": "sqlite_primary_key_over_all_retained_doc_ids_v1",
        "database": str(database.resolve()),
        "database_sha256": sha256_file(database),
        "unique_documents": unique,
        "documents_by_phase": {str(key): value for key, value in sorted(by_phase.items())},
        "duplicate_documents": 0,
    }


def _phase_mix(recipe: Mapping[str, Any], phase: int) -> dict[str, Decimal]:
    raw = recipe["phases"][f"phase_{phase}"]["mix_exact"]
    result = {
        name: Decimal(Fraction(value).numerator) / Decimal(Fraction(value).denominator)
        for name, value in raw.items()
    }
    if abs(sum(result.values()) - Decimal(1)) > Decimal("1e-50"):
        raise ValueError(f"phase-{phase} logical mix does not sum to one")
    return result


def _blend(
    recipe: Mapping[str, Any], shards: list[Mapping[str, Any]], phase: int
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in shards
        if int(row["task"]["phase_partition"]["phase"]) == phase
        and int(row["counts"]["tokens"]) > 0
    ]
    by_pool: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_source: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        logical_pool = str(row["task"]["phase_partition"]["logical_pool"])
        by_pool[logical_pool].append(row)
        by_source[(logical_pool, str(row["source_name"]))].append(row)
    logical_mix = _phase_mix(recipe, phase)
    if set(by_pool) != set(logical_mix):
        raise ValueError(
            f"phase-{phase} built pools differ from recipe: built={sorted(by_pool)}, expected={sorted(logical_mix)}"
        )
    result: list[dict[str, Any]] = []
    for pool in sorted(logical_mix):
        if pool == "foreign_replay":
            raw_source_weights: dict[str, Decimal] = {}
            for (candidate_pool, source), rows in by_source.items():
                if candidate_pool != pool:
                    continue
                observed = {str(row.get("source_weight_within_pool")) for row in rows}
                if len(observed) != 1:
                    raise ValueError(f"foreign source weight drift: {source}")
                raw_source_weights[source] = Decimal(next(iter(observed)))
            source_total = sum(raw_source_weights.values())
            if source_total <= 0:
                raise ValueError("foreign replay source weights are empty")
            for source in sorted(raw_source_weights):
                rows = by_source[(pool, source)]
                source_tokens = sum(int(row["counts"]["tokens"]) for row in rows)
                for row in rows:
                    weight = (
                        logical_mix[pool]
                        * raw_source_weights[source]
                        / source_total
                        * Decimal(int(row["counts"]["tokens"]))
                        / Decimal(source_tokens)
                    )
                    result.append(_blend_row(row, pool, weight))
        else:
            pool_tokens = sum(int(row["counts"]["tokens"]) for row in by_pool[pool])
            for row in by_pool[pool]:
                weight = (
                    logical_mix[pool]
                    * Decimal(int(row["counts"]["tokens"]))
                    / Decimal(pool_tokens)
                )
                result.append(_blend_row(row, pool, weight))
    total = sum(Decimal(row["weight_exact"]) for row in result)
    if abs(total - Decimal(1)) > Decimal("1e-50"):
        raise ValueError(f"phase-{phase} physical blend does not sum to one: {total}")
    return sorted(result, key=lambda row: int(row["task_index"]))


def _blend_row(row: Mapping[str, Any], pool: str, weight: Decimal) -> dict[str, Any]:
    prefix = str(Path(row["outputs"]["bin"]["path"]).with_suffix(""))
    return {
        "task_index": int(row["task"]["task_index"]),
        "task_id": row["task_id"],
        "logical_pool": pool,
        "source_name": row["source_name"],
        "prefix": prefix,
        "tokens": int(row["counts"]["tokens"]),
        "documents": int(row["counts"]["documents"]),
        "weight_exact": format(weight, "f"),
        "weight_cli": format(weight, ".17g"),
    }


def _write_env(
    path: Path,
    *,
    input_receipt: Mapping[str, Any],
    phase_blends: Mapping[int, list[Mapping[str, Any]]],
    heldouts: Mapping[str, Any],
    stage_root: Path,
    input_receipt_path: Path,
    heldout_manifest_path: Path,
    recipe_path: Path,
) -> None:
    phase_paths = {
        phase: " ".join(f"{row['weight_cli']} {row['prefix']}" for row in rows)
        for phase, rows in phase_blends.items()
    }
    # bakeoff_train.sbatch receives logical set names here and adds the
    # val_/val_forget_ prefix according to NEW_GREEK_VALID_SETS. Exporting
    # physical stems would make it look for val_forget_val_* and skip every
    # receipt-built heldout.
    names = [
        str(row["name"])
        for row in sorted(
            heldouts["sets"], key=lambda item: (item["pool"], item["name"])
        )
    ]
    if len(names) != len(set(names)):
        raise ValueError("heldout logical names must be globally unique")
    lines = [
        "# Receipt-generated. Do not edit.",
        f'FULL_CPT_TOKENIZER_DIR="{input_receipt["tokenizer"]["root"]}"',
        f'PHASE1_CPT_DATA_PREFIX="{phase_paths[1]}"',
        f'PHASE2_CPT_DATA_PREFIX="{phase_paths[2]}"',
        f'VAL_DATA_DIR="{stage_root / "megatron" / "heldout"}"',
        f'EXTRA_VALID_SETS="{" ".join(names)}"',
        f'FULL_CPT_BRIDGE_MANIFEST="{stage_root / "bridge_manifest.json"}"',
        f'FULL_CPT_INPUT_RECEIPT="{input_receipt_path}"',
        f'FULL_CPT_HELDOUT_MANIFEST="{heldout_manifest_path}"',
        f'FULL_CPT_MIX_RECIPE="{recipe_path}"',
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=PROBE_ROOT / "configs" / "recipe_25b_midtraining.json")
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recipe = read_json(args.recipe.resolve())
    input_receipt = read_json(args.input_receipt.resolve())
    heldouts = read_json(args.heldout_manifest.resolve())
    input_sha = sha256_file(args.input_receipt.resolve())
    heldout_sha = sha256_file(args.heldout_manifest.resolve())
    if heldouts.get("input_receipt_sha256") != input_sha:
        raise ValueError("heldouts are bound to a different input receipt")
    stage_root = args.stage_root.resolve()
    shards = [
        _validate_shard(task=task, stage_root=stage_root, input_sha=input_sha, heldout_sha=heldout_sha)
        for task in input_receipt["tasks"]
    ]
    proof = _prove_disjoint(shards, stage_root / "validation" / "phase_document_ids.sqlite")
    phase_blends = {phase: _blend(recipe, shards, phase) for phase in (1, 2)}
    env_path = stage_root / "training_data.env"
    _write_env(
        env_path,
        input_receipt=input_receipt,
        phase_blends=phase_blends,
        heldouts=heldouts,
        stage_root=stage_root,
        input_receipt_path=args.input_receipt.resolve(),
        heldout_manifest_path=args.heldout_manifest.resolve(),
        recipe_path=args.recipe.resolve(),
    )
    payload = {
        "schema_version": "greek_cpt_two_phase_bridge_manifest_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "finalizer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "recipe": {"path": str(args.recipe.resolve()), "sha256": sha256_file(args.recipe.resolve())},
        "input_receipt": {"path": str(args.input_receipt.resolve()), "sha256": input_sha},
        "heldout_manifest": {"path": str(args.heldout_manifest.resolve()), "sha256": heldout_sha},
        "phase_disjointness": proof,
        "phases": {
            str(phase): {
                "blend": rows,
                "blend_sha256": canonical_sha256(rows),
                "documents": sum(int(row["documents"]) for row in rows),
                "unique_binary_tokens": sum(int(row["tokens"]) for row in rows),
                "weight_sum": format(sum(Decimal(row["weight_exact"]) for row in rows), "f"),
            }
            for phase, rows in phase_blends.items()
        },
        "training_data_env": {"path": str(env_path), "sha256": sha256_file(env_path), "bytes": env_path.stat().st_size},
    }
    output = stage_root / "bridge_manifest.json"
    write_json_atomic(output, payload)
    print(json.dumps({"ok": True, "output": str(output), "unique_documents": proof["unique_documents"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

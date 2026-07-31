#!/usr/bin/env python3
"""Verify every shard, prove phase disjointness, and freeze both data blends."""

from __future__ import annotations

import argparse
import json
import math
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
    HEX_SHA256,
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


def _prove_disjoint_and_measure_uniqueness(
    shards: list[Mapping[str, Any]], database: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    database.parent.mkdir(parents=True, exist_ok=True)
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            phase INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            logical_pool TEXT NOT NULL,
            source_name TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            tokens INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )
    by_phase: dict[int, int] = defaultdict(int)
    task_rows: dict[str, int] = defaultdict(int)
    task_tokens: dict[str, int] = defaultdict(int)
    try:
        for shard in sorted(shards, key=lambda row: int(row["task"]["task_index"])):
            phase = int(shard["task"]["phase_partition"]["phase"])
            task_id = str(shard["task_id"])
            logical_pool = str(shard["task"]["phase_partition"]["logical_pool"])
            source_name = str(shard["source_name"])
            ledger = Path(shard["outputs"]["retained_ledger"]["path"])
            batch: list[tuple[str, int, str, str, str, str, int]] = []
            with ledger.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    doc_id = str(value.get("doc_id", ""))
                    text_sha = str(value.get("text_sha256", ""))
                    tokens = int(value.get("tokens", 0))
                    if (
                        not doc_id.startswith("docv2:")
                        or not HEX_SHA256.fullmatch(doc_id.removeprefix("docv2:"))
                        or not HEX_SHA256.fullmatch(text_sha)
                        or tokens <= 0
                    ):
                        raise ValueError(
                            f"invalid retained identity/content receipt: {ledger}:{line_number}"
                        )
                    batch.append(
                        (
                            doc_id,
                            phase,
                            task_id,
                            logical_pool,
                            source_name,
                            text_sha,
                            tokens,
                        )
                    )
                    task_rows[task_id] += 1
                    task_tokens[task_id] += tokens
                    if len(batch) >= 10000:
                        try:
                            connection.executemany(
                                "INSERT INTO documents VALUES (?,?,?,?,?,?,?)", batch
                            )
                        except sqlite3.IntegrityError as error:
                            raise ValueError(f"document assigned more than once: {ledger}") from error
                        by_phase[phase] += len(batch)
                        batch.clear()
                if batch:
                    try:
                        connection.executemany(
                            "INSERT INTO documents VALUES (?,?,?,?,?,?,?)", batch
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"document assigned more than once: {ledger}") from error
                    by_phase[phase] += len(batch)
            if (
                task_rows[task_id] != int(shard["counts"]["documents"])
                or task_tokens[task_id] != int(shard["counts"]["tokens"])
            ):
                raise ValueError(f"retained-ledger token accounting drift: {task_id}")
            connection.commit()
        unique = int(connection.execute("SELECT count(*) FROM documents").fetchone()[0])
        connection.execute(
            "CREATE INDEX documents_content ON documents(text_sha256, tokens)"
        )
        connection.execute(
            "CREATE INDEX documents_grouping ON documents(phase, logical_pool, source_name, task_id, text_sha256)"
        )
        connection.commit()
        mismatch = connection.execute(
            """
            SELECT text_sha256 FROM documents
            GROUP BY text_sha256 HAVING COUNT(DISTINCT tokens) != 1 LIMIT 1
            """
        ).fetchone()
        if mismatch:
            raise ValueError(
                "equal text hashes produced different token counts: " + str(mismatch[0])
            )

        def grouped(columns: str) -> list[dict[str, Any]]:
            names = [value.strip() for value in columns.split(",")]
            query = f"""
                SELECT {columns}, COUNT(*), SUM(tokens),
                       COUNT(DISTINCT text_sha256), SUM(unique_tokens)
                FROM (
                    SELECT {columns}, doc_id, text_sha256, tokens,
                           CASE WHEN ROW_NUMBER() OVER (
                               PARTITION BY {columns}, text_sha256 ORDER BY doc_id
                           ) = 1 THEN tokens ELSE 0 END AS unique_tokens
                    FROM documents
                )
                GROUP BY {columns} ORDER BY {columns}
            """
            result: list[dict[str, Any]] = []
            for values in connection.execute(query):
                prefix = dict(zip(names, values[: len(names)], strict=True))
                counts = values[len(names) :]
                result.append(
                    {
                        **prefix,
                        "identity_documents": int(counts[0]),
                        "identity_tokens": int(counts[1]),
                        "unique_content_documents": int(counts[2]),
                        "unique_content_tokens": int(counts[3]),
                    }
                )
            return result

        totals = connection.execute(
            "SELECT COUNT(*), SUM(tokens), COUNT(DISTINCT text_sha256) FROM documents"
        ).fetchone()
        unique_content_tokens = connection.execute(
            "SELECT SUM(tokens) FROM (SELECT text_sha256, MIN(tokens) AS tokens FROM documents GROUP BY text_sha256)"
        ).fetchone()[0]
        uniqueness = {
            "schema_version": "greek_cpt_two_phase_uniqueness_v1",
            "identity_contract": "full-cpt-document-identity-v2",
            "global": {
                "identity_documents": int(totals[0] or 0),
                "identity_tokens": int(totals[1] or 0),
                "unique_content_documents": int(totals[2] or 0),
                "unique_content_tokens": int(unique_content_tokens or 0),
                "duplicate_document_identities": 0,
            },
            "tasks": grouped("phase, task_id, logical_pool, source_name"),
            "pools": grouped("phase, logical_pool"),
            "sources": grouped("phase, logical_pool, source_name"),
        }
    finally:
        connection.close()
    observed = sum(by_phase.values())
    if unique != observed:
        raise ValueError("phase-disjoint identity accounting did not reconcile")
    proof = {
        "algorithm": "sqlite_primary_key_over_all_retained_doc_ids_v1",
        "database": str(database.resolve()),
        "database_sha256": sha256_file(database),
        "unique_documents": unique,
        "documents_by_phase": {str(key): value for key, value in sorted(by_phase.items())},
        "duplicate_documents": 0,
    }
    return proof, uniqueness


def _capacity_report(
    recipe: Mapping[str, Any],
    phase_blends: Mapping[int, list[Mapping[str, Any]]],
    uniqueness: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    policy = recipe["capacity"]
    unique_ratio = Decimal(str(policy["minimum_unique_capacity_ratio"]))
    sample_ratio = Decimal(str(policy["physical_prefix_sample_capacity_ratio"]))
    boundary = int(policy["physical_prefix_boundary_samples"])
    seq = int(recipe["geometry"]["sequence_length"])
    gbs = int(recipe["geometry"]["global_batch_sequences"])
    task_unique = {
        (int(row["phase"]), str(row["task_id"])): int(row["unique_content_tokens"])
        for row in uniqueness["tasks"]
    }
    pool_unique = {
        (int(row["phase"]), str(row["logical_pool"])): int(row["unique_content_tokens"])
        for row in uniqueness["pools"]
    }
    source_unique = {
        (int(row["phase"]), str(row["logical_pool"]), str(row["source_name"])): int(row["unique_content_tokens"])
        for row in uniqueness["sources"]
    }
    failures: list[str] = []
    phases: dict[str, Any] = {}
    for phase in (1, 2):
        iterations = int(recipe["geometry"][f"phase_{phase}_iterations"])
        phase_samples = iterations * gbs
        rows = phase_blends[phase]

        def capacity_row(
            label: str, planned: int, unique_tokens: int, ratio: Decimal
        ) -> dict[str, Any]:
            required = math.ceil(Decimal(planned) * ratio) + boundary
            available = max(0, (unique_tokens - 1) // seq)
            passed = available >= required
            if not passed:
                failures.append(
                    f"phase-{phase}/{label}: nonrepeating samples {available} < required {required}"
                )
            return {
                "planned_samples": planned,
                "required_samples": required,
                "available_nonrepeating_samples": available,
                "unique_content_tokens": unique_tokens,
                "passed": passed,
            }

        prefixes = []
        for row in rows:
            planned = math.ceil(Decimal(phase_samples) * Decimal(str(row["weight_exact"])))
            key = (phase, str(row["task_id"]))
            prefixes.append(
                {
                    "task_id": row["task_id"],
                    "logical_pool": row["logical_pool"],
                    "source_name": row["source_name"],
                    "prefix": row["prefix"],
                    "weight_exact": row["weight_exact"],
                    **capacity_row(
                        f"prefix/{row['task_id']}", planned, task_unique[key], sample_ratio
                    ),
                }
            )
        pool_rows = []
        for pool, weight in _phase_mix(recipe, phase).items():
            planned = math.ceil(Decimal(phase_samples) * weight)
            pool_rows.append(
                {
                    "logical_pool": pool,
                    **capacity_row(
                        f"pool/{pool}", planned, pool_unique[(phase, pool)], unique_ratio
                    ),
                }
            )
        source_weights: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
        for row in rows:
            source_weights[(str(row["logical_pool"]), str(row["source_name"]))] += Decimal(
                str(row["weight_exact"])
            )
        source_rows = []
        for (pool, source), weight in sorted(source_weights.items()):
            planned = math.ceil(Decimal(phase_samples) * weight)
            source_rows.append(
                {
                    "logical_pool": pool,
                    "source_name": source,
                    "weight_exact": format(weight, "f"),
                    **capacity_row(
                        f"source/{pool}/{source}",
                        planned,
                        source_unique[(phase, pool, source)],
                        unique_ratio,
                    ),
                }
            )
        phases[str(phase)] = {
            "training_samples": phase_samples,
            "pools": pool_rows,
            "sources": source_rows,
            "physical_prefixes": prefixes,
        }
    return {
        "basis": "exact unique text SHA-256 token capacity",
        "minimum_unique_ratio": format(unique_ratio, "f"),
        "physical_prefix_ratio": format(sample_ratio, "f"),
        "boundary_samples": boundary,
        "sequence_length": seq,
        "phases": phases,
    }, failures


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
    recipe_path = args.recipe.resolve()
    input_config = input_receipt.get("config", {})
    if (
        Path(str(input_config.get("path", ""))).resolve() != recipe_path
        or input_config.get("sha256") != sha256_file(recipe_path)
    ):
        raise ValueError("finalizer recipe differs from the frozen input receipt")
    if heldouts.get("input_receipt_sha256") != input_sha:
        raise ValueError("heldouts are bound to a different input receipt")
    stage_root = args.stage_root.resolve()
    shards = [
        _validate_shard(task=task, stage_root=stage_root, input_sha=input_sha, heldout_sha=heldout_sha)
        for task in input_receipt["tasks"]
    ]
    proof, uniqueness = _prove_disjoint_and_measure_uniqueness(
        shards, stage_root / "validation" / "phase_document_ids.sqlite"
    )
    phase_blends = {phase: _blend(recipe, shards, phase) for phase in (1, 2)}
    capacity, capacity_failures = _capacity_report(recipe, phase_blends, uniqueness)
    if capacity_failures:
        failure_path = stage_root / "bridge_capacity_failure.json"
        write_json_atomic(
            failure_path,
            {
                "schema_version": "greek_cpt_two_phase_capacity_failure_v1",
                "status": "failed",
                "completed_at": utc_now(),
                "input_receipt_sha256": input_sha,
                "heldout_manifest_sha256": heldout_sha,
                "capacity": capacity,
                "failures": capacity_failures,
            },
        )
        raise ValueError("unique-capacity gate failed; see " + str(failure_path))
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
        "uniqueness": uniqueness,
        "capacity": capacity,
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
        "invariants": {
            "phase_document_identities_disjoint": True,
            "capacity_discounted_for_exact_content_duplicates": True,
            "each_pool_source_and_physical_prefix_has_1_005_plus_boundary_capacity": True,
        },
    }
    output = stage_root / "bridge_manifest.json"
    write_json_atomic(output, payload)
    print(json.dumps({"ok": True, "output": str(output), "unique_documents": proof["unique_documents"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

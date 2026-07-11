#!/usr/bin/env python3
"""Validate every binary shard and freeze the 79/20/1 Megatron blend."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Mapping

from bridge_common import (
    HEX_SHA256,
    bound_code_sha,
    canonical_sha256,
    iter_index_lengths,
    iter_jsonl,
    read_json,
    safe_name,
    sha256_file,
    task_output_prefix,
    utc_now,
    write_json_atomic,
)


getcontext().prec = 50


def _heldout_expected(
    heldouts: Mapping[str, Any], stage_root: Path
) -> list[dict[str, Any]]:
    result = []
    for row in sorted(heldouts["sets"], key=lambda item: (item["pool"], item["name"])):
        stem = (
            f"val_{row['name']}"
            if row["pool"] == "new_greek"
            else f"val_forget_{row['name']}"
        )
        prefix = stage_root / "megatron" / "heldout" / f"{stem}_ext_text_document"
        result.append(
            {
                "task_index": len(result),
                "task_id": f"heldout-{len(result):02d}-{safe_name(str(row['name']))}",
                "kind": "heldout",
                "pool": row["pool"],
                "source_name": row["selection_source_name"],
                "heldout_name": row["name"],
                "input_path": row["output"]["path"],
                "input_relative": Path(row["output"]["path"]).name,
                "input_sha256": row["output"]["sha256"],
                "input_bytes": row["output"]["bytes"],
                "input_rows": row["output"]["rows"],
                "decontaminate_greekmmlu": False,
                "identity_scope": "global",
                "identity_columns": ["doc_id"],
                "requires_heldout_exclusion": False,
                "exclusion_key": "",
                "exclusion_file": "",
                "output_prefix": str(prefix.relative_to(stage_root / "megatron")),
                "prefix": str(prefix),
            }
        )
    return result


def _validate_shard(
    manifest_path: Path,
    *,
    input_receipt_sha: str,
    heldout_manifest_sha: str,
    tokenizer_tree_sha: str,
    builder_sha: str,
    expected_task: Mapping[str, Any],
    stage_root: Path,
    expected_exclusions: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(manifest_path)
    expected = {
        "schema_version": "full_cpt_megatron_shard_v1",
        "status": "completed",
        "input_receipt_sha256": input_receipt_sha,
        "heldout_manifest_sha256": heldout_manifest_sha,
        "tokenizer_tree_sha256": tokenizer_tree_sha,
        "task_id": expected_task["task_id"],
        "task_index": int(expected_task["task_index"]),
        "kind": expected_task["kind"],
        "pool": expected_task["pool"],
        "source_name": expected_task["source_name"],
        "heldout_name": expected_task.get("heldout_name"),
        "task_sha256": canonical_sha256(
            {key: value for key, value in expected_task.items() if key != "prefix"}
        ),
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError(f"binary shard receipt drift ({field}): {manifest_path}")
    if value.get("builder", {}).get("sha256") != builder_sha:
        raise ValueError(f"binary shard builder drift: {manifest_path}")
    expected_input = {
        "path": str(Path(expected_task["input_path"]).resolve()),
        "sha256": expected_task["input_sha256"],
        "bytes": expected_task["input_bytes"],
        "rows": expected_task["input_rows"],
    }
    if value.get("input") != expected_input:
        raise ValueError(f"binary shard input identity drift: {manifest_path}")
    prefix = task_output_prefix(stage_root, expected_task)
    if Path(str(value.get("output_prefix", ""))).resolve() != prefix.resolve():
        raise ValueError(f"binary shard output-prefix drift: {manifest_path}")
    outputs = value.get("outputs", {})
    if set(outputs) != {"bin", "idx", "dropped_ledger", "retained_ledger"}:
        raise ValueError(f"binary shard output inventory drift: {manifest_path}")
    for label in ("bin", "idx", "dropped_ledger", "retained_ledger"):
        receipt = outputs.get(label, {})
        path = Path(str(receipt.get("path", "")))
        suffix = {
            "bin": ".bin",
            "idx": ".idx",
            "dropped_ledger": ".dropped.jsonl",
            "retained_ledger": ".retained.jsonl",
        }[label]
        if path.resolve() != Path(str(prefix) + suffix).resolve():
            raise ValueError(f"binary shard payload path drift: {path}")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(receipt.get("bytes", -1)) or sha256_file(
            path
        ) != receipt.get("sha256"):
            raise ValueError(f"binary shard payload drift: {path}")
    sequences, document_entries, tokens = iter_index_lengths(
        Path(outputs["idx"]["path"])
    )
    counts = value.get("counts", {})
    if (sequences, document_entries, tokens) != (
        int(counts.get("documents", -1)),
        int(counts.get("document_index_entries", -1)),
        int(counts.get("tokens", -1)),
    ):
        raise ValueError(f"binary shard index accounting drift: {manifest_path}")
    if int(outputs["bin"]["bytes"]) != tokens * 4:
        raise ValueError(f"binary shard int32 byte accounting drift: {manifest_path}")
    if int(counts.get("candidate_rows", -1)) != int(counts.get("documents", -1)) + int(
        counts.get("contaminated_rows", -1)
    ):
        raise ValueError(f"binary shard candidate accounting drift: {manifest_path}")
    if int(outputs["retained_ledger"].get("rows", -1)) != int(
        counts.get("documents", -1)
    ):
        raise ValueError(f"retained-ledger row accounting drift: {manifest_path}")
    expected_exclusion = bool(expected_task.get("requires_heldout_exclusion"))
    binding = value.get("heldout_exclusion", {})
    if bool(binding.get("required")) != expected_exclusion:
        raise ValueError(f"heldout-exclusion requirement drift: {manifest_path}")
    if expected_exclusion and binding.get("key") != expected_task.get("exclusion_key"):
        raise ValueError(f"heldout-exclusion identity drift: {manifest_path}")
    if expected_exclusion:
        receipt = expected_exclusions.get(str(expected_task["exclusion_key"]))
        expected_binding = {
            "required": True,
            "key": expected_task["exclusion_key"],
            "path": str(Path(str(receipt["path"])).resolve()) if receipt else "",
            "sha256": receipt.get("sha256") if receipt else None,
            "bytes": receipt.get("bytes") if receipt else None,
            "rows": receipt.get("rows") if receipt else None,
        }
        if not receipt or binding != expected_binding:
            raise ValueError(f"heldout-exclusion receipt drift: {manifest_path}")
    elif binding != {"required": False}:
        raise ValueError(f"unexpected heldout-exclusion binding: {manifest_path}")
    value["manifest_path"] = str(manifest_path.resolve())
    value["manifest_sha256"] = sha256_file(manifest_path)
    value["prefix"] = str(prefix)
    return value


def compute_blend(
    shards: list[dict[str, Any]], mix_numerators: Mapping[str, int], denominator: int
) -> list[dict[str, Any]]:
    """Return exact logical-pool/source weights distributed over physical shards."""

    train = [
        row
        for row in shards
        if row["kind"] == "training" and int(row["counts"]["tokens"]) > 0
    ]
    by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        by_pool[row["pool"]].append(row)
        by_source[(row["pool"], row["source_name"])].append(row)
    if set(by_pool) != set(mix_numerators):
        raise ValueError("built pools differ from the frozen top-level mix")
    source_weights: dict[tuple[str, str], Decimal] = {}
    for key, rows in by_source.items():
        pool, _ = key
        if pool == "foreign_replay":
            observed = {str(row.get("source_weight_within_pool")) for row in rows}
            if len(observed) != 1:
                raise ValueError(f"foreign source weight drift across shards: {key}")
            source_weights[key] = Decimal(next(iter(observed)))
    foreign_sum = sum(
        weight
        for (pool, _), weight in source_weights.items()
        if pool == "foreign_replay"
    )
    if "foreign_replay" in by_pool and abs(foreign_sum - Decimal(1)) > Decimal("1e-12"):
        raise ValueError(f"foreign source weights do not sum to one: {foreign_sum}")

    result: list[dict[str, Any]] = []
    for pool in sorted(by_pool):
        pool_share = Decimal(int(mix_numerators[pool])) / Decimal(denominator)
        if pool == "foreign_replay":
            for key in sorted(key for key in by_source if key[0] == pool):
                rows = by_source[key]
                source_tokens = sum(int(row["counts"]["tokens"]) for row in rows)
                if source_tokens <= 0:
                    raise ValueError(f"foreign source has no encoded tokens: {key[1]}")
                for row in sorted(rows, key=lambda item: item["task_index"]):
                    weight = (
                        pool_share
                        * source_weights[key]
                        * Decimal(int(row["counts"]["tokens"]))
                        / Decimal(source_tokens)
                    )
                    result.append(_blend_row(row, weight))
        else:
            pool_tokens = sum(int(row["counts"]["tokens"]) for row in by_pool[pool])
            if pool_tokens <= 0:
                raise ValueError(f"pool has no encoded tokens: {pool}")
            for row in sorted(by_pool[pool], key=lambda item: item["task_index"]):
                weight = (
                    pool_share
                    * Decimal(int(row["counts"]["tokens"]))
                    / Decimal(pool_tokens)
                )
                result.append(_blend_row(row, weight))
    total = sum(Decimal(row["weight_exact"]) for row in result)
    if abs(total - Decimal(1)) > Decimal("1e-12"):
        raise ValueError(f"physical shard blend does not sum to one: {total}")
    return result


def _blend_row(row: Mapping[str, Any], weight: Decimal) -> dict[str, Any]:
    prefix = str(Path(row["outputs"]["bin"]["path"]).with_suffix(""))
    return {
        "pool": row["pool"],
        "source_name": row["source_name"],
        "task_id": row["task_id"],
        "prefix": prefix,
        "tokens": int(row["counts"]["tokens"]),
        "documents": int(row["counts"]["documents"]),
        "weight_exact": format(weight, "f"),
        "weight_cli": format(weight, ".17g"),
    }


def audit_unique_training_documents(
    shards: list[dict[str, Any]], database_path: Path
) -> dict[str, Any]:
    """Prove global document identity and exact-content capacity on disk."""

    database_path.unlink(missing_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            text_sha256 TEXT NOT NULL,
            tokens INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            pool TEXT NOT NULL,
            source_name TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    task_rows: dict[str, int] = defaultdict(int)
    task_tokens: dict[str, int] = defaultdict(int)
    try:
        for shard in sorted(shards, key=lambda row: int(row["task_index"])):
            if shard["kind"] != "training":
                continue
            ledger = shard["outputs"]["retained_ledger"]
            batch: list[tuple[str, str, int, str, str, str]] = []
            for row in iter_jsonl(Path(ledger["path"])):
                doc_id = str(row.get("doc_id", ""))
                text_sha = str(row.get("text_sha256", ""))
                tokens = int(row.get("tokens", 0))
                if not doc_id.startswith("docv2:") or not HEX_SHA256.fullmatch(
                    doc_id.removeprefix("docv2:")
                ):
                    raise ValueError(f"invalid retained document identity: {doc_id!r}")
                if not HEX_SHA256.fullmatch(text_sha) or tokens <= 0:
                    raise ValueError(f"invalid retained content receipt: {doc_id}")
                batch.append(
                    (
                        doc_id,
                        text_sha,
                        tokens,
                        str(shard["task_id"]),
                        str(shard["pool"]),
                        str(shard["source_name"]),
                    )
                )
                task_rows[str(shard["task_id"])] += 1
                task_tokens[str(shard["task_id"])] += tokens
                if len(batch) >= 10_000:
                    try:
                        connection.executemany(
                            "INSERT INTO documents VALUES (?,?,?,?,?,?)", batch
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            "duplicate composite document identity across training shards"
                        ) from exc
                    batch.clear()
            if batch:
                try:
                    connection.executemany(
                        "INSERT INTO documents VALUES (?,?,?,?,?,?)", batch
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "duplicate composite document identity across training shards"
                    ) from exc
            if task_rows[str(shard["task_id"])] != int(
                shard["counts"]["documents"]
            ) or task_tokens[str(shard["task_id"])] != int(shard["counts"]["tokens"]):
                raise ValueError(
                    f"retained-ledger accounting drift: {shard['task_id']}"
                )
        connection.commit()
        connection.execute(
            "CREATE INDEX documents_content ON documents(text_sha256, tokens)"
        )
        connection.execute(
            "CREATE INDEX documents_grouping ON documents(pool, source_name, task_id, text_sha256)"
        )
        connection.commit()
        mismatch = connection.execute(
            """
            SELECT text_sha256
            FROM documents
            GROUP BY text_sha256
            HAVING COUNT(DISTINCT tokens) != 1
            LIMIT 1
            """
        ).fetchone()
        if mismatch:
            raise ValueError(
                "equal text hashes produced different token counts: " + str(mismatch[0])
            )

        def grouped(columns: str) -> list[dict[str, Any]]:
            names = [value.strip() for value in columns.split(",")]
            query = f"""
                SELECT {columns},
                       COUNT(*) AS identity_documents,
                       SUM(tokens) AS identity_tokens,
                       COUNT(DISTINCT text_sha256) AS unique_content_documents,
                       SUM(unique_tokens) AS unique_content_tokens
                FROM (
                    SELECT {columns}, doc_id, text_sha256, tokens,
                           CASE WHEN ROW_NUMBER() OVER (
                               PARTITION BY {columns}, text_sha256 ORDER BY doc_id
                           ) = 1 THEN tokens ELSE 0 END AS unique_tokens
                    FROM documents
                )
                GROUP BY {columns}
                ORDER BY {columns}
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
            """
            SELECT COUNT(*), SUM(tokens), COUNT(DISTINCT text_sha256)
            FROM documents
            """
        ).fetchone()
        unique_content_tokens = connection.execute(
            "SELECT SUM(tokens) FROM (SELECT text_sha256, MIN(tokens) AS tokens FROM documents GROUP BY text_sha256)"
        ).fetchone()[0]
        return {
            "schema_version": "full_cpt_training_uniqueness_v1",
            "identity_contract": "full-cpt-document-identity-v2",
            "global": {
                "identity_documents": int(totals[0] or 0),
                "identity_tokens": int(totals[1] or 0),
                "unique_content_documents": int(totals[2] or 0),
                "unique_content_tokens": int(unique_content_tokens or 0),
                "duplicate_document_identities": 0,
            },
            "tasks": grouped("task_id, pool, source_name"),
            "pools": grouped("pool"),
            "sources": grouped("pool, source_name"),
        }
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)


def capacity_report(
    shards: list[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    uniqueness: Mapping[str, Any] | None = None,
    blend: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    probe = config["probe"]
    effective_tokens = int(
        probe.get("effective_training_tokens", probe["nominal_tokens"])
    )
    effective_samples = int(probe.get("effective_training_samples", 0))
    if blend is not None and effective_samples <= 0:
        effective_samples = effective_tokens // int(probe["sequence_length"])
    denominator = int(probe["mix_denominator"])
    ratio = Decimal(str(probe["minimum_unique_capacity_ratio"]))
    numerators = probe["mix_numerators"]
    by_pool: dict[str, int] = defaultdict(int)
    by_source: dict[tuple[str, str], int] = defaultdict(int)
    by_task: dict[str, int] = defaultdict(int)
    source_weights: dict[str, Decimal] = {}
    for row in shards:
        if row["kind"] != "training":
            continue
        tokens = int(row["counts"]["tokens"])
        by_pool[row["pool"]] += tokens
        by_source[(row["pool"], row["source_name"])] += tokens
        by_task[row["task_id"]] += tokens
        if row["pool"] == "foreign_replay":
            source_weights[row["source_name"]] = Decimal(
                str(row["source_weight_within_pool"])
            )
    if uniqueness is not None:
        by_pool = defaultdict(
            int,
            {
                str(row["pool"]): int(row["unique_content_tokens"])
                for row in uniqueness["pools"]
            },
        )
        by_source = defaultdict(
            int,
            {
                (str(row["pool"]), str(row["source_name"])): int(
                    row["unique_content_tokens"]
                )
                for row in uniqueness["sources"]
            },
        )
        by_task = defaultdict(
            int,
            {
                str(row["task_id"]): int(row["unique_content_tokens"])
                for row in uniqueness["tasks"]
            },
        )
    failures: list[str] = []
    pools: dict[str, Any] = {}
    for pool, numerator in numerators.items():
        planned = (effective_tokens * int(numerator)) // denominator
        required = math.ceil(Decimal(planned) * ratio)
        available = by_pool.get(pool, 0)
        passed = available >= required
        if not passed:
            failures.append(f"{pool}: unique tokens {available} < required {required}")
        pools[pool] = {
            "planned_sampling_tokens": planned,
            "minimum_unique_tokens": required,
            "available_unique_tokens": available,
            "passed": passed,
        }
    foreign_planned = pools.get("foreign_replay", {}).get("planned_sampling_tokens", 0)
    sources: dict[str, Any] = {}
    for source, weight in sorted(source_weights.items()):
        required = math.ceil(Decimal(foreign_planned) * weight * ratio)
        available = by_source[("foreign_replay", source)]
        passed = available >= required
        if not passed:
            failures.append(
                f"foreign/{source}: unique tokens {available} < required {required}"
            )
        sources[source] = {
            "weight_within_foreign": format(weight, "f"),
            "minimum_unique_tokens": required,
            "available_unique_tokens": available,
            "passed": passed,
        }
    prefix_rows: list[dict[str, Any]] = []
    if blend is not None:
        sequence_length = int(probe["sequence_length"])
        sample_ratio = Decimal(str(probe["physical_prefix_sample_capacity_ratio"]))
        boundary = int(probe["physical_prefix_boundary_samples"])
        for row in blend:
            planned_samples = math.ceil(
                Decimal(effective_samples) * Decimal(str(row["weight_exact"]))
            )
            required_samples = (
                math.ceil(Decimal(planned_samples) * sample_ratio) + boundary
            )
            unique_tokens = by_task[str(row["task_id"])]
            available_samples = max(0, (unique_tokens - 1) // sequence_length)
            passed = available_samples >= required_samples
            if not passed:
                failures.append(
                    f"prefix/{row['task_id']}: unique sample capacity "
                    f"{available_samples} < required {required_samples}"
                )
            prefix_rows.append(
                {
                    "task_id": row["task_id"],
                    "pool": row["pool"],
                    "source_name": row["source_name"],
                    "prefix": row["prefix"],
                    "weight_exact": row["weight_exact"],
                    "unique_content_tokens": unique_tokens,
                    "available_nonrepeating_samples": available_samples,
                    "planned_samples": planned_samples,
                    "minimum_ratio": format(sample_ratio, "f"),
                    "boundary_samples": boundary,
                    "required_samples": required_samples,
                    "passed": passed,
                }
            )
    return {
        "minimum_unique_capacity_ratio": format(ratio, "f"),
        "effective_training_tokens": effective_tokens,
        "effective_training_samples": effective_samples,
        "capacity_basis": (
            "exact unique text SHA-256 capacity"
            if uniqueness is not None
            else "encoded tokens"
        ),
        "pools": pools,
        "foreign_sources": sources,
        "physical_prefixes": prefix_rows,
    }, failures


def _write_text_atomic(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") == text:
            return
        raise ValueError(f"existing generated text artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_idempotent(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if read_json(path) == dict(value):
            return
        raise ValueError(f"existing generated JSON artifact differs: {path}")
    write_json_atomic(path, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable bridge manifest: {args.output}"
        )
    input_receipt = read_json(args.input_receipt)
    heldouts = read_json(args.heldout_manifest)
    config = read_json(args.config)
    if (
        input_receipt.get("schema_version")
        != "full_cpt_training_bridge_input_receipt_v1"
        or input_receipt.get("status") != "completed"
    ):
        raise ValueError("training-bridge input receipt is not completed")
    if (
        heldouts.get("schema_version") != "full_cpt_training_heldouts_v1"
        or heldouts.get("status") != "completed"
    ):
        raise ValueError("heldout manifest is not completed")
    if config.get("schema_version") != "full_cpt_training_bridge_config_v1":
        raise ValueError("unsupported training-bridge config")
    input_sha = sha256_file(args.input_receipt)
    heldout_sha = sha256_file(args.heldout_manifest)
    config_sha = sha256_file(args.config)
    finalizer_sha = bound_code_sha(input_receipt, Path(__file__))
    bound_code_sha(input_receipt, Path(__file__).with_name("bridge_common.py"))
    if input_receipt.get("config", {}).get("sha256") != config_sha:
        raise ValueError("final config differs from the frozen input receipt")
    if (
        heldouts.get("input_receipt_sha256") != input_sha
        or heldouts.get("config_sha256") != config_sha
    ):
        raise ValueError("heldouts differ from the frozen bridge inputs")
    heldout_builder_receipt = heldouts.get("builder", {})
    heldout_builder_path = Path(str(heldout_builder_receipt.get("path", "")))
    if not heldout_builder_path.is_file() or sha256_file(
        heldout_builder_path
    ) != heldout_builder_receipt.get("sha256"):
        raise ValueError("heldout builder differs from its completed manifest")
    bound_code_sha(input_receipt, heldout_builder_path)
    builder_receipts = {
        Path(row["path"]).name: row["sha256"]
        for row in input_receipt["repository"]["code_files"]
    }
    builder_sha = builder_receipts.get("build_binary_shard.py")
    if not builder_sha:
        raise ValueError("input receipt does not bind the binary builder")
    binary_builder_path = Path(__file__).with_name("build_binary_shard.py")
    if sha256_file(binary_builder_path) != builder_sha:
        raise ValueError("binary builder differs from the frozen input receipt")

    expected_training: dict[Path, Mapping[str, Any]] = {}
    for task in input_receipt["tasks"]:
        prefix = task_output_prefix(args.stage_root.resolve(), task)
        expected_training[Path(str(prefix) + ".manifest.json").resolve()] = task
    expected_heldout = _heldout_expected(heldouts, args.stage_root.resolve())
    expected_heldout_paths = {
        Path(row["prefix"] + ".manifest.json").resolve(): row
        for row in expected_heldout
    }
    actual_training = set(
        (args.stage_root / "megatron" / "train").rglob("*.manifest.json")
    )
    actual_heldout = set(
        (args.stage_root / "megatron" / "heldout").rglob("*.manifest.json")
    )
    if {path.resolve() for path in actual_training} != set(expected_training):
        raise ValueError(
            "training binary manifest inventory is incomplete or has unexpected files"
        )
    if {path.resolve() for path in actual_heldout} != set(expected_heldout_paths):
        raise ValueError(
            "heldout binary manifest inventory is incomplete or has unexpected files"
        )

    shards: list[dict[str, Any]] = []
    for path, expected_task in sorted(
        [*expected_training.items(), *expected_heldout_paths.items()],
        key=lambda item: str(item[0]),
    ):
        shards.append(
            _validate_shard(
                path,
                input_receipt_sha=input_sha,
                heldout_manifest_sha=heldout_sha,
                tokenizer_tree_sha=input_receipt["tokenizer"]["tree_sha256"],
                builder_sha=builder_sha,
                expected_task=expected_task,
                stage_root=args.stage_root.resolve(),
                expected_exclusions=heldouts["exclusions"],
            )
        )
    training_shards = [row for row in shards if row["kind"] == "training"]
    heldout_shards = [row for row in shards if row["kind"] == "heldout"]
    excluded_rows: dict[str, int] = defaultdict(int)
    for row in training_shards:
        binding = row["heldout_exclusion"]
        if binding["required"]:
            excluded_rows[str(binding["key"])] += int(row["counts"]["heldout_rows"])
        elif int(row["counts"]["heldout_rows"]) != 0:
            raise ValueError(
                f"unbound heldout exclusions were applied: {row['task_id']}"
            )
    expected_excluded_rows = {
        str(key): int(receipt["rows"])
        for key, receipt in heldouts["exclusions"].items()
    }
    if dict(excluded_rows) != expected_excluded_rows:
        raise ValueError(
            "heldout exclusions were not applied exactly once across training tasks: "
            f"observed={dict(excluded_rows)}, expected={expected_excluded_rows}"
        )
    exclusion_audit = {
        "status": "passed",
        "expected_and_excluded_rows_by_key": expected_excluded_rows,
        "all_exclusions_applied_exactly_once": True,
    }
    blend = compute_blend(
        training_shards,
        config["probe"]["mix_numerators"],
        int(config["probe"]["mix_denominator"]),
    )
    uniqueness = audit_unique_training_documents(
        training_shards, args.stage_root / ".capacity_identity.partial.sqlite3"
    )
    capacity, failures = capacity_report(
        training_shards, config, uniqueness=uniqueness, blend=blend
    )
    if failures:
        failure_path = args.output.with_name("bridge_capacity_failure.json")
        write_json_atomic(
            failure_path,
            {
                "schema_version": "full_cpt_training_bridge_capacity_failure_v1",
                "status": "failed",
                "completed_at": utc_now(),
                "input_receipt_sha256": input_sha,
                "heldout_manifest_sha256": heldout_sha,
                "capacity": capacity,
                "failures": failures,
            },
        )
        raise ValueError("unique-token capacity gate failed; see " + str(failure_path))
    mix_recipe_path = args.stage_root / "training_mix_79_20_1.json"
    mix_recipe = {
        "schema_version": "full_cpt_megatron_mix_v1",
        "recipe_id": config["recipe_id"],
        "application": "Megatron --data-path weights",
        "physical_materialization": "each eligible source document encoded once; no source duplication",
        "nominal_probe_tokens": config["probe"]["nominal_tokens"],
        "logical_mix_numerators": config["probe"]["mix_numerators"],
        "logical_mix_denominator": config["probe"]["mix_denominator"],
        "entries": blend,
        "entry_weight_sum": format(
            sum(Decimal(row["weight_exact"]) for row in blend), "f"
        ),
    }
    _write_json_idempotent(mix_recipe_path, mix_recipe)
    data_path = " ".join(f"{row['weight_cli']} {row['prefix']}" for row in blend)
    env_path = args.stage_root / "training_data.env"
    heldout_names = " ".join(
        row["heldout_name"]
        for row in sorted(heldout_shards, key=lambda item: item["heldout_name"])
    )
    env_text = "\n".join(
        [
            "# Generated by finalize_bridge.py; do not edit.",
            f'FULL_CPT_DATA_PREFIX="{data_path}"',
            f'VAL_DATA_DIR="{(args.stage_root / "megatron" / "heldout").resolve()}"',
            f'EXTRA_VALID_SETS="{heldout_names}"',
            f'FULL_CPT_TOKENIZER_DIR="{input_receipt["tokenizer"]["root"]}"',
            f'FULL_CPT_INPUT_RECEIPT="{args.input_receipt.resolve()}"',
            f'FULL_CPT_HELDOUT_MANIFEST="{args.heldout_manifest.resolve()}"',
            f'FULL_CPT_MIX_RECIPE="{mix_recipe_path.resolve()}"',
            f'FULL_CPT_BRIDGE_MANIFEST="{args.output.resolve()}"',
            "",
        ]
    )
    _write_text_atomic(env_path, env_text)
    pool_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"documents": 0, "tokens": 0, "contaminated": 0}
    )
    for row in training_shards:
        bucket = pool_counts[row["pool"]]
        bucket["documents"] += int(row["counts"]["documents"])
        bucket["tokens"] += int(row["counts"]["tokens"])
        bucket["contaminated"] += int(row["counts"]["contaminated_rows"])
    output_receipts = [
        {
            "task_id": row["task_id"],
            "kind": row["kind"],
            "pool": row["pool"],
            "source_name": row["source_name"],
            "manifest": row["manifest_path"],
            "manifest_sha256": row["manifest_sha256"],
            "documents": row["counts"]["documents"],
            "tokens": row["counts"]["tokens"],
            "bin": row["outputs"]["bin"],
            "idx": row["outputs"]["idx"],
            "retained_ledger": row["outputs"]["retained_ledger"],
            "task_sha256": row["task_sha256"],
        }
        for row in shards
    ]
    payload = {
        "schema_version": "full_cpt_training_bridge_manifest_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "recipe_id": config["recipe_id"],
        "input_receipt": {
            "path": str(args.input_receipt.resolve()),
            "sha256": input_sha,
        },
        "phase04_release_manifest": input_receipt["phase04"]["release_manifest"],
        "phase04_validation": input_receipt["phase04"]["validation"],
        "heldout_manifest": {
            "path": str(args.heldout_manifest.resolve()),
            "sha256": heldout_sha,
        },
        "config": {"path": str(args.config.resolve()), "sha256": config_sha},
        "tokenizer": input_receipt["tokenizer"],
        "builder": {
            "repository_commit": input_receipt["repository"]["commit"],
            "code_sha256": input_receipt["repository"]["code_sha256"],
            "megatron_commit": input_receipt["megatron"]["commit"],
            "megatron_tree_sha256": input_receipt["megatron"]["tree"]["tree_sha256"],
            "finalizer_sha256": finalizer_sha,
        },
        "mix_recipe": {
            "path": str(mix_recipe_path.resolve()),
            "sha256": sha256_file(mix_recipe_path),
        },
        "training_env": {
            "path": str(env_path.resolve()),
            "sha256": sha256_file(env_path),
        },
        "capacity": capacity,
        "uniqueness": uniqueness,
        "heldout_exclusion_audit": exclusion_audit,
        "pool_counts": dict(pool_counts),
        "training_shards": len(training_shards),
        "heldout_shards": len(heldout_shards),
        "outputs": output_receipts,
        "outputs_sha256": canonical_sha256(output_receipts),
        "invariants": {
            "phase04_validation_passed": True,
            "source_inventories_hash_bound": True,
            "tokenizer_revision_and_bytes_bound": True,
            "exact_binary_token_and_document_counts": True,
            "unique_capacity_sufficient_for_nominal_probe": True,
            "global_composite_document_identities_unique": True,
            "capacity_discounted_for_exact_content_duplicates": True,
            "heldout_exclusions_applied_exactly_once": True,
            "each_physical_prefix_has_1_005_plus_boundary_sample_capacity": True,
            "top_level_mix_applied_at_sampling": True,
            "source_documents_duplicated_during_build": False,
        },
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {"ok": True, "manifest": str(args.output), "pools": dict(pool_counts)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

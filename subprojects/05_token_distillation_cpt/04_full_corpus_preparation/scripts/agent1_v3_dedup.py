#!/usr/bin/env python3
"""Adapters and representative reconciliation for Agent 1's ordered v3 dedup.

The established repository deduper remains the near-duplicate detector.  This
module prepares its immutable pre-decontamination input and re-resolves its
content clusters with the v3 representative precedence: Nanochat base,
license, extraction completeness, quality, provenance, then stable identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from finalization_io import atomic_output_path, discover_parquet, parquet_file_receipt, sha256_file, sha256_text, utc_now, write_json_atomic


POOL_MANIFEST = "agent1_full_corpus_v3_admitted_pool_manifest_v1"
LEDGER_MANIFEST = "agent1_full_corpus_v3_dedup_ledger_manifest_v1"
MATERIALIZED_MANIFEST = "agent1_full_corpus_v3_dedup_materialization_manifest_v1"
LEDGER_SCHEMA = "agent1_full_corpus_v3_dedup_decision_ledger_v1"
ADMITTED = {"include", "include_after_cleaning", "low_weight"}
LICENSE_RANK = {"eligible_open": 0, "policy_review": 1, "noncommercial_review": 2, "per_item_review": 3}


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def source_decisions(path: Path) -> dict[str, str]:
    value = read_object(path)
    rows = value.get("sources")
    if value.get("schema_version") != "agent1_full_corpus_v3_source_admission_confirmation_v1" or value.get("status") != "approved":
        raise ValueError(f"{path}: approved v3 admission confirmation required")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: sources must be a list")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: source decision must be an object")
        source = str(row.get("source_id") or "")
        decision = str(row.get("decision") or "")
        if not source or source in result:
            raise ValueError(f"{path}: invalid/duplicate source {source!r}")
        result[source] = decision
    return result


def ensure_new_roots(*paths: Path) -> None:
    for path in paths:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing to write into non-empty root: {path}")
        path.mkdir(parents=True, exist_ok=True)


def extended_schema(schema: Any) -> Any:
    import pyarrow as pa

    extras = [
        ("input_representation_id", pa.string()),
        ("representation_id", pa.string()),
        ("parent_representation_id", pa.string()),
        ("parent_text_sha256", pa.string()),
        ("text_sha256", pa.string()),
        ("cleaned_text_sha256", pa.string()),
        ("source_admission_decision", pa.string()),
        ("sampling_weight", pa.float64()),
        ("eligible_for_training", pa.bool_()),
        ("eligible_for_redistribution", pa.bool_()),
        ("dedup_source_rank", pa.int64()),
        ("dedup_license_rank", pa.int64()),
        ("dedup_extraction_rank", pa.int64()),
        ("dedup_quality_rank", pa.float64()),
        ("dedup_provenance_key", pa.string()),
    ]
    names = set(schema.names)
    collision = names.intersection(name for name, _ in extras)
    if collision:
        raise ValueError(f"normalized input already has v3 pool fields: {sorted(collision)}")
    return pa.schema([*schema, *extras], metadata=schema.metadata)


def candidate_decision(row: dict[str, Any], decisions: dict[str, str]) -> str:
    if str(row.get("source_role") or "") == "base" or str(row.get("acquisition_source_id") or "") == "nanochat_base":
        return "include"
    return decisions.get(str(row.get("acquisition_source_id") or ""), decisions.get(str(row.get("source_dataset") or ""), "exclude"))


def rank_fields(row: dict[str, Any]) -> tuple[int, int, int, float, str]:
    base = str(row.get("source_role") or "") == "base" or str(row.get("acquisition_source_id") or "") == "nanochat_base"
    source_rank = 0 if base else 1
    license_rank = LICENSE_RANK.get(str(row.get("training_eligibility") or ""), 4)
    needs_ocr = row.get("needs_ocr")
    ocr_success = row.get("ocr_success")
    extraction_rank = 0 if needs_ocr is False or ocr_success is True else 1 if needs_ocr is None else 2
    try:
        quality_rank = float(row.get("greek_badness_score") or 0.0) + float(row.get("mojibake_badness_score") or 0.0)
    except (TypeError, ValueError):
        quality_rank = float("inf")
    provenance = "\x00".join((str(row.get("source_repo_id") or ""), str(row.get("source_revision") or "")))
    return source_rank, license_rank, extraction_rank, quality_rank, provenance


def pool_row(row: dict[str, Any], decisions: dict[str, str]) -> dict[str, Any] | None:
    decision = candidate_decision(row, decisions)
    if decision not in ADMITTED:
        return None
    text = str(row.get("text") or "")
    text_hash = sha256_text(text)
    claimed = row.get("normalized_text_sha256")
    if claimed is not None and str(claimed) != text_hash:
        raise ValueError(f"{row.get('stable_uid')}: normalized_text_sha256 drift")
    stable_uid = str(row.get("stable_uid") or "")
    if not stable_uid:
        raise ValueError("normalized row lacks stable_uid")
    source_rank, license_rank, extraction_rank, quality_rank, provenance = rank_fields(row)
    representation = f"normalized-v1:{stable_uid}:{text_hash}"
    return {
        **row,
        "input_representation_id": representation,
        "representation_id": representation,
        "parent_representation_id": None,
        "parent_text_sha256": text_hash,
        "text_sha256": text_hash,
        "cleaned_text_sha256": text_hash,
        "source_admission_decision": decision,
        "sampling_weight": 0.25 if decision == "low_weight" else 1.0,
        "eligible_for_training": True,
        "eligible_for_redistribution": decision == "include",
        "dedup_source_rank": source_rank,
        "dedup_license_rank": license_rank,
        "dedup_extraction_rank": extraction_rank,
        "dedup_quality_rank": quality_rank,
        "dedup_provenance_key": provenance,
    }


def cmd_prepare_pool(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.manifest.exists():
        raise FileExistsError(f"immutable manifest already exists: {args.manifest}")
    decisions = source_decisions(args.admission_confirmation)
    ensure_new_roots(args.output)
    counts: Counter[str] = Counter()
    receipts: list[dict[str, Any]] = []
    for input_path in discover_parquet(args.input):
        relative = input_path.relative_to(args.input)
        output_path = args.output / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parquet = pq.ParquetFile(input_path)
        schema = extended_schema(parquet.schema_arrow)
        temporary = atomic_output_path(output_path)
        writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        try:
            for batch in parquet.iter_batches(batch_size=args.batch_rows, use_threads=False):
                rows = [candidate for row in batch.to_pylist() if (candidate := pool_row(row, decisions)) is not None]
                counts["input_rows"] += batch.num_rows
                counts["admitted_rows"] += len(rows)
                counts["not_admitted_rows"] += batch.num_rows - len(rows)
                for row in rows:
                    counts[f"admission:{row['source_admission_decision']}"] += 1
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            writer.close()
            os.replace(temporary, output_path)
        except BaseException:
            writer.close()
            temporary.unlink(missing_ok=True)
            raise
        receipts.append(parquet_file_receipt(output_path, relative_to=args.output))
    payload = {
        "schema_version": POOL_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "input": str(args.input.resolve()),
        "admission_confirmation": {"path": str(args.admission_confirmation.resolve()), "sha256": sha256_file(args.admission_confirmation)},
        "counts": dict(counts),
        "files": receipts,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": dict(counts)}, sort_keys=True))


def ledger_schema() -> Any:
    import pyarrow as pa

    return pa.schema([
        ("stable_uid", pa.string()),
        ("input_representation_id", pa.string()),
        ("input_text_sha256", pa.string()),
        ("action", pa.string()),
        ("representative_stable_uid", pa.string()),
        ("representative_input_representation_id", pa.string()),
        ("cluster_id", pa.string()),
        ("method", pa.string()),
        ("raw_decision_stage", pa.string()),
        ("reason", pa.string()),
    ])


def sqlite_connect(path: Path) -> sqlite3.Connection:
    if path.exists():
        raise FileExistsError(f"immutable dedup reconciliation database exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript("""
        CREATE TABLE pool (
          stable_uid TEXT PRIMARY KEY,
          input_representation_id TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          source_rank INTEGER NOT NULL,
          license_rank INTEGER NOT NULL,
          extraction_rank INTEGER NOT NULL,
          quality_rank REAL NOT NULL,
          provenance_key TEXT NOT NULL
        );
        CREATE TABLE decisions (
          doc_key TEXT PRIMARY KEY,
          stable_uid TEXT NOT NULL UNIQUE,
          parent_doc_key TEXT NOT NULL,
          raw_cluster_id TEXT,
          raw_stage TEXT,
          raw_reason TEXT,
          root_doc_key TEXT
        );
        CREATE INDEX idx_decisions_parent ON decisions(parent_doc_key);
        CREATE INDEX idx_decisions_root ON decisions(root_doc_key);
    """)
    return connection


def load_pool(connection: sqlite3.Connection, root: Path, batch_rows: int) -> int:
    import pyarrow.parquet as pq

    count = 0
    for path in discover_parquet(root):
        required = {
            "stable_uid", "input_representation_id", "cleaned_text_sha256", "dedup_source_rank",
            "dedup_license_rank", "dedup_extraction_rank", "dedup_quality_rank", "dedup_provenance_key",
        }
        parquet = pq.ParquetFile(path)
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: missing v3 pool columns {sorted(missing)}")
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=batch_rows, use_threads=False):
            rows = batch.to_pylist()
            connection.executemany(
                "INSERT INTO pool VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        str(row["stable_uid"]), str(row["input_representation_id"]), str(row["cleaned_text_sha256"]),
                        int(row["dedup_source_rank"]), int(row["dedup_license_rank"]), int(row["dedup_extraction_rank"]),
                        float(row["dedup_quality_rank"]), str(row["dedup_provenance_key"]),
                    )
                    for row in rows
                ],
            )
            count += len(rows)
    connection.commit()
    return count


def decision_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return discover_parquet(path)


def load_raw_decisions(connection: sqlite3.Connection, path: Path, batch_rows: int) -> int:
    import pyarrow.parquet as pq

    count = 0
    for file in decision_files(path):
        parquet = pq.ParquetFile(file)
        required = {"doc_key", "source_doc_id", "decision", "kept_doc_key"}
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{file}: missing raw dedup columns {sorted(missing)}")
        columns = sorted(required | {name for name in ("cluster_id", "decision_stage", "reason") if name in parquet.schema_arrow.names})
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_rows, use_threads=False):
            records = []
            for row in batch.to_pylist():
                doc_key = str(row["doc_key"])
                parent = str(row.get("kept_doc_key") or doc_key)
                if str(row.get("decision")) == "keep":
                    parent = doc_key
                records.append((doc_key, str(row["source_doc_id"]), parent, row.get("cluster_id"), row.get("decision_stage"), row.get("reason")))
            connection.executemany("INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, NULL)", records)
            count += len(records)
    connection.commit()
    return count


def resolve_roots(connection: sqlite3.Connection) -> None:
    missing = connection.execute(
        "SELECT count(*) FROM decisions d LEFT JOIN decisions p ON p.doc_key = d.parent_doc_key WHERE p.doc_key IS NULL"
    ).fetchone()[0]
    if int(missing):
        raise ValueError(f"raw dedup decisions reference {missing} missing representative keys")
    connection.execute("UPDATE decisions SET root_doc_key = parent_doc_key")
    for _ in range(64):
        before = connection.total_changes
        connection.execute(
            """UPDATE decisions
               SET root_doc_key = (SELECT parent_doc_key FROM decisions parent WHERE parent.doc_key = decisions.root_doc_key)
               WHERE root_doc_key != (SELECT parent_doc_key FROM decisions parent WHERE parent.doc_key = decisions.root_doc_key)"""
        )
        if connection.total_changes == before:
            break
    else:
        raise ValueError("raw dedup representative chains did not converge")
    cycles = connection.execute("SELECT count(*) FROM decisions WHERE root_doc_key != parent_doc_key AND root_doc_key = doc_key").fetchone()[0]
    if int(cycles):
        raise ValueError("raw dedup representative graph contains a cycle")
    connection.commit()


def write_ledger(connection: sqlite3.Connection, output: Path) -> tuple[int, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(f"immutable v3 dedup ledger exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = atomic_output_path(output)
    query = """
      WITH ranked AS (
        SELECT d.doc_key, d.stable_uid, d.root_doc_key, d.raw_cluster_id, d.raw_stage, d.raw_reason,
               p.input_representation_id, p.input_text_sha256,
               ROW_NUMBER() OVER (
                 PARTITION BY d.root_doc_key
                 ORDER BY p.source_rank, p.license_rank, p.extraction_rank, p.quality_rank, p.provenance_key, p.stable_uid
               ) AS priority,
               FIRST_VALUE(d.stable_uid) OVER (
                 PARTITION BY d.root_doc_key
                 ORDER BY p.source_rank, p.license_rank, p.extraction_rank, p.quality_rank, p.provenance_key, p.stable_uid
               ) AS representative_stable_uid,
               FIRST_VALUE(p.input_representation_id) OVER (
                 PARTITION BY d.root_doc_key
                 ORDER BY p.source_rank, p.license_rank, p.extraction_rank, p.quality_rank, p.provenance_key, p.stable_uid
               ) AS representative_input_representation_id
        FROM decisions d JOIN pool p ON p.stable_uid = d.stable_uid
      )
      SELECT * FROM ranked ORDER BY stable_uid
    """
    cursor = connection.execute(query)
    writer = pq.ParquetWriter(temporary, ledger_schema(), compression="zstd")
    rows = keeps = 0
    try:
        while True:
            batch = cursor.fetchmany(4096)
            if not batch:
                break
            output_rows = []
            for row in batch:
                keep = int(row[8]) == 1
                keeps += int(keep)
                output_rows.append({
                    "stable_uid": str(row[1]),
                    "input_representation_id": str(row[6]),
                    "input_text_sha256": str(row[7]),
                    "action": "keep" if keep else "drop",
                    "representative_stable_uid": str(row[9]),
                    "representative_input_representation_id": str(row[10]),
                    "cluster_id": "v3-component:" + str(row[2]),
                    "method": "content_work_representation_near_precedence_v1",
                    "raw_decision_stage": str(row[4] or ""),
                    "reason": "nanochat_license_extraction_quality_provenance_stable_id" if not keep else "representative",
                })
            writer.write_table(pa.Table.from_pylist(output_rows, schema=ledger_schema()))
            rows += len(output_rows)
        writer.close()
        os.replace(temporary, output)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return rows, keeps


def cmd_reconcile(args: argparse.Namespace) -> None:
    if args.manifest.exists() or args.output_ledger.exists():
        raise FileExistsError("immutable v3 dedup output already exists")
    connection = sqlite_connect(args.work_database)
    try:
        pool_rows = load_pool(connection, args.pool, args.batch_rows)
        decision_rows = load_raw_decisions(connection, args.raw_decisions, args.batch_rows)
        missing_pool = connection.execute("SELECT count(*) FROM decisions d LEFT JOIN pool p ON p.stable_uid = d.stable_uid WHERE p.stable_uid IS NULL").fetchone()[0]
        missing_decisions = connection.execute("SELECT count(*) FROM pool p LEFT JOIN decisions d ON d.stable_uid = p.stable_uid WHERE d.stable_uid IS NULL").fetchone()[0]
        if int(missing_pool) or int(missing_decisions):
            raise ValueError(f"dedup decision coverage mismatch: decisions_without_pool={missing_pool}, pool_without_decision={missing_decisions}")
        resolve_roots(connection)
        ledger_rows, kept_rows = write_ledger(connection, args.output_ledger)
    finally:
        connection.close()
    payload = {
        "schema_version": LEDGER_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "pool": str(args.pool.resolve()),
        "raw_decisions": {"path": str(args.raw_decisions.resolve()), "sha256": sha256_file(args.raw_decisions) if args.raw_decisions.is_file() else None},
        "ledger": parquet_file_receipt(args.output_ledger),
        "counts": {"pool_rows": pool_rows, "raw_decision_rows": decision_rows, "ledger_rows": ledger_rows, "kept_rows": kept_rows, "dropped_rows": ledger_rows - kept_rows},
        "representative_precedence": ["nanochat_base", "license", "extraction_completeness", "quality", "provenance", "stable_id"],
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": payload["counts"]}, sort_keys=True))


def cmd_materialize(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.manifest.exists():
        raise FileExistsError(f"immutable manifest already exists: {args.manifest}")
    ensure_new_roots(args.output)
    database = sqlite_connect(args.work_database)
    try:
        database.execute("CREATE TABLE keepers (stable_uid TEXT PRIMARY KEY, input_text_sha256 TEXT NOT NULL)")
        for file in decision_files(args.ledger):
            parquet = pq.ParquetFile(file)
            for batch in parquet.iter_batches(columns=["stable_uid", "input_text_sha256", "action"], batch_size=args.batch_rows, use_threads=False):
                database.executemany("INSERT INTO keepers VALUES (?, ?)", [(str(row["stable_uid"]), str(row["input_text_sha256"])) for row in batch.to_pylist() if row["action"] == "keep"])
        database.commit()
        counts: Counter[str] = Counter()
        receipts: list[dict[str, Any]] = []
        for input_path in discover_parquet(args.pool):
            relative = input_path.relative_to(args.pool)
            output_path = args.output / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            parquet = pq.ParquetFile(input_path)
            temporary = atomic_output_path(output_path)
            writer = pq.ParquetWriter(temporary, parquet.schema_arrow, compression="zstd")
            try:
                for batch in parquet.iter_batches(batch_size=args.batch_rows, use_threads=False):
                    rows = batch.to_pylist()
                    ids = [str(row["stable_uid"]) for row in rows]
                    placeholders = ",".join("?" for _ in ids)
                    keeper_hashes = dict(database.execute(f"SELECT stable_uid, input_text_sha256 FROM keepers WHERE stable_uid IN ({placeholders})", ids)) if ids else {}
                    kept = []
                    for row in rows:
                        uid = str(row["stable_uid"])
                        if uid in keeper_hashes:
                            if keeper_hashes[uid] != str(row["cleaned_text_sha256"]):
                                raise ValueError(f"{uid}: dedup ledger content binding drift")
                            kept.append(row)
                    if kept:
                        writer.write_table(pa.Table.from_pylist(kept, schema=parquet.schema_arrow))
                    counts["input_rows"] += len(rows)
                    counts["kept_rows"] += len(kept)
                writer.close()
                os.replace(temporary, output_path)
            except BaseException:
                writer.close()
                temporary.unlink(missing_ok=True)
                raise
            receipts.append(parquet_file_receipt(output_path, relative_to=args.output))
    finally:
        database.close()
    payload = {
        "schema_version": MATERIALIZED_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "ledger": parquet_file_receipt(args.ledger),
        "counts": dict(counts),
        "files": receipts,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": dict(counts)}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pool = sub.add_parser("prepare-pool")
    pool.add_argument("--input", type=Path, required=True)
    pool.add_argument("--admission-confirmation", type=Path, required=True)
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--manifest", type=Path, required=True)
    pool.add_argument("--batch-rows", type=int, default=2048)
    pool.set_defaults(func=cmd_prepare_pool)
    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("--pool", type=Path, required=True)
    reconcile.add_argument("--raw-decisions", type=Path, required=True)
    reconcile.add_argument("--output-ledger", type=Path, required=True)
    reconcile.add_argument("--work-database", type=Path, required=True)
    reconcile.add_argument("--manifest", type=Path, required=True)
    reconcile.add_argument("--batch-rows", type=int, default=4096)
    reconcile.set_defaults(func=cmd_reconcile)
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--pool", type=Path, required=True)
    materialize.add_argument("--ledger", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--work-database", type=Path, required=True)
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--batch-rows", type=int, default=4096)
    materialize.set_defaults(func=cmd_materialize)
    return parser


def main() -> int:
    args = parser().parse_args()
    if args.batch_rows < 1:
        raise ValueError("--batch-rows must be positive")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

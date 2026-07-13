#!/usr/bin/env python3
"""Ordered, receipt-bound v3 deduplication adapters for Agent 1.

The established repository deduper remains the near-duplicate detector.  This
module makes its stages explicit and deterministic: canonical exact-content,
work, and representation identity is resolved first; then the detector runs
within each source, across additive candidates, and finally between candidates
and the Nanochat base.  Every pass has a complete decision ledger, and a final
composition closes representative chains without needing an in-memory corpus
graph.
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
IDENTITY_RECONCILIATION_SCHEMA = "agent1_full_corpus_v3_identity_reconciliation_v1"
SCOPE_MANIFEST = "agent1_full_corpus_v3_dedup_scope_manifest_v1"
PARTITION_MANIFEST = "agent1_full_corpus_v3_dedup_within_source_partition_manifest_v1"
ORDERED_COMPOSITION_SCHEMA = "agent1_full_corpus_v3_ordered_dedup_composition_v1"
ADMITTED = {"include", "include_after_cleaning", "low_weight"}
LICENSE_RANK = {"eligible_open": 0, "policy_review": 1, "noncommercial_review": 2, "per_item_review": 3}
EXACT_IDENTITY_EDGE_KINDS = (
    "exact_content",
    "exact_work",
    "exact_representation",
)
IDENTITY_EDGE_KINDS = (
    "generic_near_component",
    *EXACT_IDENTITY_EDGE_KINDS,
)
ORDERED_PASS_LABELS = (
    "exact_content_work_representation",
    "within_source_near",
    "cross_candidate_near",
    "candidate_to_nanochat_near",
)
NEAR_PASS_KINDS = {
    "within-source": "within_source_near",
    "cross-candidate": "cross_candidate_near",
    "candidate-to-nanochat": "candidate_to_nanochat_near",
}


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
          acquisition_source_id TEXT NOT NULL,
          source_role TEXT NOT NULL,
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


def identity_sqlite_connect(path: Path) -> sqlite3.Connection:
    """Create the SQLite state used by the streamed identity closure pass.

    ``sqlite_connect`` retains the original reconciliation tables for the
    existing ``reconcile`` command.  Keeping identity state separate prevents
    an accidental mix of raw detector identifiers and canonical stable UIDs.
    """

    connection = sqlite_connect(path)
    connection.executescript("""
        CREATE TABLE identity_pool (
          stable_uid TEXT PRIMARY KEY,
          input_representation_id TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          source_rank INTEGER NOT NULL,
          license_rank INTEGER NOT NULL,
          extraction_rank INTEGER NOT NULL,
          quality_rank REAL NOT NULL,
          provenance_key TEXT NOT NULL,
          work_key TEXT NOT NULL,
          representation_generation TEXT NOT NULL,
          component_id TEXT NOT NULL
        );
        CREATE TABLE provisional_ledger (
          stable_uid TEXT PRIMARY KEY,
          input_representation_id TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          action TEXT NOT NULL,
          representative_stable_uid TEXT NOT NULL,
          representative_input_representation_id TEXT NOT NULL,
          cluster_id TEXT NOT NULL,
          raw_stage TEXT,
          raw_reason TEXT
        );
        CREATE INDEX provisional_ledger_cluster_idx ON provisional_ledger(cluster_id);
        CREATE TABLE identity_membership (
          edge_kind TEXT NOT NULL,
          key_one TEXT NOT NULL,
          key_two TEXT NOT NULL,
          stable_uid TEXT NOT NULL,
          PRIMARY KEY(edge_kind, key_one, key_two, stable_uid)
        ) WITHOUT ROWID;
        CREATE INDEX identity_membership_uid_idx ON identity_membership(stable_uid);
    """)
    return connection


def load_pool(connection: sqlite3.Connection, root: Path, batch_rows: int) -> int:
    import pyarrow.parquet as pq

    count = 0
    for path in discover_parquet(root):
        required = {
            "stable_uid", "input_representation_id", "cleaned_text_sha256", "dedup_source_rank",
            "dedup_license_rank", "dedup_extraction_rank", "dedup_quality_rank", "dedup_provenance_key",
            "acquisition_source_id", "source_role",
        }
        parquet = pq.ParquetFile(path)
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: missing v3 pool columns {sorted(missing)}")
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=batch_rows, use_threads=False):
            rows = batch.to_pylist()
            connection.executemany(
                "INSERT INTO pool VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        str(row["stable_uid"]), str(row["input_representation_id"]), str(row["cleaned_text_sha256"]),
                        nonempty_string(row.get("acquisition_source_id"), label="pool acquisition_source_id"),
                        nonempty_string(row.get("source_role"), label="pool source_role"),
                        int(row["dedup_source_rank"]), int(row["dedup_license_rank"]), int(row["dedup_extraction_rank"]),
                        float(row["dedup_quality_rank"]), str(row["dedup_provenance_key"]),
                    )
                    for row in rows
                ],
            )
            count += len(rows)
    connection.commit()
    return count


def nonempty_string(value: Any, *, label: str) -> str:
    rendered = str(value or "")
    if not rendered:
        raise ValueError(f"{label} must be a non-empty string")
    return rendered


def load_identity_pool(connection: sqlite3.Connection, root: Path, batch_rows: int) -> int:
    """Stream the admitted pool and index its canonical identity fields.

    A missing work or representation identity is unsafe here.  The canonical
    normalizer is responsible for producing both, so fail closed rather than
    treating a missing field as a singleton and silently weakening closure.
    """

    import pyarrow.parquet as pq

    required = {
        "stable_uid", "input_representation_id", "cleaned_text_sha256",
        "dedup_source_rank", "dedup_license_rank", "dedup_extraction_rank",
        "dedup_quality_rank", "dedup_provenance_key", "work_key",
        "representation_generation",
    }
    count = 0
    for path in discover_parquet(root):
        parquet = pq.ParquetFile(path)
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: missing identity-reconciliation pool columns {sorted(missing)}")
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=batch_rows, use_threads=False):
            pool_records: list[tuple[Any, ...]] = []
            membership_records: list[tuple[str, str, str, str]] = []
            for row in batch.to_pylist():
                stable_uid = nonempty_string(row.get("stable_uid"), label="pool stable_uid")
                input_representation_id = nonempty_string(
                    row.get("input_representation_id"), label=f"{stable_uid}: input_representation_id"
                )
                input_text_sha256 = nonempty_string(
                    row.get("cleaned_text_sha256"), label=f"{stable_uid}: cleaned_text_sha256"
                )
                work_key = nonempty_string(row.get("work_key"), label=f"{stable_uid}: work_key")
                representation_generation = nonempty_string(
                    row.get("representation_generation"),
                    label=f"{stable_uid}: representation_generation",
                )
                try:
                    pool_records.append((
                        stable_uid,
                        input_representation_id,
                        input_text_sha256,
                        int(row["dedup_source_rank"]),
                        int(row["dedup_license_rank"]),
                        int(row["dedup_extraction_rank"]),
                        float(row["dedup_quality_rank"]),
                        str(row["dedup_provenance_key"]),
                        work_key,
                        representation_generation,
                        stable_uid,
                    ))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{stable_uid}: invalid v3 representative rank") from exc
                # The representation key is intentionally recorded even though
                # it is a subset of work identity.  Its explicit edge count is
                # an audit signal for multiple views of one canonical work.
                membership_records.extend((
                    ("exact_content", input_text_sha256, "", stable_uid),
                    ("exact_work", work_key, "", stable_uid),
                    ("exact_representation", work_key, representation_generation, stable_uid),
                ))
            try:
                connection.executemany(
                    "INSERT INTO identity_pool VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    pool_records,
                )
                connection.executemany(
                    "INSERT INTO identity_membership VALUES (?, ?, ?, ?)",
                    membership_records,
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("identity pool has duplicate stable_uid or identity membership") from exc
            count += len(pool_records)
        connection.commit()
    if not count:
        raise ValueError("identity reconciliation pool is empty")
    return count


def load_provisional_ledger(connection: sqlite3.Connection, path: Path, batch_rows: int) -> int:
    """Load and validate the generic v3 ledger before adding its components.

    The provisional ledger is a complete decision ledger, not a keep-list.  A
    coverage check below is deliberately strict so a document omitted by a
    detector/adapter failure cannot be mistaken for an intentional drop.
    """

    import pyarrow.parquet as pq

    required = {
        "stable_uid", "input_representation_id", "input_text_sha256", "action",
        "representative_stable_uid", "representative_input_representation_id",
        "cluster_id", "raw_decision_stage", "reason",
    }
    count = 0
    for file in decision_files(path):
        parquet = pq.ParquetFile(file)
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{file}: missing provisional v3 ledger columns {sorted(missing)}")
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=batch_rows, use_threads=False):
            records: list[tuple[Any, ...]] = []
            for row in batch.to_pylist():
                stable_uid = nonempty_string(row.get("stable_uid"), label="provisional ledger stable_uid")
                action = nonempty_string(row.get("action"), label=f"{stable_uid}: action")
                if action not in {"keep", "drop"}:
                    raise ValueError(f"{stable_uid}: provisional ledger action must be keep or drop")
                records.append((
                    stable_uid,
                    nonempty_string(row.get("input_representation_id"), label=f"{stable_uid}: input_representation_id"),
                    nonempty_string(row.get("input_text_sha256"), label=f"{stable_uid}: input_text_sha256"),
                    action,
                    nonempty_string(row.get("representative_stable_uid"), label=f"{stable_uid}: representative_stable_uid"),
                    nonempty_string(
                        row.get("representative_input_representation_id"),
                        label=f"{stable_uid}: representative_input_representation_id",
                    ),
                    nonempty_string(row.get("cluster_id"), label=f"{stable_uid}: cluster_id"),
                    row.get("raw_decision_stage"),
                    row.get("reason"),
                ))
            try:
                connection.executemany("INSERT INTO provisional_ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
            except sqlite3.IntegrityError as exc:
                raise ValueError("provisional v3 ledger has duplicate stable_uid") from exc
            count += len(records)
        connection.commit()
    if not count:
        raise ValueError("provisional v3 ledger is empty")
    return count


def validate_provisional_ledger_coverage(connection: sqlite3.Connection) -> dict[str, int]:
    """Reject incomplete, extra, or internally inconsistent generic ledgers."""

    missing = int(connection.execute(
        "SELECT count(*) FROM identity_pool p LEFT JOIN provisional_ledger l USING(stable_uid) "
        "WHERE l.stable_uid IS NULL"
    ).fetchone()[0])
    extra = int(connection.execute(
        "SELECT count(*) FROM provisional_ledger l LEFT JOIN identity_pool p USING(stable_uid) "
        "WHERE p.stable_uid IS NULL"
    ).fetchone()[0])
    content_drift = int(connection.execute(
        "SELECT count(*) FROM provisional_ledger l JOIN identity_pool p USING(stable_uid) "
        "WHERE l.input_representation_id != p.input_representation_id "
        "OR l.input_text_sha256 != p.input_text_sha256"
    ).fetchone()[0])
    missing_representative = int(connection.execute(
        "SELECT count(*) FROM provisional_ledger l "
        "LEFT JOIN identity_pool p ON p.stable_uid = l.representative_stable_uid "
        "WHERE p.stable_uid IS NULL"
    ).fetchone()[0])
    representative_drift = int(connection.execute(
        "SELECT count(*) FROM provisional_ledger l "
        "JOIN identity_pool p ON p.stable_uid = l.representative_stable_uid "
        "WHERE l.representative_input_representation_id != p.input_representation_id"
    ).fetchone()[0])
    invalid_clusters = int(connection.execute("""
        SELECT count(*) FROM (
          SELECT cluster_id
          FROM provisional_ledger
          GROUP BY cluster_id
          HAVING COUNT(DISTINCT representative_stable_uid) != 1
             OR COUNT(DISTINCT representative_input_representation_id) != 1
             OR SUM(CASE WHEN action = 'keep' THEN 1 ELSE 0 END) != 1
             OR SUM(CASE WHEN stable_uid = representative_stable_uid AND action = 'keep' THEN 1 ELSE 0 END) != 1
        )
    """).fetchone()[0])
    result = {
        "pool_rows_without_provisional_ledger": missing,
        "provisional_ledger_rows_outside_pool": extra,
        "content_binding_drift": content_drift,
        "missing_declared_representative": missing_representative,
        "representative_binding_drift": representative_drift,
        "invalid_generic_components": invalid_clusters,
    }
    if any(result.values()):
        raise ValueError(
            "provisional v3 ledger must be a complete, content-bound generic decision ledger: "
            + ", ".join(f"{key}={value}" for key, value in result.items())
        )
    connection.execute(
        "INSERT INTO identity_membership "
        "SELECT 'generic_near_component', cluster_id, '', stable_uid FROM provisional_ledger"
    )
    connection.commit()
    return result


def identity_group_counts(
    connection: sqlite3.Connection,
    *,
    required_edge_kinds: Iterable[str] = IDENTITY_EDGE_KINDS,
) -> dict[str, dict[str, int]]:
    rows = connection.execute("""
        SELECT edge_kind,
               COUNT(*) AS group_count,
               SUM(member_count) AS membership_rows,
               SUM(CASE WHEN member_count > 1 THEN 1 ELSE 0 END) AS linked_group_count,
               SUM(CASE WHEN member_count > 1 THEN member_count - 1 ELSE 0 END) AS link_count
        FROM (
          SELECT edge_kind, key_one, key_two, COUNT(*) AS member_count
          FROM identity_membership
          GROUP BY edge_kind, key_one, key_two
        )
        GROUP BY edge_kind
        ORDER BY edge_kind
    """).fetchall()
    result = {
        str(kind): {
            "groups": int(groups),
            "membership_rows": int(members),
            "linked_groups": int(linked or 0),
            "links": int(links or 0),
        }
        for kind, groups, members, linked, links in rows
    }
    missing = set(required_edge_kinds) - set(result)
    if missing:
        raise ValueError(f"identity reconciliation is missing edge kinds: {sorted(missing)}")
    return result


def resolve_identity_components(connection: sqlite3.Connection) -> dict[str, int]:
    """Compute the full transitive closure of generic and exact identities.

    Each iteration propagates the lexicographically smallest stable UID across
    every equality group.  There is intentionally no arbitrary iteration cap:
    labels strictly decrease over a finite set, and completion is verified
    afterwards.  This remains disk-backed and bounded in memory even for the
    full corpus.
    """

    connection.executescript("""
        CREATE TEMP TABLE identity_group_min (
          edge_kind TEXT NOT NULL,
          key_one TEXT NOT NULL,
          key_two TEXT NOT NULL,
          component_id TEXT NOT NULL,
          PRIMARY KEY(edge_kind, key_one, key_two)
        ) WITHOUT ROWID;
        CREATE TEMP TABLE identity_component_update (
          stable_uid TEXT PRIMARY KEY,
          component_id TEXT NOT NULL
        ) WITHOUT ROWID;
    """)
    iterations = 0
    updates = 0
    while True:
        connection.execute("DELETE FROM identity_group_min")
        connection.execute("DELETE FROM identity_component_update")
        connection.execute("""
            INSERT INTO identity_group_min
            SELECT m.edge_kind, m.key_one, m.key_two, MIN(p.component_id)
            FROM identity_membership m
            JOIN identity_pool p USING(stable_uid)
            GROUP BY m.edge_kind, m.key_one, m.key_two
        """)
        connection.execute("""
            INSERT INTO identity_component_update
            SELECT m.stable_uid, MIN(g.component_id)
            FROM identity_membership m
            JOIN identity_group_min g
              ON g.edge_kind = m.edge_kind
             AND g.key_one = m.key_one
             AND g.key_two = m.key_two
            GROUP BY m.stable_uid
        """)
        connection.execute("""
            UPDATE identity_pool
            SET component_id = (
              SELECT u.component_id
              FROM identity_component_update u
              WHERE u.stable_uid = identity_pool.stable_uid
            )
            WHERE component_id > (
              SELECT u.component_id
              FROM identity_component_update u
              WHERE u.stable_uid = identity_pool.stable_uid
            )
        """)
        updates = int(connection.execute("SELECT changes()").fetchone()[0])
        connection.commit()
        iterations += 1
        if updates == 0:
            break
    unresolved = int(connection.execute("""
        SELECT count(*)
        FROM identity_membership m
        JOIN identity_pool p USING(stable_uid)
        JOIN identity_group_min g
          ON g.edge_kind = m.edge_kind
         AND g.key_one = m.key_one
         AND g.key_two = m.key_two
        WHERE p.component_id != g.component_id
    """).fetchone()[0])
    if unresolved:
        raise ValueError(f"identity closure did not converge: unresolved_memberships={unresolved}")
    components = int(connection.execute("SELECT count(DISTINCT component_id) FROM identity_pool").fetchone()[0])
    return {
        "iterations": iterations,
        "last_iteration_updates": updates,
        "components": components,
        "unresolved_memberships": unresolved,
    }


def decision_files(paths: Path | Iterable[Path]) -> list[Path]:
    """Return an ordered, duplicate-free decision-file inventory.

    The within-source pass legitimately has one production-detector output per
    source.  Accepting an explicit list avoids treating a detector work root
    (which also contains staged input and state) as a decision directory.
    """

    candidates = [paths] if isinstance(paths, Path) else list(paths)
    result: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        files = [path] if path.is_file() else discover_parquet(path)
        for file in files:
            resolved = file.resolve()
            if resolved in seen:
                raise ValueError(f"duplicate raw decision file: {file}")
            seen.add(resolved)
            result.append(file)
    if not result:
        raise ValueError("at least one raw decision Parquet file is required")
    return result


def load_raw_decisions(connection: sqlite3.Connection, paths: Path | Iterable[Path], batch_rows: int) -> int:
    import pyarrow.parquet as pq

    count = 0
    for file in decision_files(paths):
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


def validate_near_pass_scope(connection: sqlite3.Connection, pass_kind: str) -> dict[str, int | str]:
    """Prove that a detector invocation cannot leak across its ordered scope."""

    if pass_kind not in NEAR_PASS_KINDS:
        raise ValueError(f"unsupported near pass kind: {pass_kind}")
    base_predicate = "p.source_role = 'base' OR p.acquisition_source_id = 'nanochat_base'"
    overall = connection.execute(
        f"""
        SELECT
          count(*) AS rows,
          sum(CASE WHEN {base_predicate} THEN 1 ELSE 0 END) AS base_rows,
          sum(CASE WHEN NOT ({base_predicate}) THEN 1 ELSE 0 END) AS candidate_rows
        FROM decisions d JOIN pool p ON p.stable_uid = d.stable_uid
        """
    ).fetchone()
    rows = int(overall[0])
    base_rows = int(overall[1] or 0)
    candidate_rows = int(overall[2] or 0)
    components = connection.execute(
        f"""
        SELECT
          d.root_doc_key,
          count(*) AS member_count,
          count(DISTINCT p.acquisition_source_id) AS source_count,
          sum(CASE WHEN {base_predicate} THEN 1 ELSE 0 END) AS base_members,
          sum(CASE WHEN NOT ({base_predicate}) THEN 1 ELSE 0 END) AS candidate_members
        FROM decisions d JOIN pool p ON p.stable_uid = d.stable_uid
        GROUP BY d.root_doc_key
        """
    ).fetchall()
    multi_member = [row for row in components if int(row[1]) > 1]
    invalid_scope = 0
    invalid_input = 0
    if pass_kind == "within-source":
        invalid_scope = sum(1 for row in multi_member if int(row[2]) != 1)
    elif pass_kind == "cross-candidate":
        invalid_input = base_rows
        invalid_scope = sum(1 for row in multi_member if int(row[2]) < 2)
    else:
        invalid_input = int(base_rows == 0 or candidate_rows == 0)
        invalid_scope = sum(
            1
            for row in multi_member
            if int(row[3] or 0) == 0 or int(row[4] or 0) == 0
        )
    result: dict[str, int | str] = {
        "pass_kind": pass_kind,
        "rows": rows,
        "base_rows": base_rows,
        "candidate_rows": candidate_rows,
        "components": len(components),
        "multi_member_components": len(multi_member),
        "invalid_input_rows_or_role": invalid_input,
        "invalid_scope_components": invalid_scope,
    }
    if invalid_input or invalid_scope:
        raise ValueError(
            f"{pass_kind}: detector decision scope violates the ordered dedup contract: {result}"
        )
    return result


def write_ledger(
    connection: sqlite3.Connection,
    output: Path,
    *,
    method: str = "content_work_representation_near_precedence_v1",
    raw_decision_stage: str | None = None,
) -> tuple[int, int]:
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
                    "method": method,
                    "raw_decision_stage": raw_decision_stage or str(row[4] or ""),
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


def write_identity_ledger(
    connection: sqlite3.Connection,
    output: Path,
    *,
    method: str = "content_work_representation_near_precedence_v2",
    raw_decision_stage: str = "identity_closure",
) -> tuple[int, int]:
    """Write one complete ledger after an identity-component closure."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(f"immutable identity-reconciled v3 dedup ledger exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = atomic_output_path(output)
    query = """
      WITH ranked AS (
        SELECT p.stable_uid, p.input_representation_id, p.input_text_sha256,
               p.component_id,
               ROW_NUMBER() OVER (
                 PARTITION BY p.component_id
                 ORDER BY p.source_rank, p.license_rank, p.extraction_rank,
                          p.quality_rank, p.provenance_key, p.stable_uid
               ) AS priority,
               FIRST_VALUE(p.stable_uid) OVER (
                 PARTITION BY p.component_id
                 ORDER BY p.source_rank, p.license_rank, p.extraction_rank,
                          p.quality_rank, p.provenance_key, p.stable_uid
               ) AS representative_stable_uid,
               FIRST_VALUE(p.input_representation_id) OVER (
                 PARTITION BY p.component_id
                 ORDER BY p.source_rank, p.license_rank, p.extraction_rank,
                          p.quality_rank, p.provenance_key, p.stable_uid
               ) AS representative_input_representation_id
        FROM identity_pool p
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
                keep = int(row[4]) == 1
                keeps += int(keep)
                output_rows.append({
                    "stable_uid": str(row[0]),
                    "input_representation_id": str(row[1]),
                    "input_text_sha256": str(row[2]),
                    "action": "keep" if keep else "drop",
                    "representative_stable_uid": str(row[5]),
                    "representative_input_representation_id": str(row[6]),
                    "cluster_id": "v3-component:" + str(row[3]),
                    "method": method,
                    "raw_decision_stage": raw_decision_stage,
                    "reason": "representative" if keep else "nanochat_license_extraction_quality_provenance_stable_id",
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


def cmd_exact_reconcile(args: argparse.Namespace) -> None:
    """Resolve canonical exact identity before any near-duplicate detector run."""

    if args.manifest.exists() or args.output_ledger.exists():
        raise FileExistsError("immutable exact-identity v3 dedup output already exists")
    connection = identity_sqlite_connect(args.work_database)
    try:
        pool_rows = load_identity_pool(connection, args.pool, args.batch_rows)
        groups = identity_group_counts(
            connection,
            required_edge_kinds=EXACT_IDENTITY_EDGE_KINDS,
        )
        closure = resolve_identity_components(connection)
        ledger_rows, kept_rows = write_identity_ledger(
            connection,
            args.output_ledger,
            method="exact_content_work_representation_precedence_v1",
            raw_decision_stage="exact_content_work_representation",
        )
    finally:
        connection.close()
    payload = {
        "schema_version": LEDGER_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "pool": str(args.pool.resolve()),
        "ledger": parquet_file_receipt(args.output_ledger),
        "counts": {
            "pool_rows": pool_rows,
            "ledger_rows": ledger_rows,
            "kept_rows": kept_rows,
            "dropped_rows": ledger_rows - kept_rows,
        },
        "identity_reconciliation": {
            "schema_version": IDENTITY_RECONCILIATION_SCHEMA,
            "edge_kinds": list(EXACT_IDENTITY_EDGE_KINDS),
            "edge_groups": groups,
            "closure": closure,
            "selection_order": "exact_identity_before_near_passes_before_representative_precedence",
        },
        "representative_precedence": [
            "nanochat_base", "license", "extraction_completeness", "quality", "provenance", "stable_id",
        ],
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": payload["counts"]}, sort_keys=True))


def validate_raw_decision_coverage(connection: sqlite3.Connection) -> dict[str, int]:
    missing_pool = int(connection.execute(
        "SELECT count(*) FROM decisions d LEFT JOIN pool p ON p.stable_uid = d.stable_uid WHERE p.stable_uid IS NULL"
    ).fetchone()[0])
    missing_decisions = int(connection.execute(
        "SELECT count(*) FROM pool p LEFT JOIN decisions d ON d.stable_uid = p.stable_uid WHERE d.stable_uid IS NULL"
    ).fetchone()[0])
    result = {
        "decisions_without_pool": missing_pool,
        "pool_without_decision": missing_decisions,
    }
    if any(result.values()):
        raise ValueError(
            "dedup decision coverage mismatch: "
            + ", ".join(f"{key}={value}" for key, value in result.items())
        )
    return result


def raw_decision_receipts(paths: Path | Iterable[Path]) -> list[dict[str, Any]]:
    return [parquet_file_receipt(path) for path in decision_files(paths)]


def cmd_reconcile(args: argparse.Namespace) -> None:
    if args.manifest.exists() or args.output_ledger.exists():
        raise FileExistsError("immutable v3 dedup output already exists")
    connection = sqlite_connect(args.work_database)
    try:
        pool_rows = load_pool(connection, args.pool, args.batch_rows)
        decision_rows = load_raw_decisions(connection, args.raw_decisions, args.batch_rows)
        coverage = validate_raw_decision_coverage(connection)
        resolve_roots(connection)
        ledger_rows, kept_rows = write_ledger(connection, args.output_ledger)
    finally:
        connection.close()
    raw_receipts = raw_decision_receipts(args.raw_decisions)
    payload = {
        "schema_version": LEDGER_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "pool": str(args.pool.resolve()),
        # Retain the legacy object shape for one detector output while allowing
        # the ordered within-source pass to supply an explicit file list.
        "raw_decisions": raw_receipts[0] if len(raw_receipts) == 1 else raw_receipts,
        "ledger": parquet_file_receipt(args.output_ledger),
        "counts": {"pool_rows": pool_rows, "raw_decision_rows": decision_rows, "ledger_rows": ledger_rows, "kept_rows": kept_rows, "dropped_rows": ledger_rows - kept_rows, **coverage},
        "representative_precedence": ["nanochat_base", "license", "extraction_completeness", "quality", "provenance", "stable_id"],
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": payload["counts"]}, sort_keys=True))


def cmd_near_reconcile(args: argparse.Namespace) -> None:
    """Reconcile one explicitly scoped production near-dedup pass."""

    if args.manifest.exists() or args.output_ledger.exists():
        raise FileExistsError("immutable ordered-near v3 dedup output already exists")
    if args.pass_kind not in NEAR_PASS_KINDS:
        raise ValueError(f"unsupported --pass-kind: {args.pass_kind}")
    connection = sqlite_connect(args.work_database)
    try:
        pool_rows = load_pool(connection, args.pool, args.batch_rows)
        decision_rows = load_raw_decisions(connection, args.raw_decisions, args.batch_rows)
        coverage = validate_raw_decision_coverage(connection)
        resolve_roots(connection)
        scope_validation = validate_near_pass_scope(connection, args.pass_kind)
        pass_label = NEAR_PASS_KINDS[args.pass_kind]
        ledger_rows, kept_rows = write_ledger(
            connection,
            args.output_ledger,
            method=f"{pass_label}_production_detector_precedence_v1",
            raw_decision_stage=pass_label,
        )
    finally:
        connection.close()
    payload = {
        "schema_version": LEDGER_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "pool": str(args.pool.resolve()),
        "ordered_pass": pass_label,
        "raw_decisions": raw_decision_receipts(args.raw_decisions),
        "ledger": parquet_file_receipt(args.output_ledger),
        "counts": {
            "pool_rows": pool_rows,
            "raw_decision_rows": decision_rows,
            "ledger_rows": ledger_rows,
            "kept_rows": kept_rows,
            "dropped_rows": ledger_rows - kept_rows,
            **coverage,
        },
        "scope_validation": scope_validation,
        "representative_precedence": [
            "nanochat_base", "license", "extraction_completeness", "quality", "provenance", "stable_id",
        ],
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": payload["counts"]}, sort_keys=True))


def cmd_identity_reconcile(args: argparse.Namespace) -> None:
    """Close generic components with exact canonical identity before selection."""

    if args.manifest.exists() or args.output_ledger.exists():
        raise FileExistsError("immutable identity-reconciled v3 dedup output already exists")
    if not args.provisional_ledger.is_file():
        raise ValueError("--provisional-ledger must be one immutable Parquet ledger file")
    if args.provisional_ledger.resolve() == args.output_ledger.resolve():
        raise ValueError("--output-ledger must differ from --provisional-ledger")
    connection = identity_sqlite_connect(args.work_database)
    try:
        pool_rows = load_identity_pool(connection, args.pool, args.batch_rows)
        provisional_rows = load_provisional_ledger(connection, args.provisional_ledger, args.batch_rows)
        coverage = validate_provisional_ledger_coverage(connection)
        groups = identity_group_counts(connection)
        closure = resolve_identity_components(connection)
        ledger_rows, kept_rows = write_identity_ledger(connection, args.output_ledger)
    finally:
        connection.close()
    payload = {
        "schema_version": LEDGER_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "pool": str(args.pool.resolve()),
        "provisional_ledger": parquet_file_receipt(args.provisional_ledger),
        "ledger": parquet_file_receipt(args.output_ledger),
        "counts": {
            "pool_rows": pool_rows,
            "provisional_ledger_rows": provisional_rows,
            "ledger_rows": ledger_rows,
            "kept_rows": kept_rows,
            "dropped_rows": ledger_rows - kept_rows,
        },
        "identity_reconciliation": {
            "schema_version": IDENTITY_RECONCILIATION_SCHEMA,
            "edge_kinds": list(IDENTITY_EDGE_KINDS),
            "edge_groups": groups,
            "provisional_ledger_coverage": coverage,
            "closure": closure,
            "selection_order": "after_generic_near_and_exact_identity_closure_before_representative_precedence",
        },
        "representative_precedence": [
            "nanochat_base", "license", "extraction_completeness", "quality", "provenance", "stable_id",
        ],
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": payload["counts"]}, sort_keys=True))


def is_base_source(row: dict[str, Any]) -> bool:
    source_id = nonempty_string(row.get("acquisition_source_id"), label="acquisition_source_id")
    role = nonempty_string(row.get("source_role"), label="source_role")
    # The frozen registry deliberately retains distinctions such as
    # ``replacement_candidate``, ``overlap_candidate``, and
    # ``replacement_audit``.  For this ordered pass they are all additive
    # candidates; only the one canonical base may enter the base side.
    if role == "base_overlay":
        raise ValueError(f"{source_id}: base_overlay rows must not enter the admitted v3 pool")
    by_role = role == "base"
    by_source = source_id == "nanochat_base"
    if by_role != by_source:
        raise ValueError(
            f"{source_id}: source_role/base identity disagreement; expected Nanochat base to be uniquely labelled"
        )
    return by_role


def cmd_filter_candidates(args: argparse.Namespace) -> None:
    """Write a receipt-bound current-survivor root containing candidates only."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.manifest.exists():
        raise FileExistsError(f"immutable manifest already exists: {args.manifest}")
    ensure_new_roots(args.output)
    counts: Counter[str] = Counter()
    receipts: list[dict[str, Any]] = []
    for input_path in discover_parquet(args.input):
        relative = input_path.relative_to(args.input)
        output_path = args.output / relative
        parquet = pq.ParquetFile(input_path)
        required = {"stable_uid", "acquisition_source_id", "source_role"}
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{input_path}: missing candidate-scope columns {sorted(missing)}")
        temporary = atomic_output_path(output_path)
        writer: Any | None = None
        try:
            for batch in parquet.iter_batches(batch_size=args.batch_rows, use_threads=False):
                selected: list[dict[str, Any]] = []
                for row in batch.to_pylist():
                    counts["input_rows"] += 1
                    if not is_base_source(row):
                        selected.append(row)
                        counts["candidate_rows"] += 1
                    else:
                        counts["base_rows_excluded"] += 1
                if selected:
                    if writer is None:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(temporary, parquet.schema_arrow, compression="zstd")
                    writer.write_table(pa.Table.from_pylist(selected, schema=parquet.schema_arrow))
            if writer is not None:
                writer.close()
                os.replace(temporary, output_path)
                receipts.append(parquet_file_receipt(output_path, relative_to=args.output))
        except BaseException:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)
            raise
    payload = {
        "schema_version": SCOPE_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "scope": "additive_candidates_only",
        "input": str(args.input.resolve()),
        "counts": dict(counts),
        "files": receipts,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": payload["counts"]}, sort_keys=True))


def cmd_partition_within_source(args: argparse.Namespace) -> None:
    """Partition current survivors so each within-source detector run is isolated."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.manifest.exists():
        raise FileExistsError(f"immutable manifest already exists: {args.manifest}")
    ensure_new_roots(args.output)
    writers: dict[str, tuple[Any, Path, Path, Any]] = {}
    source_roles: dict[str, str] = {}
    counts: Counter[str] = Counter()
    try:
        for input_path in discover_parquet(args.input):
            parquet = pq.ParquetFile(input_path)
            required = {"stable_uid", "acquisition_source_id", "source_role"}
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{input_path}: missing within-source partition columns {sorted(missing)}")
            for batch in parquet.iter_batches(batch_size=args.batch_rows, use_threads=False):
                grouped: dict[str, list[dict[str, Any]]] = {}
                for row in batch.to_pylist():
                    source_id = nonempty_string(row.get("acquisition_source_id"), label="partition acquisition_source_id")
                    role = nonempty_string(row.get("source_role"), label=f"{source_id}: source_role")
                    # Also validates the canonical one-base classification.
                    is_base_source(row)
                    previous_role = source_roles.setdefault(source_id, role)
                    if previous_role != role:
                        raise ValueError(f"{source_id}: source_role changes within the admitted pool")
                    grouped.setdefault(source_id, []).append(row)
                    counts["input_rows"] += 1
                for source_id in sorted(grouped):
                    rows = grouped[source_id]
                    if source_id not in writers:
                        digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:20]
                        final_path = args.output / f"source-{digest}" / "data.parquet"
                        temporary = atomic_output_path(final_path)
                        final_path.parent.mkdir(parents=True, exist_ok=True)
                        writers[source_id] = (
                            pq.ParquetWriter(temporary, parquet.schema_arrow, compression="zstd"),
                            temporary,
                            final_path,
                            parquet.schema_arrow,
                        )
                    writer, _, _, schema = writers[source_id]
                    if not schema.equals(parquet.schema_arrow, check_metadata=True):
                        raise ValueError(f"{source_id}: canonical source shards have incompatible schemas")
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    counts[f"source:{source_id}"] += len(rows)
        if not writers:
            raise ValueError("within-source partition input has no rows")
        records: list[dict[str, Any]] = []
        for source_id in sorted(writers):
            writer, temporary, final_path, _ = writers[source_id]
            writer.close()
            os.replace(temporary, final_path)
            records.append({
                "acquisition_source_id": source_id,
                "source_role": source_roles[source_id],
                "rows": int(counts[f"source:{source_id}"]),
                "path": str(final_path.resolve()),
                "file": parquet_file_receipt(final_path, relative_to=args.output),
            })
    except BaseException:
        for writer, temporary, _, _ in writers.values():
            try:
                writer.close()
            finally:
                temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema_version": PARTITION_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "input": str(args.input.resolve()),
        "counts": {**dict(counts), "sources": len(records)},
        "sources": records,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": payload["counts"]}, sort_keys=True))


def write_passthrough_ledger(
    connection: sqlite3.Connection,
    output: Path,
    *,
    pass_label: str,
    reason: str,
) -> tuple[int, int]:
    """Write an explicit no-op ledger when a required comparison scope is empty."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(f"immutable ordered-pass ledger exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = atomic_output_path(output)
    cursor = connection.execute(
        "SELECT stable_uid, input_representation_id, input_text_sha256 FROM pool ORDER BY stable_uid"
    )
    writer = pq.ParquetWriter(temporary, ledger_schema(), compression="zstd")
    rows = 0
    try:
        while True:
            batch = cursor.fetchmany(4096)
            if not batch:
                break
            output_rows = [
                {
                    "stable_uid": str(row[0]),
                    "input_representation_id": str(row[1]),
                    "input_text_sha256": str(row[2]),
                    "action": "keep",
                    "representative_stable_uid": str(row[0]),
                    "representative_input_representation_id": str(row[1]),
                    "cluster_id": f"v3-{pass_label}-singleton:{row[0]}",
                    "method": "ordered_near_pass_passthrough_v1",
                    "raw_decision_stage": pass_label,
                    "reason": reason,
                }
                for row in batch
            ]
            writer.write_table(pa.Table.from_pylist(output_rows, schema=ledger_schema()))
            rows += len(output_rows)
        writer.close()
        os.replace(temporary, output)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return rows, rows


def cmd_passthrough_near(args: argparse.Namespace) -> None:
    if args.pass_kind not in NEAR_PASS_KINDS:
        raise ValueError(f"unsupported --pass-kind: {args.pass_kind}")
    if args.manifest.exists() or args.output_ledger.exists():
        raise FileExistsError("immutable ordered-pass passthrough output already exists")
    connection = sqlite_connect(args.work_database)
    try:
        pool_rows = load_pool(connection, args.pool, args.batch_rows)
        pass_label = NEAR_PASS_KINDS[args.pass_kind]
        ledger_rows, kept_rows = write_passthrough_ledger(
            connection,
            args.output_ledger,
            pass_label=pass_label,
            reason=args.reason,
        )
    finally:
        connection.close()
    payload = {
        "schema_version": LEDGER_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "pool": str(args.pool.resolve()),
        "ordered_pass": pass_label,
        "detector_invoked": False,
        "passthrough_reason": args.reason,
        "ledger": parquet_file_receipt(args.output_ledger),
        "counts": {
            "pool_rows": pool_rows,
            "ledger_rows": ledger_rows,
            "kept_rows": kept_rows,
            "dropped_rows": ledger_rows - kept_rows,
        },
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": payload["counts"]}, sort_keys=True))


def load_scope_ledger(connection: sqlite3.Connection, path: Path, batch_rows: int) -> int:
    """Load the candidate-only cross-source ledger before extending it to base rows."""

    import pyarrow.parquet as pq

    if not path.is_file() or path.is_symlink():
        raise ValueError("--scope-ledger must be one regular immutable Parquet file")
    connection.execute("""
        CREATE TABLE scope_ledger (
          stable_uid TEXT PRIMARY KEY,
          input_representation_id TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          action TEXT NOT NULL,
          representative_stable_uid TEXT NOT NULL,
          representative_input_representation_id TEXT NOT NULL,
          cluster_id TEXT NOT NULL,
          method TEXT NOT NULL,
          raw_decision_stage TEXT NOT NULL,
          reason TEXT NOT NULL
        )
    """)
    required = set(ledger_schema().names)
    parquet = pq.ParquetFile(path)
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing scoped ledger columns {sorted(missing)}")
    rows = 0
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=batch_rows, use_threads=False):
        records: list[tuple[str, str, str, str, str, str, str, str, str, str]] = []
        for row in batch.to_pylist():
            uid = nonempty_string(row.get("stable_uid"), label="scope ledger stable_uid")
            action = nonempty_string(row.get("action"), label=f"{uid}: scope action")
            if action not in {"keep", "drop"}:
                raise ValueError(f"{uid}: scope ledger action must be keep or drop")
            records.append(tuple(
                nonempty_string(row.get(name), label=f"{uid}: scope {name}")
                for name in ledger_schema().names
            ))
        try:
            connection.executemany("INSERT INTO scope_ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
        except sqlite3.IntegrityError as exc:
            raise ValueError("scope ledger has duplicate stable_uid") from exc
        rows += len(records)
    connection.commit()
    if not rows:
        raise ValueError("scope ledger is empty")
    return rows


def validate_candidate_scope_ledger(connection: sqlite3.Connection) -> dict[str, int]:
    base_predicate = "p.source_role = 'base' OR p.acquisition_source_id = 'nanochat_base'"
    missing_candidates = int(connection.execute(
        f"SELECT count(*) FROM pool p LEFT JOIN scope_ledger s USING(stable_uid) WHERE NOT ({base_predicate}) AND s.stable_uid IS NULL"
    ).fetchone()[0])
    scope_outside_candidates = int(connection.execute(
        f"SELECT count(*) FROM scope_ledger s JOIN pool p USING(stable_uid) WHERE {base_predicate}"
    ).fetchone()[0])
    scope_outside_pool = int(connection.execute(
        "SELECT count(*) FROM scope_ledger s LEFT JOIN pool p USING(stable_uid) WHERE p.stable_uid IS NULL"
    ).fetchone()[0])
    content_drift = int(connection.execute(
        "SELECT count(*) FROM scope_ledger s JOIN pool p USING(stable_uid) "
        "WHERE s.input_representation_id != p.input_representation_id OR s.input_text_sha256 != p.input_text_sha256"
    ).fetchone()[0])
    missing_representative = int(connection.execute(
        "SELECT count(*) FROM scope_ledger s LEFT JOIN scope_ledger r ON r.stable_uid = s.representative_stable_uid "
        "WHERE r.stable_uid IS NULL"
    ).fetchone()[0])
    representative_drift = int(connection.execute(
        "SELECT count(*) FROM scope_ledger s JOIN scope_ledger r ON r.stable_uid = s.representative_stable_uid "
        "WHERE s.representative_input_representation_id != r.input_representation_id"
    ).fetchone()[0])
    invalid_action_graph = int(connection.execute(
        "SELECT count(*) FROM scope_ledger s JOIN scope_ledger r ON r.stable_uid = s.representative_stable_uid "
        "WHERE (s.action = 'keep' AND s.representative_stable_uid != s.stable_uid) "
        "OR (s.action = 'drop' AND (s.representative_stable_uid = s.stable_uid OR r.action != 'keep'))"
    ).fetchone()[0])
    result = {
        "missing_candidates": missing_candidates,
        "scope_outside_candidates": scope_outside_candidates,
        "scope_outside_pool": scope_outside_pool,
        "content_binding_drift": content_drift,
        "missing_declared_representative": missing_representative,
        "representative_binding_drift": representative_drift,
        "invalid_scope_action_graph": invalid_action_graph,
    }
    if any(result.values()):
        raise ValueError(
            "candidate scope ledger does not exactly extend the current pool: "
            + ", ".join(f"{key}={value}" for key, value in result.items())
        )
    return result


def write_extended_candidate_ledger(connection: sqlite3.Connection, output: Path) -> tuple[int, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(f"immutable extended candidate ledger exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = atomic_output_path(output)
    cursor = connection.execute("""
        SELECT p.stable_uid, p.input_representation_id, p.input_text_sha256,
               p.source_role, p.acquisition_source_id,
               s.action, s.representative_stable_uid,
               s.representative_input_representation_id, s.cluster_id,
               s.method, s.raw_decision_stage, s.reason
        FROM pool p LEFT JOIN scope_ledger s USING(stable_uid)
        ORDER BY p.stable_uid
    """)
    writer = pq.ParquetWriter(temporary, ledger_schema(), compression="zstd")
    rows = keeps = 0
    try:
        while True:
            batch = cursor.fetchmany(4096)
            if not batch:
                break
            output_rows: list[dict[str, str]] = []
            for row in batch:
                uid, representation, text_hash, role, source_id = map(str, row[:5])
                base = role == "base" or source_id == "nanochat_base"
                if base:
                    output_rows.append({
                        "stable_uid": uid,
                        "input_representation_id": representation,
                        "input_text_sha256": text_hash,
                        "action": "keep",
                        "representative_stable_uid": uid,
                        "representative_input_representation_id": representation,
                        "cluster_id": f"v3-cross_candidate_near-passthrough:{uid}",
                        "method": "cross_candidate_near_scope_extension_v1",
                        "raw_decision_stage": "cross_candidate_near",
                        "reason": "outside_candidate_scope",
                    })
                    keeps += 1
                else:
                    action = str(row[5])
                    output_rows.append({
                        "stable_uid": uid,
                        "input_representation_id": representation,
                        "input_text_sha256": text_hash,
                        "action": action,
                        "representative_stable_uid": str(row[6]),
                        "representative_input_representation_id": str(row[7]),
                        "cluster_id": str(row[8]),
                        "method": str(row[9]),
                        "raw_decision_stage": str(row[10]),
                        "reason": str(row[11]),
                    })
                    keeps += int(action == "keep")
            writer.write_table(pa.Table.from_pylist(output_rows, schema=ledger_schema()))
            rows += len(output_rows)
        writer.close()
        os.replace(temporary, output)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return rows, keeps


def cmd_extend_candidate_scope_ledger(args: argparse.Namespace) -> None:
    """Merge candidate-only cross-source decisions with base passthrough rows."""

    if args.manifest.exists() or args.output_ledger.exists():
        raise FileExistsError("immutable extended candidate ledger output already exists")
    connection = sqlite_connect(args.work_database)
    try:
        pool_rows = load_pool(connection, args.pool, args.batch_rows)
        scope_rows = load_scope_ledger(connection, args.scope_ledger, args.batch_rows)
        scope_validation = validate_candidate_scope_ledger(connection)
        ledger_rows, kept_rows = write_extended_candidate_ledger(connection, args.output_ledger)
    finally:
        connection.close()
    payload = {
        "schema_version": LEDGER_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "pool": str(args.pool.resolve()),
        "ordered_pass": "cross_candidate_near",
        "scope_ledger": parquet_file_receipt(args.scope_ledger),
        "ledger": parquet_file_receipt(args.output_ledger),
        "counts": {
            "pool_rows": pool_rows,
            "scope_ledger_rows": scope_rows,
            "ledger_rows": ledger_rows,
            "kept_rows": kept_rows,
            "dropped_rows": ledger_rows - kept_rows,
        },
        "scope_validation": scope_validation,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": payload["counts"]}, sort_keys=True))


def regular_file_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"regular file receipt requires a non-symlink file: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_stage_manifest(stage: str, ledger: Path, manifest: Path) -> dict[str, Any]:
    value = read_object(manifest)
    if value.get("schema_version") != LEDGER_MANIFEST or value.get("status") != "passed":
        raise ValueError(f"{stage}: completed v3 dedup ledger manifest required")
    receipt = value.get("ledger")
    if not isinstance(receipt, dict):
        raise ValueError(f"{stage}: ledger manifest lacks a ledger receipt")
    if Path(str(receipt.get("path", ""))).resolve() != ledger.resolve():
        raise ValueError(f"{stage}: ledger manifest path does not bind its supplied ledger")
    if receipt.get("sha256") != sha256_file(ledger) or receipt.get("bytes") != ledger.stat().st_size:
        raise ValueError(f"{stage}: ledger manifest receipt drift")
    expected_ordered_pass = None if stage == "exact_content_work_representation" else stage
    if expected_ordered_pass is not None and value.get("ordered_pass") != expected_ordered_pass:
        raise ValueError(f"{stage}: ledger manifest ordered_pass drift")
    if stage == "exact_content_work_representation":
        identity = value.get("identity_reconciliation")
        if not isinstance(identity, dict) or identity.get("selection_order") != "exact_identity_before_near_passes_before_representative_precedence":
            raise ValueError("exact identity ledger manifest does not prove identity-before-near ordering")
    return value


def ordered_compose_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite_connect(path)
    connection.executescript("""
        CREATE TABLE ordered_state (
          stable_uid TEXT PRIMARY KEY,
          input_representation_id TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          active INTEGER NOT NULL,
          parent_stable_uid TEXT NOT NULL,
          first_drop_stage TEXT,
          first_drop_reason TEXT,
          root_stable_uid TEXT
        );
        CREATE TABLE ordered_stage (
          stable_uid TEXT PRIMARY KEY,
          input_representation_id TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          action TEXT NOT NULL,
          representative_stable_uid TEXT NOT NULL,
          representative_input_representation_id TEXT NOT NULL,
          cluster_id TEXT NOT NULL,
          method TEXT NOT NULL,
          raw_decision_stage TEXT NOT NULL,
          reason TEXT NOT NULL
        );
    """)
    return connection


def initialize_ordered_state(connection: sqlite3.Connection) -> None:
    connection.execute("""
        INSERT INTO ordered_state (
          stable_uid, input_representation_id, input_text_sha256, active,
          parent_stable_uid, first_drop_stage, first_drop_reason, root_stable_uid
        )
        SELECT stable_uid, input_representation_id, input_text_sha256, 1,
               stable_uid, NULL, NULL, stable_uid
        FROM pool
    """)
    connection.commit()


def load_ordered_stage_ledger(connection: sqlite3.Connection, path: Path, batch_rows: int) -> int:
    import pyarrow.parquet as pq

    if not path.is_file() or path.is_symlink():
        raise ValueError("ordered stage ledger must be one regular immutable Parquet file")
    connection.execute("DELETE FROM ordered_stage")
    required = set(ledger_schema().names)
    parquet = pq.ParquetFile(path)
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing ordered stage ledger columns {sorted(missing)}")
    rows = 0
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=batch_rows, use_threads=False):
        records: list[tuple[str, str, str, str, str, str, str, str, str, str]] = []
        for row in batch.to_pylist():
            uid = nonempty_string(row.get("stable_uid"), label="ordered stage stable_uid")
            action = nonempty_string(row.get("action"), label=f"{uid}: ordered stage action")
            if action not in {"keep", "drop"}:
                raise ValueError(f"{uid}: ordered stage action must be keep or drop")
            records.append(tuple(
                nonempty_string(row.get(name), label=f"{uid}: ordered stage {name}")
                for name in ledger_schema().names
            ))
        try:
            connection.executemany("INSERT INTO ordered_stage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
        except sqlite3.IntegrityError as exc:
            raise ValueError("ordered stage ledger has duplicate stable_uid") from exc
        rows += len(records)
    connection.commit()
    if not rows:
        raise ValueError("ordered stage ledger is empty")
    return rows


def validate_ordered_stage_ledger(connection: sqlite3.Connection, stage: str) -> dict[str, int]:
    active_without_stage = int(connection.execute(
        "SELECT count(*) FROM ordered_state o LEFT JOIN ordered_stage s USING(stable_uid) "
        "WHERE o.active = 1 AND s.stable_uid IS NULL"
    ).fetchone()[0])
    stage_outside_active = int(connection.execute(
        "SELECT count(*) FROM ordered_stage s LEFT JOIN ordered_state o USING(stable_uid) "
        "WHERE o.stable_uid IS NULL OR o.active != 1"
    ).fetchone()[0])
    content_drift = int(connection.execute(
        "SELECT count(*) FROM ordered_stage s JOIN ordered_state o USING(stable_uid) "
        "WHERE s.input_representation_id != o.input_representation_id OR s.input_text_sha256 != o.input_text_sha256"
    ).fetchone()[0])
    missing_representative = int(connection.execute(
        "SELECT count(*) FROM ordered_stage s LEFT JOIN ordered_stage r "
        "ON r.stable_uid = s.representative_stable_uid WHERE r.stable_uid IS NULL"
    ).fetchone()[0])
    representative_drift = int(connection.execute(
        "SELECT count(*) FROM ordered_stage s JOIN ordered_stage r "
        "ON r.stable_uid = s.representative_stable_uid "
        "WHERE s.representative_input_representation_id != r.input_representation_id"
    ).fetchone()[0])
    invalid_action_graph = int(connection.execute(
        "SELECT count(*) FROM ordered_stage s JOIN ordered_stage r "
        "ON r.stable_uid = s.representative_stable_uid "
        "WHERE (s.action = 'keep' AND s.representative_stable_uid != s.stable_uid) "
        "OR (s.action = 'drop' AND (s.representative_stable_uid = s.stable_uid OR r.action != 'keep'))"
    ).fetchone()[0])
    result = {
        "active_without_stage_ledger": active_without_stage,
        "stage_ledger_outside_active_pool": stage_outside_active,
        "content_binding_drift": content_drift,
        "missing_declared_representative": missing_representative,
        "representative_binding_drift": representative_drift,
        "invalid_action_graph": invalid_action_graph,
    }
    if any(result.values()):
        raise ValueError(
            f"{stage}: ordered stage ledger does not exactly cover prior survivors: "
            + ", ".join(f"{key}={value}" for key, value in result.items())
        )
    return result


def apply_ordered_stage(connection: sqlite3.Connection, stage: str) -> None:
    connection.execute(
        """
        UPDATE ordered_state
        SET parent_stable_uid = (
              SELECT s.representative_stable_uid FROM ordered_stage s
              WHERE s.stable_uid = ordered_state.stable_uid
            ),
            active = (
              SELECT CASE WHEN s.action = 'keep' THEN 1 ELSE 0 END FROM ordered_stage s
              WHERE s.stable_uid = ordered_state.stable_uid
            ),
            first_drop_stage = CASE
              WHEN (SELECT s.action FROM ordered_stage s WHERE s.stable_uid = ordered_state.stable_uid) = 'drop'
              THEN ? ELSE first_drop_stage END,
            first_drop_reason = CASE
              WHEN (SELECT s.action FROM ordered_stage s WHERE s.stable_uid = ordered_state.stable_uid) = 'drop'
              THEN (SELECT s.reason FROM ordered_stage s WHERE s.stable_uid = ordered_state.stable_uid)
              ELSE first_drop_reason END
        WHERE stable_uid IN (SELECT stable_uid FROM ordered_stage)
        """,
        (stage,),
    )
    connection.commit()


def resolve_ordered_roots(connection: sqlite3.Connection) -> dict[str, int]:
    connection.execute("UPDATE ordered_state SET root_stable_uid = parent_stable_uid")
    iterations = 0
    while True:
        connection.execute(
            """
            UPDATE ordered_state
            SET root_stable_uid = (
              SELECT parent_stable_uid FROM ordered_state parent
              WHERE parent.stable_uid = ordered_state.root_stable_uid
            )
            WHERE root_stable_uid != (
              SELECT parent_stable_uid FROM ordered_state parent
              WHERE parent.stable_uid = ordered_state.root_stable_uid
            )
            """
        )
        updates = int(connection.execute("SELECT changes()").fetchone()[0])
        connection.commit()
        iterations += 1
        if updates == 0:
            break
        if iterations > len(ORDERED_PASS_LABELS) + 2:
            raise ValueError("ordered representative chains did not converge")
    invalid_roots = int(connection.execute(
        "SELECT count(*) FROM ordered_state o JOIN ordered_state r "
        "ON r.stable_uid = o.root_stable_uid WHERE r.active != 1 OR r.parent_stable_uid != r.stable_uid"
    ).fetchone()[0])
    active_root_drift = int(connection.execute(
        "SELECT count(*) FROM ordered_state WHERE (active = 1) != (stable_uid = root_stable_uid)"
    ).fetchone()[0])
    if invalid_roots or active_root_drift:
        raise ValueError(
            "ordered representative closure does not terminate at final survivors: "
            f"invalid_roots={invalid_roots}, active_root_drift={active_root_drift}"
        )
    return {
        "iterations": iterations,
        "invalid_roots": invalid_roots,
        "active_root_drift": active_root_drift,
        "final_components": int(connection.execute("SELECT count(DISTINCT root_stable_uid) FROM ordered_state").fetchone()[0]),
    }


def write_ordered_composed_ledger(connection: sqlite3.Connection, output: Path) -> tuple[int, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(f"immutable ordered final ledger exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = atomic_output_path(output)
    cursor = connection.execute("""
        SELECT o.stable_uid, o.input_representation_id, o.input_text_sha256,
               o.root_stable_uid, r.input_representation_id,
               o.first_drop_stage, o.first_drop_reason
        FROM ordered_state o JOIN ordered_state r ON r.stable_uid = o.root_stable_uid
        ORDER BY o.stable_uid
    """)
    writer = pq.ParquetWriter(temporary, ledger_schema(), compression="zstd")
    rows = keeps = 0
    try:
        while True:
            batch = cursor.fetchmany(4096)
            if not batch:
                break
            output_rows: list[dict[str, str]] = []
            for row in batch:
                uid, representation, text_hash, representative, representative_representation, drop_stage, drop_reason = row
                keep = str(uid) == str(representative)
                keeps += int(keep)
                output_rows.append({
                    "stable_uid": str(uid),
                    "input_representation_id": str(representation),
                    "input_text_sha256": str(text_hash),
                    "action": "keep" if keep else "drop",
                    "representative_stable_uid": str(representative),
                    "representative_input_representation_id": str(representative_representation),
                    "cluster_id": f"v3-ordered-component:{representative}",
                    "method": "exact_then_within_source_then_cross_candidate_then_candidate_to_nanochat_precedence_v1",
                    "raw_decision_stage": "candidate_to_nanochat_near" if keep else str(drop_stage),
                    "reason": "representative" if keep else str(drop_reason),
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


def cmd_compose_ordered_ledgers(args: argparse.Namespace) -> None:
    """Compose exact and all ordered near-pass ledgers into one final ledger."""

    if args.manifest.exists() or args.output_ledger.exists():
        raise FileExistsError("immutable ordered v3 final ledger output already exists")
    stages = list(args.stage)
    ledgers = list(args.stage_ledger)
    manifests = list(args.stage_manifest)
    if stages != list(ORDERED_PASS_LABELS):
        raise ValueError(f"ordered stages must be exactly {list(ORDERED_PASS_LABELS)}")
    if len(ledgers) != len(stages) or len(manifests) != len(stages):
        raise ValueError("--stage, --stage-ledger, and --stage-manifest counts must match")
    for stage, ledger, manifest in zip(stages, ledgers, manifests, strict=True):
        verify_stage_manifest(stage, ledger, manifest)
    connection = ordered_compose_connect(args.work_database)
    stage_validation: list[dict[str, Any]] = []
    try:
        pool_rows = load_pool(connection, args.pool, args.batch_rows)
        initialize_ordered_state(connection)
        for stage, ledger in zip(stages, ledgers, strict=True):
            ledger_rows = load_ordered_stage_ledger(connection, ledger, args.batch_rows)
            validation = validate_ordered_stage_ledger(connection, stage)
            apply_ordered_stage(connection, stage)
            active_after = int(connection.execute("SELECT count(*) FROM ordered_state WHERE active = 1").fetchone()[0])
            stage_validation.append({
                "stage": stage,
                "ledger": parquet_file_receipt(ledger),
                "ledger_rows": ledger_rows,
                "validation": validation,
                "active_rows_after": active_after,
            })
        closure = resolve_ordered_roots(connection)
        ledger_rows, kept_rows = write_ordered_composed_ledger(connection, args.output_ledger)
    finally:
        connection.close()
    payload = {
        "schema_version": LEDGER_MANIFEST,
        "status": "passed",
        "completed_at": utc_now(),
        "pool": str(args.pool.resolve()),
        "ledger": parquet_file_receipt(args.output_ledger),
        "counts": {
            "pool_rows": pool_rows,
            "ledger_rows": ledger_rows,
            "kept_rows": kept_rows,
            "dropped_rows": ledger_rows - kept_rows,
        },
        "identity_reconciliation": {
            "schema_version": IDENTITY_RECONCILIATION_SCHEMA,
            "selection_order": "exact_identity_then_within_source_then_cross_candidate_then_candidate_to_nanochat_before_representative_precedence",
            "exact_identity_precedes_near_passes": True,
        },
        "ordered_dedup": {
            "schema_version": ORDERED_COMPOSITION_SCHEMA,
            "pass_order": list(ORDERED_PASS_LABELS),
            "exact_identity_precedes_near_passes": True,
            "stage_ledgers": [
                {
                    "stage": stage,
                    "ledger": parquet_file_receipt(ledger),
                    "manifest": regular_file_receipt(manifest),
                }
                for stage, ledger, manifest in zip(stages, ledgers, manifests, strict=True)
            ],
            "stage_validation": stage_validation,
            "representative_chain_closure": closure,
        },
        "representative_precedence": [
            "nanochat_base", "license", "extraction_completeness", "quality", "provenance", "stable_id",
        ],
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": payload["counts"]}, sort_keys=True))


def load_materialization_ledger(
    connection: sqlite3.Connection,
    ledger: Path,
    batch_rows: int,
) -> dict[str, int]:
    """Load a final ledger and prove it covers the entire admitted pool.

    Materialization used to look only at keep rows.  That makes a malformed
    partial ledger indistinguishable from a deliberate set of drops, so keep
    the full ledger in SQLite and reject missing/extra/content-drift rows
    before writing any corpus output.
    """

    import pyarrow.parquet as pq

    connection.execute("""
        CREATE TABLE materialization_ledger (
          stable_uid TEXT PRIMARY KEY,
          input_representation_id TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          action TEXT NOT NULL
        )
    """)
    required = {"stable_uid", "input_representation_id", "input_text_sha256", "action"}
    rows = 0
    for file in decision_files(ledger):
        parquet = pq.ParquetFile(file)
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{file}: missing final dedup ledger columns {sorted(missing)}")
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=batch_rows, use_threads=False):
            records: list[tuple[str, str, str, str]] = []
            for row in batch.to_pylist():
                stable_uid = nonempty_string(row.get("stable_uid"), label="final ledger stable_uid")
                action = nonempty_string(row.get("action"), label=f"{stable_uid}: action")
                if action not in {"keep", "drop"}:
                    raise ValueError(f"{stable_uid}: final ledger action must be keep or drop")
                records.append((
                    stable_uid,
                    nonempty_string(row.get("input_representation_id"), label=f"{stable_uid}: input_representation_id"),
                    nonempty_string(row.get("input_text_sha256"), label=f"{stable_uid}: input_text_sha256"),
                    action,
                ))
            try:
                connection.executemany("INSERT INTO materialization_ledger VALUES (?, ?, ?, ?)", records)
            except sqlite3.IntegrityError as exc:
                raise ValueError("final dedup ledger has duplicate stable_uid") from exc
            rows += len(records)
        connection.commit()
    missing = int(connection.execute(
        "SELECT count(*) FROM pool p LEFT JOIN materialization_ledger l USING(stable_uid) "
        "WHERE l.stable_uid IS NULL"
    ).fetchone()[0])
    extra = int(connection.execute(
        "SELECT count(*) FROM materialization_ledger l LEFT JOIN pool p USING(stable_uid) "
        "WHERE p.stable_uid IS NULL"
    ).fetchone()[0])
    drift = int(connection.execute(
        "SELECT count(*) FROM materialization_ledger l JOIN pool p USING(stable_uid) "
        "WHERE l.input_representation_id != p.input_representation_id "
        "OR l.input_text_sha256 != p.input_text_sha256"
    ).fetchone()[0])
    result = {
        "ledger_rows": rows,
        "pool_rows_without_ledger": missing,
        "ledger_rows_outside_pool": extra,
        "ledger_content_binding_drift": drift,
    }
    if missing or extra or drift:
        raise ValueError(
            "final dedup ledger must be a complete content-bound decision ledger: "
            + ", ".join(f"{key}={value}" for key, value in result.items())
        )
    connection.execute(
        "CREATE TABLE keepers (stable_uid TEXT PRIMARY KEY, input_text_sha256 TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO keepers SELECT stable_uid, input_text_sha256 FROM materialization_ledger WHERE action = 'keep'"
    )
    connection.commit()
    result["kept_rows"] = int(connection.execute("SELECT count(*) FROM keepers").fetchone()[0])
    result["dropped_rows"] = rows - result["kept_rows"]
    return result


def cmd_materialize(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.manifest.exists():
        raise FileExistsError(f"immutable manifest already exists: {args.manifest}")
    ensure_new_roots(args.output)
    database = sqlite_connect(args.work_database)
    try:
        pool_rows = load_pool(database, args.pool, args.batch_rows)
        ledger_validation = load_materialization_ledger(database, args.ledger, args.batch_rows)
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
        "counts": {**dict(counts), "pool_rows": pool_rows, **ledger_validation},
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
    exact = sub.add_parser(
        "exact-reconcile",
        help="resolve exact content/work/representation identity before any near pass",
    )
    exact.add_argument("--pool", type=Path, required=True)
    exact.add_argument("--output-ledger", type=Path, required=True)
    exact.add_argument("--work-database", type=Path, required=True)
    exact.add_argument("--manifest", type=Path, required=True)
    exact.add_argument("--batch-rows", type=int, default=4096)
    exact.set_defaults(func=cmd_exact_reconcile)
    partition = sub.add_parser(
        "partition-within-source",
        help="partition current survivors for isolated within-source near passes",
    )
    partition.add_argument("--input", type=Path, required=True)
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument("--manifest", type=Path, required=True)
    partition.add_argument("--batch-rows", type=int, default=2048)
    partition.set_defaults(func=cmd_partition_within_source)
    candidates = sub.add_parser(
        "filter-candidates",
        help="write the current additive-candidate-only survivor scope",
    )
    candidates.add_argument("--input", type=Path, required=True)
    candidates.add_argument("--output", type=Path, required=True)
    candidates.add_argument("--manifest", type=Path, required=True)
    candidates.add_argument("--batch-rows", type=int, default=2048)
    candidates.set_defaults(func=cmd_filter_candidates)
    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("--pool", type=Path, required=True)
    reconcile.add_argument("--raw-decisions", type=Path, action="append", required=True)
    reconcile.add_argument("--output-ledger", type=Path, required=True)
    reconcile.add_argument("--work-database", type=Path, required=True)
    reconcile.add_argument("--manifest", type=Path, required=True)
    reconcile.add_argument("--batch-rows", type=int, default=4096)
    reconcile.set_defaults(func=cmd_reconcile)
    near = sub.add_parser(
        "near-reconcile",
        help="reconcile one scope-proven ordered production near-dedup pass",
    )
    near.add_argument("--pool", type=Path, required=True)
    near.add_argument("--pass-kind", choices=sorted(NEAR_PASS_KINDS), required=True)
    near.add_argument("--raw-decisions", type=Path, action="append", required=True)
    near.add_argument("--output-ledger", type=Path, required=True)
    near.add_argument("--work-database", type=Path, required=True)
    near.add_argument("--manifest", type=Path, required=True)
    near.add_argument("--batch-rows", type=int, default=4096)
    near.set_defaults(func=cmd_near_reconcile)
    passthrough = sub.add_parser(
        "passthrough-near",
        help="ledger an ordered near pass whose required comparison scope is empty",
    )
    passthrough.add_argument("--pool", type=Path, required=True)
    passthrough.add_argument("--pass-kind", choices=sorted(NEAR_PASS_KINDS), required=True)
    passthrough.add_argument("--reason", required=True)
    passthrough.add_argument("--output-ledger", type=Path, required=True)
    passthrough.add_argument("--work-database", type=Path, required=True)
    passthrough.add_argument("--manifest", type=Path, required=True)
    passthrough.add_argument("--batch-rows", type=int, default=4096)
    passthrough.set_defaults(func=cmd_passthrough_near)
    extend = sub.add_parser(
        "extend-candidate-scope-ledger",
        help="extend candidate-only cross-source decisions with base passthrough rows",
    )
    extend.add_argument("--pool", type=Path, required=True)
    extend.add_argument("--scope-ledger", type=Path, required=True)
    extend.add_argument("--output-ledger", type=Path, required=True)
    extend.add_argument("--work-database", type=Path, required=True)
    extend.add_argument("--manifest", type=Path, required=True)
    extend.add_argument("--batch-rows", type=int, default=4096)
    extend.set_defaults(func=cmd_extend_candidate_scope_ledger)
    identity = sub.add_parser(
        "identity-reconcile",
        help="close a complete generic ledger with exact content/work/representation identities",
    )
    identity.add_argument("--pool", type=Path, required=True)
    identity.add_argument("--provisional-ledger", type=Path, required=True)
    identity.add_argument("--output-ledger", type=Path, required=True)
    identity.add_argument("--work-database", type=Path, required=True)
    identity.add_argument("--manifest", type=Path, required=True)
    identity.add_argument("--batch-rows", type=int, default=4096)
    identity.set_defaults(func=cmd_identity_reconcile)
    compose = sub.add_parser(
        "compose-ordered-ledgers",
        help="compose exact and ordered near-pass ledgers into the final complete decision ledger",
    )
    compose.add_argument("--pool", type=Path, required=True)
    compose.add_argument("--stage", action="append", required=True)
    compose.add_argument("--stage-ledger", type=Path, action="append", required=True)
    compose.add_argument("--stage-manifest", type=Path, action="append", required=True)
    compose.add_argument("--output-ledger", type=Path, required=True)
    compose.add_argument("--work-database", type=Path, required=True)
    compose.add_argument("--manifest", type=Path, required=True)
    compose.add_argument("--batch-rows", type=int, default=4096)
    compose.set_defaults(func=cmd_compose_ordered_ledgers)
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

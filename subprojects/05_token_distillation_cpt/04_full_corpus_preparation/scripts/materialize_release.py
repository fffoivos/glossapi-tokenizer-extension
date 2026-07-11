#!/usr/bin/env python3
"""Materialize deduplicated training and redistribution Parquet releases."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_token_waterfall import build_waterfall
from finalization_io import (
    atomic_output_path,
    configure_duckdb,
    discover_parquet,
    parquet_file_receipt,
    read_json_object,
    sha256_file,
    sql_path_list,
    sql_string,
    utc_now,
    write_json_atomic,
)


REDISTRIBUTION_DENY_COLUMNS = {"author", "source_metadata_json"}
DEDUP_COLUMNS = [
    "decision_stage AS dedup_decision_stage",
    "cluster_id AS dedup_cluster_id",
    "kept_doc_key AS dedup_kept_doc_key",
    "exact_strict_version AS dedup_exact_strict_version",
    "exact_relaxed_version AS dedup_exact_relaxed_version",
    "near_norm_version AS dedup_near_norm_version",
    "shingle_version AS dedup_shingle_version",
    "selection_version AS dedup_selection_version",
]


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def validate_inputs(connection: Any) -> dict[str, int]:
    result = connection.execute(
        """
        SELECT
          (SELECT count(*) FROM corpus WHERE eligible_for_training) AS eligible_rows,
          (SELECT count(*) - count(DISTINCT stable_uid) FROM corpus WHERE eligible_for_training)
            AS duplicate_stable_uid,
          (SELECT count(*) FROM decisions) AS decision_rows,
          (SELECT count(*) - count(DISTINCT source_doc_id) FROM decisions) AS duplicate_decision_uid,
          (SELECT count(*) FROM decisions WHERE decision NOT IN ('keep', 'drop')) AS bad_decisions,
          (SELECT count(*) FROM corpus c LEFT JOIN decisions d
             ON d.source_doc_id = c.stable_uid AND d.source_dataset = c.source_dataset
             WHERE c.eligible_for_training AND d.source_doc_id IS NULL) AS eligible_without_decision,
          (SELECT count(*) FROM decisions d LEFT JOIN corpus c
             ON d.source_doc_id = c.stable_uid AND d.source_dataset = c.source_dataset
             WHERE c.stable_uid IS NULL) AS decision_without_input,
          (SELECT count(*) FROM decisions WHERE decision = 'keep') AS expected_training_rows
        """
    ).fetchone()
    names = [
        "eligible_rows",
        "duplicate_stable_uid",
        "decision_rows",
        "duplicate_decision_uid",
        "bad_decisions",
        "eligible_without_decision",
        "decision_without_input",
        "expected_training_rows",
    ]
    payload = {name: int(value) for name, value in zip(names, result, strict=True)}
    if payload["eligible_rows"] != payload["decision_rows"]:
        raise ValueError(f"eligible input and dedup decision counts differ: {payload}")
    if any(
        payload[key]
        for key in (
            "duplicate_stable_uid",
            "duplicate_decision_uid",
            "bad_decisions",
            "eligible_without_decision",
            "decision_without_input",
        )
    ):
        raise ValueError(f"release materialization input gate failed: {payload}")
    return payload


def copy_partition(
    connection: Any,
    *,
    input_path: Path,
    output_path: Path,
    columns: list[str],
    redistribution: bool,
) -> dict[str, Any]:
    selected = ", ".join(f"c.{quote_identifier(column)}" for column in columns)
    dedup = ", ".join(f"d.{expression}" for expression in DEDUP_COLUMNS)
    eligibility = "AND c.eligible_for_redistribution" if redistribution else "AND c.eligible_for_training"
    query = f"""
        SELECT {selected}, {dedup}
        FROM read_parquet({sql_string(input_path.resolve())}) c
        JOIN decisions d
          ON d.source_doc_id = c.stable_uid
         AND d.source_dataset = c.source_dataset
        WHERE d.decision = 'keep' {eligibility}
        ORDER BY c.stable_uid
    """
    temporary = atomic_output_path(output_path)
    connection.execute(
        f"COPY ({query}) TO {sql_string(temporary.resolve())} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    os.replace(temporary, output_path)
    return parquet_file_receipt(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Decontaminated canonical Parquet root")
    parser.add_argument("--dedup-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--token-waterfall", type=Path, required=True)
    parser.add_argument("--cleaning-ledger", type=Path)
    parser.add_argument("--decontam-ledger", type=Path)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--memory-limit", default="200GB")
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    import duckdb
    import pyarrow.parquet as pq

    args = parse_args()
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {args.manifest}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to materialize into non-empty output root: {args.output}")
    if not args.dedup_decisions.is_file():
        raise FileNotFoundError(args.dedup_decisions)
    input_files = discover_parquet(args.input)
    first_schema = pq.ParquetFile(input_files[0]).schema_arrow
    required = {
        "source_dataset",
        "source_doc_id",
        "text",
        "stable_uid",
        "cleaned_text_sha256",
        "acquisition_source_id",
        "source_repo_id",
        "source_revision",
        "eligible_for_training",
        "eligible_for_redistribution",
    }
    missing = required - set(first_schema.names)
    if missing:
        raise ValueError(f"release input schema misses required columns: {sorted(missing)}")
    collisions = {expression.split(" AS ", 1)[1] for expression in DEDUP_COLUMNS} & set(first_schema.names)
    if collisions:
        raise ValueError(f"release input already contains reserved dedup columns: {sorted(collisions)}")
    for path in input_files[1:]:
        if pq.ParquetFile(path).schema_arrow != first_schema:
            raise ValueError(f"canonical input schema drift: {path}")

    if not args.token_waterfall.exists():
        if args.cleaning_ledger is None or args.decontam_ledger is None:
            raise FileNotFoundError(
                f"{args.token_waterfall} is missing; provide --cleaning-ledger and --decontam-ledger to build it"
            )
        build_waterfall(
            cleaning_ledger=args.cleaning_ledger,
            decontam_ledger=args.decontam_ledger,
            dedup_decisions=args.dedup_decisions,
            output=args.token_waterfall,
            temporary_directory=args.temporary_directory / "waterfall",
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
    waterfall = read_json_object(args.token_waterfall)
    if waterfall.get("schema_version") != "full_cpt_token_waterfall_v1":
        raise ValueError("--token-waterfall has an unsupported schema")
    if not bool(waterfall.get("invariants", {}).get("reconciled")):
        raise ValueError("--token-waterfall is not reconciled")
    if waterfall.get("inputs", {}).get("dedup_decisions_sha256") != sha256_file(args.dedup_decisions):
        raise ValueError("--token-waterfall is bound to different dedup decisions")

    args.output.mkdir(parents=True, exist_ok=True)
    training_root = args.output / "training" / "data"
    redistribution_root = args.output / "redistribution" / "data"
    training_columns = list(first_schema.names)
    redistribution_columns = [name for name in first_schema.names if name not in REDISTRIBUTION_DENY_COLUMNS]
    connection = duckdb.connect()
    configure_duckdb(
        connection,
        temporary_directory=args.temporary_directory / "materialize",
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    try:
        connection.execute(
            f"CREATE TEMP VIEW corpus AS SELECT * FROM read_parquet({sql_path_list(input_files)}, union_by_name=true)"
        )
        connection.execute(
            f"CREATE TEMP VIEW decisions AS SELECT * FROM read_parquet({sql_string(args.dedup_decisions.resolve())})"
        )
        gates = validate_inputs(connection)
        files: list[dict[str, Any]] = []
        totals: dict[str, int] = defaultdict(int)
        for input_path in input_files:
            relative = input_path.relative_to(args.input)
            training_path = training_root / relative
            redistribution_path = redistribution_root / relative
            training_receipt = copy_partition(
                connection,
                input_path=input_path,
                output_path=training_path,
                columns=training_columns,
                redistribution=False,
            )
            redistribution_receipt = copy_partition(
                connection,
                input_path=input_path,
                output_path=redistribution_path,
                columns=redistribution_columns,
                redistribution=True,
            )
            training_receipt["path"] = str(training_path.relative_to(args.output))
            redistribution_receipt["path"] = str(redistribution_path.relative_to(args.output))
            files.append(
                {
                    "input": str(input_path.resolve()),
                    "training": training_receipt,
                    "redistribution": redistribution_receipt,
                }
            )
            totals["training_rows"] += int(training_receipt["rows"])
            totals["redistribution_rows"] += int(redistribution_receipt["rows"])
    finally:
        connection.close()
    if totals["training_rows"] != gates["expected_training_rows"]:
        raise RuntimeError(f"materialized training rows do not match kept decisions: {dict(totals)} vs {gates}")
    if totals["redistribution_rows"] > totals["training_rows"]:
        raise RuntimeError("redistribution output cannot exceed training output")
    payload = {
        "schema_version": "full_cpt_release_manifest_v1",
        "completed_at": utc_now(),
        "input": str(args.input.resolve()),
        "dedup_decisions": str(args.dedup_decisions.resolve()),
        "dedup_decisions_sha256": sha256_file(args.dedup_decisions),
        "token_waterfall": str(args.token_waterfall.resolve()),
        "token_waterfall_sha256": sha256_file(args.token_waterfall),
        "output": str(args.output.resolve()),
        "training_root": "training/data",
        "redistribution_root": "redistribution/data",
        "redistribution_policy": {
            "requires_eligible_for_training": True,
            "requires_eligible_for_redistribution": True,
            "excluded_columns": sorted(REDISTRIBUTION_DENY_COLUMNS),
        },
        "input_gates": gates,
        "counts": dict(totals),
        "files": files,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": dict(totals)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

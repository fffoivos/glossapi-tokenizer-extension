#!/usr/bin/env python3
"""Run fail-closed integrity, provenance, eligibility and HF-readiness gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from finalization_io import (
    configure_duckdb,
    discover_parquet,
    read_json_object,
    sha256_file,
    sql_path_list,
    sql_string,
    utc_now,
    write_json_atomic,
)


PROVENANCE_COLUMNS = [
    "stable_uid",
    "source_dataset",
    "source_doc_id",
    "acquisition_source_id",
    "source_repo_id",
    "source_revision",
    "source_artifact_path",
    "source_row_id",
    "cleaned_text_sha256",
]
REDISTRIBUTION_FORBIDDEN_COLUMNS = {"author", "source_metadata_json"}


def check(name: str, passed: bool, observed: Any, expected: Any = 0) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "expected": expected}


def schema_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def main() -> int:
    import duckdb
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", "--input", dest="release", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dedup-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Immutable validation receipt")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--memory-limit", default="200GB")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable validation receipt: {args.output}")
    manifest = read_json_object(args.manifest)
    if manifest.get("schema_version") != "full_cpt_release_manifest_v1":
        raise ValueError("release manifest schema is unsupported")
    checks: list[dict[str, Any]] = []
    checks.append(check("manifest_output_binding", Path(str(manifest.get("output"))).resolve() == args.release.resolve(), manifest.get("output"), str(args.release.resolve())))
    checks.append(
        check(
            "dedup_decisions_binding",
            manifest.get("dedup_decisions_sha256") == sha256_file(args.dedup_decisions),
            manifest.get("dedup_decisions_sha256"),
            sha256_file(args.dedup_decisions),
        )
    )
    waterfall_path = Path(str(manifest.get("token_waterfall", "")))
    checks.append(
        check(
            "token_waterfall_checksum",
            waterfall_path.is_file() and sha256_file(waterfall_path) == manifest.get("token_waterfall_sha256"),
            sha256_file(waterfall_path) if waterfall_path.is_file() else "missing",
            manifest.get("token_waterfall_sha256"),
        )
    )
    partials = sorted(str(path.relative_to(args.release)) for path in args.release.rglob("*.partial"))
    checks.append(check("no_partial_files", not partials, partials, []))

    expected_training: set[str] = set()
    expected_redistribution: set[str] = set()
    checksum_failures: list[dict[str, Any]] = []
    row_count_failures: list[dict[str, Any]] = []
    for file_row in manifest.get("files", []):
        for bucket, expected in (("training", expected_training), ("redistribution", expected_redistribution)):
            receipt = file_row[bucket]
            relative = str(receipt["path"])
            expected.add(relative)
            path = args.release / relative
            if not path.is_file():
                checksum_failures.append({"path": relative, "error": "missing"})
                continue
            actual_hash = sha256_file(path)
            if actual_hash != receipt["sha256"]:
                checksum_failures.append({"path": relative, "expected": receipt["sha256"], "actual": actual_hash})
            actual_rows = pq.ParquetFile(path).metadata.num_rows
            if actual_rows != int(receipt["rows"]):
                row_count_failures.append({"path": relative, "expected": receipt["rows"], "actual": actual_rows})
    actual_training = {
        str(path.relative_to(args.release)) for path in discover_parquet(args.release / "training" / "data")
    }
    actual_redistribution = {
        str(path.relative_to(args.release)) for path in discover_parquet(args.release / "redistribution" / "data")
    }
    checks.append(check("manifest_file_inventory", actual_training == expected_training and actual_redistribution == expected_redistribution, {"training_extra": sorted(actual_training - expected_training), "training_missing": sorted(expected_training - actual_training), "redistribution_extra": sorted(actual_redistribution - expected_redistribution), "redistribution_missing": sorted(expected_redistribution - actual_redistribution)}, "exact inventory match"))
    checks.append(check("file_checksums", not checksum_failures, checksum_failures, []))
    checks.append(check("file_row_counts", not row_count_failures, row_count_failures, []))

    training_files = [args.release / value for value in sorted(actual_training)]
    redistribution_files = [args.release / value for value in sorted(actual_redistribution)]
    training_columns = schema_columns(training_files[0])
    redistribution_columns = schema_columns(redistribution_files[0])
    training_schema = pq.ParquetFile(training_files[0]).schema_arrow
    redistribution_schema = pq.ParquetFile(redistribution_files[0]).schema_arrow
    training_drift = [
        str(path.relative_to(args.release))
        for path in training_files[1:]
        if pq.ParquetFile(path).schema_arrow != training_schema
    ]
    redistribution_drift = [
        str(path.relative_to(args.release))
        for path in redistribution_files[1:]
        if pq.ParquetFile(path).schema_arrow != redistribution_schema
    ]
    checks.append(check("training_schema_consistency", not training_drift, training_drift, []))
    checks.append(check("redistribution_schema_consistency", not redistribution_drift, redistribution_drift, []))
    missing_provenance = sorted(set(PROVENANCE_COLUMNS) - training_columns)
    checks.append(check("training_provenance_schema", not missing_provenance, missing_provenance, []))
    missing_redis_provenance = sorted(set(PROVENANCE_COLUMNS) - redistribution_columns)
    checks.append(check("redistribution_provenance_schema", not missing_redis_provenance, missing_redis_provenance, []))
    forbidden = sorted(REDISTRIBUTION_FORBIDDEN_COLUMNS & redistribution_columns)
    checks.append(check("redistribution_safe_schema", not forbidden, forbidden, []))

    connection = duckdb.connect()
    configure_duckdb(connection, temporary_directory=args.temporary_directory, memory_limit=args.memory_limit, threads=args.threads)
    try:
        connection.execute(f"CREATE TEMP VIEW training AS SELECT * FROM read_parquet({sql_path_list(training_files)}, union_by_name=true)")
        connection.execute(f"CREATE TEMP VIEW redistribution AS SELECT * FROM read_parquet({sql_path_list(redistribution_files)}, union_by_name=true)")
        connection.execute(f"CREATE TEMP VIEW decisions AS SELECT * FROM read_parquet({sql_string(args.dedup_decisions.resolve())})")
        training_stats = connection.execute(
            """
            SELECT
              count(*) AS rows,
              count(*) - count(DISTINCT stable_uid) AS duplicate_uid,
              count(*) - count(DISTINCT cleaned_text_sha256) AS duplicate_hash,
              count(*) FILTER (WHERE text IS NULL OR trim(text) = '') AS empty_text,
              count(*) FILTER (WHERE cleaned_text_sha256 <> sha256(text)) AS bad_text_hash,
              count(*) FILTER (WHERE NOT eligible_for_training) AS ineligible,
              count(*) FILTER (WHERE source_artifact_path LIKE '/%') AS absolute_artifact_path
            FROM training
            """
        ).fetchone()
        redistribution_stats = connection.execute(
            """
            SELECT
              count(*) AS rows,
              count(*) - count(DISTINCT stable_uid) AS duplicate_uid,
              count(*) - count(DISTINCT cleaned_text_sha256) AS duplicate_hash,
              count(*) FILTER (WHERE text IS NULL OR trim(text) = '') AS empty_text,
              count(*) FILTER (WHERE cleaned_text_sha256 <> sha256(text)) AS bad_text_hash,
              count(*) FILTER (WHERE NOT eligible_for_training OR NOT eligible_for_redistribution) AS ineligible,
              count(*) FILTER (WHERE source_artifact_path LIKE '/%') AS absolute_artifact_path
            FROM redistribution
            """
        ).fetchone()
        for scope, values in (("training", training_stats), ("redistribution", redistribution_stats)):
            names = ["rows", "duplicate_stable_uid", "duplicate_cleaned_hash", "empty_text", "bad_text_hash", "ineligible", "absolute_artifact_path"]
            stats = dict(zip(names, map(int, values), strict=True))
            for key in names[1:]:
                checks.append(check(f"{scope}_{key}", stats[key] == 0, stats[key]))
        null_predicate = " OR ".join(f"{quote} IS NULL OR {quote} = ''" for quote in (f'\"{name}\"' for name in PROVENANCE_COLUMNS))
        training_missing = int(connection.execute(f"SELECT count(*) FROM training WHERE {null_predicate}").fetchone()[0])
        redistribution_missing = int(connection.execute(f"SELECT count(*) FROM redistribution WHERE {null_predicate}").fetchone()[0])
        checks.append(check("training_nonnull_provenance", training_missing == 0, training_missing))
        checks.append(check("redistribution_nonnull_provenance", redistribution_missing == 0, redistribution_missing))
        relation = connection.execute(
            """
            SELECT
              (SELECT count(*) FROM redistribution r LEFT JOIN training t USING (stable_uid)
                 WHERE t.stable_uid IS NULL) AS redistribution_not_training,
              (SELECT count(*) FROM training t LEFT JOIN decisions d
                 ON d.source_doc_id = t.stable_uid AND d.source_dataset = t.source_dataset
                 WHERE d.source_doc_id IS NULL OR d.decision <> 'keep') AS training_without_keep,
              (SELECT count(*) FROM decisions d LEFT JOIN training t
                 ON d.source_doc_id = t.stable_uid AND d.source_dataset = t.source_dataset
                 WHERE d.decision = 'keep' AND t.stable_uid IS NULL) AS keep_not_materialized,
              (SELECT count(*) FROM decisions WHERE decision = 'keep') AS expected_training
            """
        ).fetchone()
        checks.append(check("redistribution_subset_training", int(relation[0]) == 0, int(relation[0])))
        checks.append(check("training_rows_are_dedup_keeps", int(relation[1]) == 0, int(relation[1])))
        checks.append(check("all_dedup_keeps_materialized", int(relation[2]) == 0, int(relation[2])))
        checks.append(check("training_count_matches_dedup_keeps", int(training_stats[0]) == int(relation[3]), int(training_stats[0]), int(relation[3])))
        checks.append(check("manifest_training_count", int(training_stats[0]) == int(manifest["counts"]["training_rows"]), int(training_stats[0]), manifest["counts"]["training_rows"]))
        checks.append(check("manifest_redistribution_count", int(redistribution_stats[0]) == int(manifest["counts"]["redistribution_rows"]), int(redistribution_stats[0]), manifest["counts"]["redistribution_rows"]))
        decision_gates = connection.execute(
            """
            SELECT
              count(*) - count(DISTINCT source_doc_id) AS duplicate_uid,
              count(*) FILTER (WHERE decision NOT IN ('keep', 'drop')) AS bad_decision,
              count(*) FILTER (WHERE source_dataset IS NULL OR source_dataset = ''
                                  OR source_doc_id IS NULL OR source_doc_id = '') AS missing_identity
            FROM decisions
            """
        ).fetchone()
        checks.append(check("dedup_decision_unique_identity", int(decision_gates[0]) == 0, int(decision_gates[0])))
        checks.append(check("dedup_decision_values", int(decision_gates[1]) == 0, int(decision_gates[1])))
        checks.append(check("dedup_decision_provenance", int(decision_gates[2]) == 0, int(decision_gates[2])))
    finally:
        connection.close()

    failed = [row for row in checks if not row["passed"]]
    payload = {
        "schema_version": "full_cpt_release_validation_v1",
        "completed_at": utc_now(),
        "status": "passed" if not failed else "failed",
        "release": str(args.release.resolve()),
        "release_manifest": str(args.manifest.resolve()),
        "release_manifest_sha256": sha256_file(args.manifest),
        "dedup_decisions": str(args.dedup_decisions.resolve()),
        "checks": checks,
        "failed_checks": [row["name"] for row in failed],
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": not failed, "output": str(args.output), "failed_checks": payload["failed_checks"]}, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

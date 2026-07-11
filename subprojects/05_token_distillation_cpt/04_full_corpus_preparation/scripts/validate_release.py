#!/usr/bin/env python3
"""Run fail-closed release integrity, parity and publication-readiness gates."""

from __future__ import annotations

import argparse
import json
import os
import re
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
from materialize_release import (
    INTEGRITY_CONTRACT_VERSION,
    PUBLIC_ALWAYS_FORBIDDEN,
    PUBLIC_HASHED_COLUMNS,
    PUBLIC_METADATA_POLICY_VERSION,
    PUBLIC_SHARED_ALLOWLIST,
    verify_upstream_manifests,
)
from source_license import load_adjudication as load_license_adjudication


TRAINING_PROVENANCE_COLUMNS = {
    "stable_uid",
    "source_dataset",
    "source_doc_id",
    "acquisition_source_id",
    "source_repo_id",
    "source_revision",
    "source_artifact_path",
    "source_row_id",
    "cleaned_text_sha256",
}
PUBLIC_PROVENANCE_COLUMNS = {
    "stable_uid",
    "source_dataset",
    "acquisition_source_id",
    "source_repo_id",
    "source_revision",
    "cleaned_text_sha256",
    "source_doc_id_sha256",
    "source_artifact_path_sha256",
    "source_row_id_sha256",
}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def check(name: str, passed: bool, observed: Any, expected: Any = 0) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "expected": expected}


def schema_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def _safe_relative_path(value: object, *, prefix: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("release file receipt path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"release file receipt has unsafe path: {value!r}")
    if not value.startswith(prefix + "/") or path.suffix != ".parquet":
        raise ValueError(f"release file receipt is outside {prefix!r}: {value!r}")
    return value


def _data_tree_files(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    unsafe: list[str] = []
    if root.is_symlink():
        unsafe.append(str(root))
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            candidate = current_path / directory
            if candidate.is_symlink():
                unsafe.append(str(candidate))
        for name in names:
            candidate = current_path / name
            if candidate.is_symlink() or candidate.suffix != ".parquet" or name.startswith("."):
                unsafe.append(str(candidate))
            else:
                files.append(candidate)
    return sorted(files), sorted(unsafe)


def _hash_sql(source: str) -> str:
    domain = f"{PUBLIC_METADATA_POLICY_VERSION}:{source}:"
    quoted = f"t.{quote_identifier(source)}"
    return (
        f"CASE WHEN {quoted} IS NULL OR trim(CAST({quoted} AS VARCHAR)) = '' THEN NULL "
        f"ELSE sha256({sql_string(domain)} || CAST({quoted} AS VARCHAR)) END"
    )


def main() -> int:
    import duckdb
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", "--input", dest="release", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cleaning-manifest", type=Path, required=True)
    parser.add_argument("--decontamination-manifest", type=Path, required=True)
    parser.add_argument("--dedup-manifest", type=Path, required=True)
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
    if manifest.get("integrity_contract_version") != INTEGRITY_CONTRACT_VERSION:
        raise ValueError("release manifest lacks the current integrity contract")

    upstream = verify_upstream_manifests(
        cleaning_manifest_path=args.cleaning_manifest,
        decontamination_manifest_path=args.decontamination_manifest,
        dedup_manifest_path=args.dedup_manifest,
        input_root=Path(str(manifest.get("input", ""))),
        dedup_decisions=args.dedup_decisions,
    )
    checks: list[dict[str, Any]] = []
    cleaning_payload = read_json_object(args.cleaning_manifest)
    license_path = Path(str(cleaning_payload.get("license_adjudication", ""))).resolve()
    source_config_path = Path(str(cleaning_payload.get("source_config", ""))).resolve()
    license_decisions = load_license_adjudication(
        license_path, source_registry_path=source_config_path
    )
    expected_license_receipt = {
        "path": str(license_path),
        "sha256": sha256_file(license_path),
        "schema_version": "full_cpt_source_license_adjudication_v1",
        "status": "technical_audit_complete",
        "audited_at": read_json_object(license_path).get("audited_at"),
    }
    checks.append(
        check(
            "source_license_adjudication_binding",
            manifest.get("source_license_adjudication") == expected_license_receipt,
            manifest.get("source_license_adjudication"),
            expected_license_receipt,
        )
    )
    checks.append(
        check(
            "manifest_output_binding",
            Path(str(manifest.get("output"))).resolve() == args.release.resolve(),
            manifest.get("output"),
            str(args.release.resolve()),
        )
    )
    checks.append(
        check(
            "upstream_manifest_bindings",
            manifest.get("upstream_manifests") == upstream,
            manifest.get("upstream_manifests"),
            upstream,
        )
    )
    decisions_sha = sha256_file(args.dedup_decisions)
    checks.append(
        check(
            "dedup_decisions_binding",
            manifest.get("dedup_decisions_sha256") == decisions_sha,
            manifest.get("dedup_decisions_sha256"),
            decisions_sha,
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

    dataset_card_receipt = manifest.get("dataset_card")
    if not isinstance(dataset_card_receipt, dict):
        raise ValueError("release manifest lacks its dataset-card receipt")
    if (
        dataset_card_receipt.get("path") != "publication/README.md"
        or dataset_card_receipt.get("remote_path") != "README.md"
    ):
        raise ValueError("release manifest has an unsafe dataset-card path")
    dataset_card_path = args.release / "publication" / "README.md"
    actual_card = {
        "path": "publication/README.md",
        "remote_path": "README.md",
        "sha256": sha256_file(dataset_card_path) if dataset_card_path.is_file() else None,
        "bytes": dataset_card_path.stat().st_size if dataset_card_path.is_file() else -1,
    }
    checks.append(
        check(
            "dataset_card_checksum_binding",
            not dataset_card_path.is_symlink() and actual_card == dataset_card_receipt,
            actual_card,
            dataset_card_receipt,
        )
    )
    card_text = dataset_card_path.read_text(encoding="utf-8") if dataset_card_path.is_file() else ""
    required_card_notices = (
        "not** the full private training corpus",
        "license: other",
        "ShareAlike",
        "not legal advice",
        "GreekMMLU",
    )
    missing_card_notices = [notice for notice in required_card_notices if notice not in card_text]
    checks.append(check("dataset_card_required_notices", not missing_card_notices, missing_card_notices, []))
    publication_metadata_files = sorted(
        path.relative_to(args.release).as_posix()
        for path in (args.release / "publication").rglob("*")
        if path.is_file() or path.is_symlink()
    ) if (args.release / "publication").is_dir() else []
    checks.append(
        check(
            "dataset_card_exact_local_inventory",
            publication_metadata_files == ["publication/README.md"],
            publication_metadata_files,
            ["publication/README.md"],
        )
    )

    expected_training: set[str] = set()
    expected_redistribution: set[str] = set()
    checksum_failures: list[dict[str, Any]] = []
    row_count_failures: list[dict[str, Any]] = []
    publication_inventory: list[dict[str, Any]] = []
    file_rows = manifest.get("files")
    if not isinstance(file_rows, list) or not file_rows:
        raise ValueError("release manifest has no file inventory")
    for file_row in file_rows:
        if not isinstance(file_row, dict):
            raise ValueError("release manifest file inventory row is not an object")
        for bucket, expected, prefix in (
            ("training", expected_training, "training/data"),
            ("redistribution", expected_redistribution, "redistribution/data"),
        ):
            receipt = file_row.get(bucket)
            if not isinstance(receipt, dict):
                raise ValueError(f"release manifest file row lacks {bucket!r} receipt")
            relative = _safe_relative_path(receipt.get("path"), prefix=prefix)
            if relative in expected:
                raise ValueError(f"duplicate release inventory path: {relative}")
            expected.add(relative)
            path = args.release / relative
            if not path.is_file() or path.is_symlink():
                checksum_failures.append({"path": relative, "error": "missing_or_symlink"})
                continue
            actual_hash = sha256_file(path)
            actual_bytes = path.stat().st_size
            if actual_hash != receipt.get("sha256") or actual_bytes != int(receipt.get("bytes", -1)):
                checksum_failures.append(
                    {
                        "path": relative,
                        "expected_sha256": receipt.get("sha256"),
                        "actual_sha256": actual_hash,
                        "expected_bytes": receipt.get("bytes"),
                        "actual_bytes": actual_bytes,
                    }
                )
            actual_rows = pq.ParquetFile(path).metadata.num_rows
            if actual_rows != int(receipt.get("rows", -1)):
                row_count_failures.append(
                    {"path": relative, "expected": receipt.get("rows"), "actual": actual_rows}
                )
            if bucket == "redistribution":
                data_relative = Path(relative).relative_to("redistribution/data").as_posix()
                publication_inventory.append(
                    {
                        "path": data_relative,
                        "remote_path": f"data/{data_relative}",
                        "sha256": str(receipt.get("sha256")),
                        "bytes": int(receipt.get("bytes", -1)),
                        "rows": int(receipt.get("rows", -1)),
                    }
                )

    training_root = args.release / "training" / "data"
    redistribution_root = args.release / "redistribution" / "data"
    training_tree, training_unsafe = _data_tree_files(training_root)
    redistribution_tree, redistribution_unsafe = _data_tree_files(redistribution_root)
    actual_training = {str(path.relative_to(args.release)) for path in training_tree}
    actual_redistribution = {str(path.relative_to(args.release)) for path in redistribution_tree}
    checks.append(check("training_data_tree_safe", not training_unsafe, training_unsafe, []))
    checks.append(check("redistribution_data_tree_safe", not redistribution_unsafe, redistribution_unsafe, []))
    checks.append(
        check(
            "manifest_file_inventory",
            actual_training == expected_training and actual_redistribution == expected_redistribution,
            {
                "training_extra": sorted(actual_training - expected_training),
                "training_missing": sorted(expected_training - actual_training),
                "redistribution_extra": sorted(actual_redistribution - expected_redistribution),
                "redistribution_missing": sorted(expected_redistribution - actual_redistribution),
            },
            "exact inventory match",
        )
    )
    checks.append(check("file_checksums", not checksum_failures, checksum_failures, []))
    checks.append(check("file_row_counts", not row_count_failures, row_count_failures, []))

    expected_checkpoint_paths: set[Path] = set()
    checkpoint_failures: list[dict[str, Any]] = []
    for file_row in file_rows:
        training_receipt = file_row["training"]
        redistribution_receipt = file_row["redistribution"]
        source_relative = Path(str(training_receipt["path"])).relative_to("training/data")
        checkpoint = (
            args.release
            / ".materialization-checkpoints"
            / source_relative.parent
            / f"{source_relative.name}.json"
        )
        expected_checkpoint_paths.add(checkpoint.resolve())
        if not checkpoint.is_file() or checkpoint.is_symlink():
            checkpoint_failures.append({"path": str(checkpoint), "error": "missing_or_symlink"})
            continue
        try:
            checkpoint_value = read_json_object(checkpoint)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            checkpoint_failures.append({"path": str(checkpoint), "error": str(exc)})
            continue
        if (
            checkpoint_value.get("schema_version") != "full_cpt_materialization_checkpoint_v1"
            or not HEX_SHA256.fullmatch(str(checkpoint_value.get("contract_sha256", "")))
            or checkpoint_value.get("training") != training_receipt
            or checkpoint_value.get("redistribution") != redistribution_receipt
        ):
            checkpoint_failures.append({"path": str(checkpoint), "error": "receipt_or_contract_drift"})
    checks.append(check("materialization_checkpoints", not checkpoint_failures, checkpoint_failures, []))

    allowed_release_files = {
        *(path.resolve() for path in training_tree),
        *(path.resolve() for path in redistribution_tree),
        *expected_checkpoint_paths,
        dataset_card_path.resolve(),
    }
    unexpected_release_files: list[str] = []
    for path in args.release.rglob("*"):
        if path.is_symlink():
            unexpected_release_files.append(f"symlink:{path}")
        elif path.is_file() and path.resolve() not in allowed_release_files:
            unexpected_release_files.append(str(path))
    checks.append(
        check("release_root_exact_inventory", not unexpected_release_files, unexpected_release_files, [])
    )

    if not training_tree or not redistribution_tree:
        raise ValueError("release must contain materialized training and redistribution Parquet inventories")
    training_columns = schema_columns(training_tree[0])
    redistribution_columns = schema_columns(redistribution_tree[0])
    training_schema = pq.ParquetFile(training_tree[0]).schema_arrow
    redistribution_schema = pq.ParquetFile(redistribution_tree[0]).schema_arrow
    training_drift = [
        str(path.relative_to(args.release))
        for path in training_tree[1:]
        if pq.ParquetFile(path).schema_arrow != training_schema
    ]
    redistribution_drift = [
        str(path.relative_to(args.release))
        for path in redistribution_tree[1:]
        if pq.ParquetFile(path).schema_arrow != redistribution_schema
    ]
    checks.append(check("training_schema_consistency", not training_drift, training_drift, []))
    checks.append(check("redistribution_schema_consistency", not redistribution_drift, redistribution_drift, []))
    missing_provenance = sorted(TRAINING_PROVENANCE_COLUMNS - training_columns)
    checks.append(check("training_provenance_schema", not missing_provenance, missing_provenance, []))
    missing_public_provenance = sorted(PUBLIC_PROVENANCE_COLUMNS - redistribution_columns)
    checks.append(
        check("redistribution_provenance_schema", not missing_public_provenance, missing_public_provenance, [])
    )

    policy = manifest.get("redistribution_policy")
    if not isinstance(policy, dict) or policy.get("policy_version") != PUBLIC_METADATA_POLICY_VERSION:
        raise ValueError("release manifest has an unsupported public metadata policy")
    if policy.get("mode") != "explicit_allowlist_with_domain_separated_hashes":
        raise ValueError("release manifest public metadata policy mode mismatch")
    expected_shared = [name for name in PUBLIC_SHARED_ALLOWLIST if name in training_columns]
    expected_shared.extend(sorted(name for name in training_columns if name.startswith("dedup_")))
    expected_hashed = {
        source: output for source, output in PUBLIC_HASHED_COLUMNS.items() if source in training_columns
    }
    expected_public_columns = [*expected_shared, *expected_hashed.values()]
    expected_dropped = sorted(set(training_columns) - set(expected_shared) - set(expected_hashed))
    policy_mismatch = {
        "shared_columns": policy.get("shared_columns") != expected_shared,
        "hashed_columns": policy.get("hashed_columns") != expected_hashed,
        "public_columns": policy.get("public_columns") != expected_public_columns,
        "dropped_columns": policy.get("dropped_columns") != expected_dropped,
        "always_forbidden_columns": policy.get("always_forbidden_columns")
        != sorted(PUBLIC_ALWAYS_FORBIDDEN),
    }
    checks.append(
        check(
            "redistribution_policy_identity",
            not any(policy_mismatch.values()),
            {key: value for key, value in policy_mismatch.items() if value},
            {},
        )
    )
    checks.append(
        check(
            "redistribution_exact_public_schema",
            redistribution_columns == set(expected_public_columns),
            {
                "extra": sorted(redistribution_columns - set(expected_public_columns)),
                "missing": sorted(set(expected_public_columns) - redistribution_columns),
            },
            {"extra": [], "missing": []},
        )
    )
    forbidden = sorted(PUBLIC_ALWAYS_FORBIDDEN & redistribution_columns)
    checks.append(check("redistribution_safe_schema", not forbidden, forbidden, []))

    input_root = Path(str(manifest["input"]))
    input_files = discover_parquet(input_root)
    connection = duckdb.connect()
    configure_duckdb(
        connection,
        temporary_directory=args.temporary_directory,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    try:
        connection.execute(
            f"CREATE TEMP VIEW training AS SELECT * FROM read_parquet({sql_path_list(training_tree)}, union_by_name=true)"
        )
        connection.execute(
            f"CREATE TEMP VIEW redistribution AS SELECT * FROM read_parquet({sql_path_list(redistribution_tree)}, "
            "union_by_name=true)"
        )
        connection.execute(
            f"CREATE TEMP VIEW corpus AS SELECT * FROM read_parquet({sql_path_list(input_files)}, union_by_name=true)"
        )
        connection.execute(
            f"CREATE TEMP VIEW decisions AS SELECT * FROM read_parquet({sql_string(args.dedup_decisions.resolve())})"
        )
        observed_training_sources = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT acquisition_source_id FROM training"
            ).fetchall()
        }
        observed_redistribution_sources = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT acquisition_source_id FROM redistribution"
            ).fetchall()
        }
        disallowed_training_sources = sorted(
            source_id
            for source_id in observed_training_sources
            if not license_decisions.get(source_id, {}).get("training_eligible", False)
        )
        disallowed_redistribution_sources = sorted(
            source_id
            for source_id in observed_redistribution_sources
            if not license_decisions.get(source_id, {}).get("redistribution_eligible", False)
        )
        checks.append(
            check(
                "training_sources_match_license_adjudication",
                not disallowed_training_sources,
                disallowed_training_sources,
                [],
            )
        )
        checks.append(
            check(
                "redistribution_sources_match_license_adjudication",
                not disallowed_redistribution_sources,
                disallowed_redistribution_sources,
                [],
            )
        )
        actual_redistribution_sources = [
            {
                "source_id": str(row[0]),
                "repo_id": str(row[1]),
                "revision": str(row[2]),
                "rows": int(row[3]),
            }
            for row in connection.execute(
                """
                SELECT acquisition_source_id, source_repo_id, source_revision, count(*)
                FROM redistribution
                GROUP BY acquisition_source_id, source_repo_id, source_revision
                ORDER BY acquisition_source_id, source_repo_id, source_revision
                """
            ).fetchall()
        ]
        checks.append(
            check(
                "redistribution_source_counts_and_provenance",
                manifest.get("redistribution_sources") == actual_redistribution_sources,
                actual_redistribution_sources,
                manifest.get("redistribution_sources"),
            )
        )
        missing_card_sources = [
            row["source_id"]
            for row in actual_redistribution_sources
            if f"`{row['source_id']}`" not in card_text
        ]
        checks.append(
            check(
                "dataset_card_source_attribution_coverage",
                not missing_card_sources,
                missing_card_sources,
                [],
            )
        )
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
              count(*) FILTER (WHERE NOT eligible_for_training OR NOT eligible_for_redistribution) AS ineligible
            FROM redistribution
            """
        ).fetchone()
        for scope, values, names in (
            (
                "training",
                training_stats,
                [
                    "rows",
                    "duplicate_stable_uid",
                    "duplicate_cleaned_hash",
                    "empty_text",
                    "bad_text_hash",
                    "ineligible",
                    "absolute_artifact_path",
                ],
            ),
            (
                "redistribution",
                redistribution_stats,
                [
                    "rows",
                    "duplicate_stable_uid",
                    "duplicate_cleaned_hash",
                    "empty_text",
                    "bad_text_hash",
                    "ineligible",
                ],
            ),
        ):
            stats = dict(zip(names, map(int, values), strict=True))
            for key in names[1:]:
                checks.append(check(f"{scope}_{key}", stats[key] == 0, stats[key]))

        training_null_predicate = " OR ".join(
            f"{quote_identifier(name)} IS NULL OR CAST({quote_identifier(name)} AS VARCHAR) = ''"
            for name in sorted(TRAINING_PROVENANCE_COLUMNS)
        )
        public_required = PUBLIC_PROVENANCE_COLUMNS & redistribution_columns
        redistribution_null_predicate = " OR ".join(
            f"{quote_identifier(name)} IS NULL OR CAST({quote_identifier(name)} AS VARCHAR) = ''"
            for name in sorted(public_required)
        )
        training_missing = int(
            connection.execute(f"SELECT count(*) FROM training WHERE {training_null_predicate}").fetchone()[0]
        )
        redistribution_missing = int(
            connection.execute(
                f"SELECT count(*) FROM redistribution WHERE {redistribution_null_predicate}"
            ).fetchone()[0]
        )
        checks.append(check("training_nonnull_provenance", training_missing == 0, training_missing))
        checks.append(
            check("redistribution_nonnull_provenance", redistribution_missing == 0, redistribution_missing)
        )

        relation = connection.execute(
            """
            SELECT
              (SELECT count(*) FROM training t LEFT JOIN decisions d
                 ON d.source_doc_id = t.stable_uid AND d.source_dataset = t.source_dataset
                 WHERE d.source_doc_id IS NULL OR d.decision <> 'keep'
                    OR d.input_text_sha256 <> t.cleaned_text_sha256) AS training_without_bound_keep,
              (SELECT count(*) FROM decisions d LEFT JOIN corpus c
                 ON d.source_doc_id = c.stable_uid AND d.source_dataset = c.source_dataset
                 AND c.eligible_for_training
                 WHERE c.stable_uid IS NULL) AS decision_without_eligible_input,
              (SELECT count(*) FROM corpus c LEFT JOIN decisions d
                 ON d.source_doc_id = c.stable_uid AND d.source_dataset = c.source_dataset
                 WHERE c.eligible_for_training AND d.source_doc_id IS NULL) AS eligible_input_without_decision,
              (SELECT count(*) FROM decisions d JOIN corpus c
                 ON d.source_doc_id = c.stable_uid AND d.source_dataset = c.source_dataset
                 WHERE d.input_text_sha256 <> c.cleaned_text_sha256
                    OR c.cleaned_text_sha256 <> sha256(c.text)) AS decision_content_mismatch,
              (SELECT count(*) FROM decisions d LEFT JOIN training t
                 ON d.source_doc_id = t.stable_uid AND d.source_dataset = t.source_dataset
                 WHERE d.decision = 'keep' AND t.stable_uid IS NULL) AS keep_not_materialized,
              (SELECT count(*) FROM decisions WHERE decision = 'keep') AS expected_training
            """
        ).fetchone()
        checks.append(check("training_rows_are_content_bound_dedup_keeps", int(relation[0]) == 0, int(relation[0])))
        checks.append(check("dedup_decisions_cover_eligible_input", int(relation[1]) == 0, int(relation[1])))
        checks.append(check("eligible_input_has_dedup_decision", int(relation[2]) == 0, int(relation[2])))
        checks.append(check("dedup_decision_content_hash_binding", int(relation[3]) == 0, int(relation[3])))
        checks.append(check("all_dedup_keeps_materialized", int(relation[4]) == 0, int(relation[4])))
        checks.append(
            check(
                "training_count_matches_dedup_keeps",
                int(training_stats[0]) == int(relation[5]),
                int(training_stats[0]),
                int(relation[5]),
            )
        )

        redis_relation = connection.execute(
            """
            SELECT
              (SELECT count(*) FROM training t LEFT JOIN redistribution r
                 ON r.stable_uid = t.stable_uid AND r.source_dataset = t.source_dataset
                 WHERE t.eligible_for_redistribution AND r.stable_uid IS NULL)
                AS eligible_training_not_redistributed,
              (SELECT count(*) FROM redistribution r LEFT JOIN training t
                 ON r.stable_uid = t.stable_uid AND r.source_dataset = t.source_dataset
                 AND t.eligible_for_redistribution
                 WHERE t.stable_uid IS NULL) AS redistribution_without_eligible_training,
              (SELECT count(*) FROM training WHERE eligible_for_redistribution) AS expected_redistribution
            """
        ).fetchone()
        checks.append(
            check("all_redistribution_eligible_training_rows_present", int(redis_relation[0]) == 0, int(redis_relation[0]))
        )
        checks.append(
            check("redistribution_has_only_eligible_training_rows", int(redis_relation[1]) == 0, int(redis_relation[1]))
        )
        checks.append(
            check(
                "redistribution_expected_count",
                int(redistribution_stats[0]) == int(redis_relation[2]),
                int(redistribution_stats[0]),
                int(redis_relation[2]),
            )
        )

        content_columns = [name for name in ("text", "cleaned_text_sha256") if name in expected_shared]
        provenance_columns = [
            name
            for name in expected_shared
            if name not in {*content_columns, "eligible_for_training", "eligible_for_redistribution"}
        ]
        content_predicate = " OR ".join(
            f"t.{quote_identifier(name)} IS DISTINCT FROM r.{quote_identifier(name)}" for name in content_columns
        ) or "FALSE"
        provenance_predicate = " OR ".join(
            f"t.{quote_identifier(name)} IS DISTINCT FROM r.{quote_identifier(name)}"
            for name in provenance_columns
        ) or "FALSE"
        content_mismatch = int(
            connection.execute(
                "SELECT count(*) FROM training t JOIN redistribution r "
                "ON r.stable_uid=t.stable_uid AND r.source_dataset=t.source_dataset "
                f"WHERE {content_predicate}"
            ).fetchone()[0]
        )
        provenance_mismatch = int(
            connection.execute(
                "SELECT count(*) FROM training t JOIN redistribution r "
                "ON r.stable_uid=t.stable_uid AND r.source_dataset=t.source_dataset "
                f"WHERE {provenance_predicate}"
            ).fetchone()[0]
        )
        checks.append(check("redistribution_content_parity", content_mismatch == 0, content_mismatch))
        checks.append(check("redistribution_provenance_parity", provenance_mismatch == 0, provenance_mismatch))

        hash_predicate = " OR ".join(
            f"r.{quote_identifier(output)} IS DISTINCT FROM {_hash_sql(source)}"
            for source, output in expected_hashed.items()
        ) or "FALSE"
        hashed_mismatch = int(
            connection.execute(
                "SELECT count(*) FROM training t JOIN redistribution r "
                "ON r.stable_uid=t.stable_uid AND r.source_dataset=t.source_dataset "
                f"WHERE {hash_predicate}"
            ).fetchone()[0]
        )
        checks.append(check("redistribution_hashed_metadata_parity", hashed_mismatch == 0, hashed_mismatch))

        checks.append(
            check(
                "manifest_training_count",
                int(training_stats[0]) == int(manifest["counts"]["training_rows"]),
                int(training_stats[0]),
                manifest["counts"]["training_rows"],
            )
        )
        checks.append(
            check(
                "manifest_redistribution_count",
                int(redistribution_stats[0]) == int(manifest["counts"]["redistribution_rows"]),
                int(redistribution_stats[0]),
                manifest["counts"]["redistribution_rows"],
            )
        )
        decision_gates = connection.execute(
            """
            SELECT
              count(*) - count(DISTINCT (source_dataset, source_doc_id)) AS duplicate_identity,
              count(*) FILTER (WHERE decision NOT IN ('keep', 'drop')) AS bad_decision,
              count(*) FILTER (WHERE source_dataset IS NULL OR source_dataset = ''
                                  OR source_doc_id IS NULL OR source_doc_id = ''
                                  OR stable_uid IS NULL OR stable_uid = '' OR stable_uid <> source_doc_id
                                  OR input_text_sha256 IS NULL OR input_text_sha256 = '') AS missing_identity
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
        "integrity_contract_version": INTEGRITY_CONTRACT_VERSION,
        "completed_at": utc_now(),
        "status": "passed" if not failed else "failed",
        "release": str(args.release.resolve()),
        "release_manifest": str(args.manifest.resolve()),
        "release_manifest_sha256": sha256_file(args.manifest),
        "upstream_manifests": upstream,
        "source_license_adjudication": expected_license_receipt,
        "dedup_decisions": str(args.dedup_decisions.resolve()),
        "dedup_decisions_sha256": decisions_sha,
        "publication_inventory": {
            "root": str(redistribution_root.resolve()),
            "files": sorted(publication_inventory, key=lambda row: row["path"]),
            "bytes": sum(int(row["bytes"]) for row in publication_inventory),
            "rows": sum(int(row["rows"]) for row in publication_inventory),
        },
        "publication_metadata_inventory": [dataset_card_receipt],
        "checks": checks,
        "failed_checks": [row["name"] for row in failed],
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {"ok": not failed, "output": str(args.output), "failed_checks": payload["failed_checks"]},
            sort_keys=True,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

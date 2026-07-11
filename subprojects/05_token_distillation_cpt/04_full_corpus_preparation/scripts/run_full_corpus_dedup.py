#!/usr/bin/env python3
"""Stage canonical CPT rows and run the established repository deduplicator.

The wrapper owns schema adaptation, restart-safe staging receipts, the frozen
production recipe, and content binding.  Duplicate detection remains entirely
in ``glossapi_corpus_cli.text_dedup`` and is launched through a dependency-light
invoker instead of the broad Typer CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
from full_corpus_dedup_recipe import (
    APPROVED_PRODUCTION_RECIPE,
    PRODUCTION_RECIPE_ID,
    validate_recipe_parameters,
)


MANIFEST_SCHEMA_VERSION = "full_cpt_dedup_wrapper_manifest_v1"
STAGING_CONTRACT_VERSION = "full_cpt_dedup_staging_v2_content_hash_verified"
CONTENT_BOUND_DECISIONS_VERSION = "full_cpt_dedup_decisions_content_bound_v1"
STAGING_PROGRESS_VERSION = "full_cpt_dedup_staging_progress_v1"
MAX_STAGING_WORKERS = 16
DEFAULT_STAGING_WORKERS = min(8, max(1, os.cpu_count() or 1))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def staged_schema(input_schema: Any) -> Any:
    import pyarrow as pa

    names = set(input_schema.names)
    required = {
        "source_dataset",
        "source_doc_id",
        "stable_uid",
        "text",
        "cleaned_text_sha256",
        "eligible_for_training",
        "title",
        "author",
        "greek_badness_score",
        "mojibake_badness_score",
        "needs_ocr",
        "is_empty",
        "ocr_success",
        "is_historical_or_polytonic",
    }
    missing = required - names
    if missing:
        raise ValueError(f"input schema lacks dedup-required columns: {sorted(missing)}")
    if "upstream_source_doc_id" in names:
        raise ValueError("input already contains upstream_source_doc_id; refusing ambiguous restaging")
    fields: list[Any] = []
    for field in input_schema:
        if field.name == "source_doc_id":
            fields.append(pa.field("source_doc_id", pa.string(), nullable=False))
            fields.append(pa.field("upstream_source_doc_id", field.type, nullable=field.nullable))
        else:
            fields.append(field)
    return pa.schema(fields, metadata=input_schema.metadata)


def _verify_cleaned_text_hashes(table: Any) -> None:
    texts = table["text"].to_pylist()
    claimed_hashes = table["cleaned_text_sha256"].to_pylist()
    for row_index, (text, claimed_hash) in enumerate(zip(texts, claimed_hashes, strict=True)):
        if not isinstance(text, str):
            raise ValueError(f"eligible staged row {row_index} has non-string text")
        observed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if claimed_hash != observed_hash:
            raise ValueError(
                "cleaned-text hash drift before dedup staging: "
                f"row={row_index}, claimed={claimed_hash!r}, observed={observed_hash}"
            )


def adapt_batch(batch: Any, output_schema: Any) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    table = pa.Table.from_batches([batch])
    training_mask = pc.fill_null(pc.equal(table["eligible_for_training"], True), False)  # noqa: E712
    table = table.filter(training_mask)
    _verify_cleaned_text_hashes(table)
    arrays = []
    for field in output_schema:
        if field.name == "source_doc_id":
            arrays.append(table["stable_uid"])
        elif field.name == "upstream_source_doc_id":
            arrays.append(table["source_doc_id"])
        else:
            arrays.append(table[field.name])
    return pa.Table.from_arrays(arrays, schema=output_schema)


def stage_file(input_path: Path, input_root: Path, staged_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    relative = input_path.relative_to(input_root)
    output_path = staged_root / relative
    parquet = pq.ParquetFile(input_path)
    schema = staged_schema(parquet.schema_arrow)
    temporary = atomic_output_path(output_path)
    # A killed worker may leave only its private temporary.  It is never a
    # committed artifact and is safe to replace on the next attempt.
    temporary.unlink(missing_ok=True)
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    input_rows = 0
    staged_rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=2048, use_threads=False):
            input_rows += batch.num_rows
            adapted = adapt_batch(batch, schema)
            if adapted.num_rows:
                writer.write_table(adapted)
                staged_rows += adapted.num_rows
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(temporary, output_path)
    return {
        "input": str(input_path.resolve()),
        "relative_path": relative.as_posix(),
        "input_rows": input_rows,
        "training_eligible_rows": staged_rows,
        "text_hash_verified_rows": staged_rows,
        "input_receipt": parquet_file_receipt(input_path, relative_to=input_root),
        "staged": parquet_file_receipt(output_path, relative_to=staged_root),
    }


def stage_files(
    input_files: list[Path],
    *,
    input_root: Path,
    staged_root: Path,
    staging_workers: int,
) -> tuple[list[dict[str, Any]], int]:
    effective_workers = min(staging_workers, len(input_files))
    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="dedup-stage") as executor:
        receipts = list(
            executor.map(
                lambda path: stage_file(path, input_root, staged_root),
                input_files,
            )
        )
    return receipts, effective_workers


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _progress_receipt_path(progress_root: Path, relative_path: str) -> Path:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
    return progress_root / "files" / f"{digest}.json"


def _validate_progress_file(
    value: dict[str, Any],
    *,
    contract_sha256: str,
    input_root: Path,
    staged_root: Path,
) -> dict[str, Any]:
    if value.get("schema_version") != STAGING_PROGRESS_VERSION:
        raise ValueError("partial staging receipt has an unsupported schema")
    if value.get("contract_sha256") != contract_sha256:
        raise ValueError("partial staging receipt belongs to a different staging contract")
    row = value.get("file")
    if not isinstance(row, dict):
        raise ValueError("partial staging receipt lacks a file record")
    relative = str(row.get("relative_path", ""))
    if not relative or Path(relative).is_absolute():
        raise ValueError("partial staging receipt has an invalid relative path")
    input_receipt = row.get("input_receipt")
    staged_receipt = row.get("staged")
    if not isinstance(input_receipt, dict) or not isinstance(staged_receipt, dict):
        raise ValueError("partial staging receipt lacks content receipts")
    if str(input_receipt.get("path")) != relative or str(staged_receipt.get("path")) != relative:
        raise ValueError("partial staging receipt path disagreement")
    _validate_file_receipt(_resolve_receipt_path(input_root, input_receipt), input_receipt)
    _validate_file_receipt(_resolve_receipt_path(staged_root, staged_receipt), staged_receipt)
    return dict(row)


def stage_files_resumable(
    input_files: list[Path],
    *,
    input_root: Path,
    staged_root: Path,
    staging_workers: int,
    progress_root: Path,
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    relative_inputs = {
        path.relative_to(input_root).as_posix(): path for path in input_files
    }
    contract_payload = {
        "schema_version": STAGING_PROGRESS_VERSION,
        **contract,
        "input_relative_paths": sorted(relative_inputs),
    }
    contract_sha256 = _canonical_sha256(contract_payload)
    contract_document = {**contract_payload, "contract_sha256": contract_sha256}
    contract_path = progress_root / "contract.json"
    if contract_path.exists():
        if read_json_object(contract_path) != contract_document:
            raise ValueError("partial staging progress belongs to a different invocation")
    else:
        write_json_atomic(contract_path, contract_document)

    completed: dict[str, dict[str, Any]] = {}
    receipts_root = progress_root / "files"
    if receipts_root.exists():
        for receipt_path in sorted(receipts_root.glob("*.json")):
            row = _validate_progress_file(
                read_json_object(receipt_path),
                contract_sha256=contract_sha256,
                input_root=input_root,
                staged_root=staged_root,
            )
            relative = str(row["relative_path"])
            if relative in completed:
                raise ValueError(f"duplicate partial staging receipt for {relative}")
            completed[relative] = row

    actual_staged = {
        path.relative_to(staged_root).as_posix(): path
        for path in discover_parquet(staged_root)
    } if any(staged_root.rglob("*.parquet")) else {}
    unexpected_receipts = set(completed) - set(relative_inputs)
    if unexpected_receipts:
        raise ValueError(
            f"partial staging receipts reference removed inputs: {sorted(unexpected_receipts)[:20]}"
        )
    # The output rename happens immediately before its receipt is committed.
    # If killed in that tiny window, discard only that unreceipted generated
    # output and reproduce it from the still-receipted canonical input.
    for relative in sorted(set(actual_staged) - set(completed)):
        if relative not in relative_inputs:
            raise ValueError(f"unreceipted staged output has no matching input: {relative}")
        actual_staged[relative].unlink()
    if set(completed) - set(actual_staged):
        raise ValueError("partial staging receipt exists without its committed Parquet output")

    missing = [path for relative, path in relative_inputs.items() if relative not in completed]
    effective_workers = min(staging_workers, len(input_files))
    if missing:
        with ThreadPoolExecutor(
            max_workers=min(staging_workers, len(missing)),
            thread_name_prefix="dedup-stage",
        ) as executor:
            futures = {
                executor.submit(stage_file, path, input_root, staged_root): path
                for path in missing
            }
            for future in as_completed(futures):
                row = future.result()
                relative = str(row["relative_path"])
                receipt_path = _progress_receipt_path(progress_root, relative)
                write_json_atomic(
                    receipt_path,
                    {
                        "schema_version": STAGING_PROGRESS_VERSION,
                        "contract_sha256": contract_sha256,
                        "file": row,
                    },
                )
                completed[relative] = row
    if set(completed) != set(relative_inputs):
        raise ValueError("partial staging did not cover the exact input inventory")
    return [completed[key] for key in sorted(completed)], effective_workers


def validate_staged_identity(
    staged_files: list[Path],
    temporary_directory: Path,
    memory_limit: str,
    threads: int,
) -> dict[str, int]:
    import duckdb

    connection = duckdb.connect()
    configure_duckdb(
        connection,
        temporary_directory=temporary_directory,
        memory_limit=memory_limit,
        threads=threads,
    )
    try:
        paths = sql_path_list(staged_files)
        row = connection.execute(
            f"""
            SELECT
              count(*) AS rows,
              count(*) FILTER (
                WHERE source_doc_id IS NULL OR source_doc_id = ''
                   OR stable_uid IS NULL OR stable_uid = ''
              ) AS missing_identity,
              count(*) - count(DISTINCT source_doc_id) AS duplicate_stable_uid,
              count(*) FILTER (WHERE source_doc_id IS DISTINCT FROM stable_uid) AS identity_drift,
              count(*) FILTER (
                WHERE cleaned_text_sha256 IS NULL
                   OR NOT regexp_full_match(cleaned_text_sha256, '[0-9a-f]{{64}}')
              ) AS invalid_cleaned_text_sha256
            FROM read_parquet({paths})
            """
        ).fetchone()
    finally:
        connection.close()
    result = {
        "rows": int(row[0]),
        "missing_identity": int(row[1]),
        "duplicate_stable_uid": int(row[2]),
        "identity_drift": int(row[3]),
        "invalid_cleaned_text_sha256": int(row[4]),
    }
    failure_keys = (
        "missing_identity",
        "duplicate_stable_uid",
        "identity_drift",
        "invalid_cleaned_text_sha256",
    )
    if any(result[key] for key in failure_keys):
        raise ValueError(f"staged dedup identity gate failed: {result}")
    return result


def _resolve_receipt_path(root: Path, receipt: dict[str, Any]) -> Path:
    relative = Path(str(receipt["path"]))
    if relative.is_absolute():
        raise ValueError(f"staging receipt path must be relative: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"staging receipt escapes its root: {relative}")
    return path


def _validate_file_receipt(path: Path, receipt: dict[str, Any]) -> None:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(f"staging receipt file is missing: {path}")
    if path.stat().st_size != int(receipt["bytes"]):
        raise ValueError(f"staging receipt byte-size mismatch: {path}")
    if sha256_file(path) != str(receipt["sha256"]):
        raise ValueError(f"staging receipt SHA256 mismatch: {path}")
    metadata = pq.ParquetFile(path).metadata
    if metadata.num_rows != int(receipt["rows"]):
        raise ValueError(f"staging receipt row-count mismatch: {path}")
    if metadata.num_row_groups != int(receipt["row_groups"]):
        raise ValueError(f"staging receipt row-group mismatch: {path}")


def validate_staging_receipts(
    files: list[dict[str, Any]],
    *,
    input_root: Path,
    staged_root: Path,
    workers: int,
) -> None:
    expected_input: dict[str, tuple[Path, dict[str, Any]]] = {}
    expected_staged: dict[str, tuple[Path, dict[str, Any]]] = {}
    for row in files:
        relative_path = str(row["relative_path"])
        input_receipt = dict(row["input_receipt"])
        staged_receipt = dict(row["staged"])
        if str(input_receipt["path"]) != relative_path or str(staged_receipt["path"]) != relative_path:
            raise ValueError(f"staging receipt relative-path disagreement: {relative_path}")
        expected_input[relative_path] = (
            _resolve_receipt_path(input_root, input_receipt),
            input_receipt,
        )
        expected_staged[relative_path] = (
            _resolve_receipt_path(staged_root, staged_receipt),
            staged_receipt,
        )
    if len(expected_input) != len(files) or len(expected_staged) != len(files):
        raise ValueError("staging receipt contains duplicate relative paths")

    actual_input = {path.relative_to(input_root).as_posix() for path in discover_parquet(input_root)}
    actual_staged = {path.relative_to(staged_root).as_posix() for path in discover_parquet(staged_root)}
    if actual_input != set(expected_input):
        raise ValueError("input Parquet inventory differs from the staging receipt")
    if actual_staged != set(expected_staged):
        raise ValueError("staged Parquet inventory differs from the staging receipt")

    work = [*expected_input.values(), *expected_staged.values()]
    effective_workers = min(max(1, workers), len(work))
    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="dedup-receipt") as executor:
        list(executor.map(lambda item: _validate_file_receipt(*item), work))


def recipe_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in APPROVED_PRODUCTION_RECIPE}


def validate_recipe(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    parameters = recipe_parameters(args)
    mode = validate_recipe_parameters(
        parameters,
        experimental=args.experimental_parameters,
    )
    return mode, parameters


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _decision_coverage(
    connection: Any,
    *,
    staged_files: list[Path],
    decisions: Path,
) -> dict[str, int]:
    staged_paths = sql_path_list(staged_files)
    decisions_path = sql_string(decisions.resolve())
    row = connection.execute(
        f"""
        WITH staged AS (
          SELECT source_doc_id, stable_uid, source_dataset, cleaned_text_sha256
          FROM read_parquet({staged_paths})
        ), decisions AS (
          SELECT source_doc_id, source_dataset
          FROM read_parquet({decisions_path})
        )
        SELECT
          (SELECT count(*) FROM staged) AS staged_rows,
          (SELECT count(*) FROM decisions) AS decision_rows,
          (SELECT count(*) - count(DISTINCT source_doc_id) FROM decisions) AS duplicate_decisions,
          (SELECT count(*) FROM staged s LEFT JOIN decisions d USING (source_doc_id)
             WHERE d.source_doc_id IS NULL) AS missing_decisions,
          (SELECT count(*) FROM decisions d LEFT JOIN staged s USING (source_doc_id)
             WHERE s.source_doc_id IS NULL) AS decisions_without_input,
          (SELECT count(*) FROM decisions d JOIN staged s USING (source_doc_id)
             WHERE d.source_dataset IS DISTINCT FROM s.source_dataset) AS source_dataset_drift,
          (SELECT count(*) FROM staged
             WHERE source_doc_id IS DISTINCT FROM stable_uid) AS identity_drift
        """
    ).fetchone()
    result = {
        "staged_rows": int(row[0]),
        "decision_rows": int(row[1]),
        "duplicate_decisions": int(row[2]),
        "missing_decisions": int(row[3]),
        "decisions_without_input": int(row[4]),
        "source_dataset_drift": int(row[5]),
        "identity_drift": int(row[6]),
    }
    failure_keys = (
        "duplicate_decisions",
        "missing_decisions",
        "decisions_without_input",
        "source_dataset_drift",
        "identity_drift",
    )
    if any(result[key] for key in failure_keys):
        raise ValueError(f"dedup decision coverage/content-binding gate failed: {result}")
    return result


def _validate_existing_bound_decisions(
    connection: Any,
    *,
    bound_decisions: Path,
    raw_decisions: Path,
    staged_files: list[Path],
) -> None:
    import pyarrow.parquet as pq

    raw_columns = pq.ParquetFile(raw_decisions).schema_arrow.names
    bound_columns = pq.ParquetFile(bound_decisions).schema_arrow.names
    expected_columns = [*raw_columns, "stable_uid", "input_text_sha256"]
    if bound_columns != expected_columns:
        raise ValueError(
            f"existing content-bound decisions schema mismatch: {bound_columns} != {expected_columns}"
        )
    projection = ", ".join(_sql_identifier(column) for column in raw_columns)
    bound_path = sql_string(bound_decisions.resolve())
    raw_path = sql_string(raw_decisions.resolve())
    staged_paths = sql_path_list(staged_files)
    mismatch_row = connection.execute(
        f"""
        SELECT
          (SELECT count(*) FROM (
            SELECT {projection} FROM read_parquet({bound_path})
            EXCEPT ALL
            SELECT {projection} FROM read_parquet({raw_path})
          ))
          +
          (SELECT count(*) FROM (
            SELECT {projection} FROM read_parquet({raw_path})
            EXCEPT ALL
            SELECT {projection} FROM read_parquet({bound_path})
          )) AS raw_decision_drift,
          (SELECT count(*)
             FROM read_parquet({bound_path}) b
             LEFT JOIN read_parquet({staged_paths}) s ON s.source_doc_id = b.stable_uid
            WHERE s.source_doc_id IS NULL
               OR b.stable_uid IS DISTINCT FROM b.source_doc_id
               OR b.input_text_sha256 IS DISTINCT FROM s.cleaned_text_sha256) AS binding_drift
        """
    ).fetchone()
    if int(mismatch_row[0]) or int(mismatch_row[1]):
        raise ValueError(
            "existing content-bound decisions failed verification: "
            f"raw_decision_drift={int(mismatch_row[0])}, binding_drift={int(mismatch_row[1])}"
        )


def build_content_bound_decisions(
    *,
    staged_files: list[Path],
    raw_decisions: Path,
    output: Path,
    temporary_directory: Path,
    memory_limit: str,
    threads: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    import duckdb
    import pyarrow.parquet as pq

    raw_columns = pq.ParquetFile(raw_decisions).schema_arrow.names
    collisions = {"stable_uid", "input_text_sha256"}.intersection(raw_columns)
    if collisions:
        raise ValueError(f"raw decisions already contain reserved binding columns: {sorted(collisions)}")

    connection = duckdb.connect()
    configure_duckdb(
        connection,
        temporary_directory=temporary_directory,
        memory_limit=memory_limit,
        threads=threads,
    )
    try:
        coverage = _decision_coverage(
            connection,
            staged_files=staged_files,
            decisions=raw_decisions,
        )
        if output.exists():
            _validate_existing_bound_decisions(
                connection,
                bound_decisions=output,
                raw_decisions=raw_decisions,
                staged_files=staged_files,
            )
        else:
            temporary = atomic_output_path(output)
            staged_paths = sql_path_list(staged_files)
            raw_path = sql_string(raw_decisions.resolve())
            try:
                connection.execute(
                    f"""
                    COPY (
                      SELECT
                        d.*,
                        d.source_doc_id AS stable_uid,
                        s.cleaned_text_sha256 AS input_text_sha256
                      FROM read_parquet({raw_path}) d
                      JOIN read_parquet({staged_paths}) s
                        ON s.source_doc_id = d.source_doc_id
                      ORDER BY d.source_doc_id
                    ) TO {sql_string(temporary)}
                    (FORMAT PARQUET, COMPRESSION ZSTD)
                    """
                )
                os.replace(temporary, output)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            _validate_existing_bound_decisions(
                connection,
                bound_decisions=output,
                raw_decisions=raw_decisions,
                staged_files=staged_files,
            )
    finally:
        connection.close()
    receipt = parquet_file_receipt(output)
    receipt["schema_version"] = CONTENT_BOUND_DECISIONS_VERSION
    return receipt, coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Decontaminated canonical Parquet root")
    parser.add_argument("--staged-input", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--memory-limit", default="200GB")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--duckdb-threads", type=int)
    parser.add_argument("--staging-workers", type=int, default=DEFAULT_STAGING_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-staged",
        action="store_true",
        help="Reuse a verified stage-only receipt when no dedup run has started",
    )
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--experimental-parameters", action="store_true")
    parser.add_argument("--greek-diacritic-policy", choices=["preserve", "strip"], default="preserve")
    parser.add_argument("--minhash-threshold", type=float, default=0.85)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("--rows-per-band", type=int, default=4)
    parser.add_argument("--shingle-mode", choices=["token", "char"], default="token")
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--max-bucket-size", type=int, default=5000)
    return parser.parse_args()


def _resolved_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "input",
        "staged_input",
        "state_root",
        "run_root",
        "manifest",
        "temporary_directory",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest_path_gate(manifest: dict[str, Any], key: str, expected: Path) -> None:
    if Path(str(manifest.get(key, ""))).resolve() != expected:
        raise ValueError(f"staging manifest {key} does not match this invocation")


def validate_resumable_manifest(
    manifest: dict[str, Any],
    *,
    args: argparse.Namespace,
    parameters: dict[str, Any],
    dedup_source: Path,
    invoker: Path,
    recipe_contract: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("staging manifest has an unsupported schema_version")
    if manifest.get("status") != "staged":
        raise FileExistsError("only a status=staged manifest may be resumed")
    if manifest.get("staging_contract", {}).get("version") != STAGING_CONTRACT_VERSION:
        raise ValueError("staging manifest predates the content-verified staging contract")
    for key, expected in (
        ("input", args.input),
        ("staged_input", args.staged_input),
        ("state_root", args.state_root),
        ("run_root", args.run_root),
    ):
        _manifest_path_gate(manifest, key, expected)
    if manifest.get("dedup_parameters") != parameters:
        raise ValueError("dedup parameters differ from the staged manifest")
    expected_recipe_mode = "experimental" if args.experimental_parameters else "production"
    if manifest.get("recipe", {}).get("mode") != expected_recipe_mode:
        raise ValueError("dedup recipe mode differs from the staged manifest")
    if int(manifest.get("resources", {}).get("dedup_workers", 0)) != args.workers:
        raise ValueError("--workers differs from the staged manifest/run configuration")
    implementation = manifest.get("dedup_implementation", {})
    if implementation.get("sha256") != sha256_file(dedup_source):
        raise ValueError("dedup implementation changed since staging; refusing unsafe resume")
    if implementation.get("invoker_sha256") != sha256_file(invoker):
        raise ValueError("dedup invoker changed since staging; refusing unsafe resume")
    if implementation.get("recipe_contract_sha256") != sha256_file(recipe_contract):
        raise ValueError("dedup recipe contract changed since staging; refusing unsafe resume")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("staging manifest has no file receipts")
    validate_staging_receipts(
        files,
        input_root=args.input,
        staged_root=args.staged_input,
        workers=args.staging_workers,
    )
    staged_files = discover_parquet(args.staged_input)
    identity = validate_staged_identity(
        staged_files,
        temporary_directory=args.temporary_directory,
        memory_limit=args.memory_limit,
        threads=args.duckdb_threads,
    )
    if manifest.get("identity_contract", {}).get("rows") != identity["rows"]:
        raise ValueError("staged identity row count differs from the staging manifest")
    return [dict(row) for row in files], identity


def main() -> int:
    args = _resolved_args(parse_args())
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.duckdb_threads is None:
        args.duckdb_threads = min(args.workers, 32)
    if args.duckdb_threads < 1:
        raise ValueError("--duckdb-threads must be >= 1")
    if not 1 <= args.staging_workers <= MAX_STAGING_WORKERS:
        raise ValueError(f"--staging-workers must be between 1 and {MAX_STAGING_WORKERS}")
    if args.resume and args.reuse_staged:
        raise ValueError("--resume and --reuse-staged are mutually exclusive")
    recipe_mode, parameters = validate_recipe(args)

    root = repo_root()
    dedup_source = root / "glossapi_corpus_cli" / "text_dedup.py"
    invoker = Path(__file__).with_name("invoke_text_dedup.py")
    recipe_contract = Path(__file__).with_name("full_corpus_dedup_recipe.py")
    if not dedup_source.is_file():
        raise FileNotFoundError(f"existing dedup implementation is missing: {dedup_source}")
    if not invoker.is_file():
        raise FileNotFoundError(f"lightweight dedup invoker is missing: {invoker}")
    if not recipe_contract.is_file():
        raise FileNotFoundError(f"dedup recipe contract is missing: {recipe_contract}")

    existing_manifest: dict[str, Any] | None = None
    staging_manifest_sha256: str | None = None
    if args.manifest.exists():
        if not (args.resume or args.reuse_staged):
            raise FileExistsError(f"refusing to overwrite existing manifest: {args.manifest}")
        existing_manifest = read_json_object(args.manifest)
        staging_manifest_sha256 = sha256_file(args.manifest)
        receipts, identity = validate_resumable_manifest(
            existing_manifest,
            args=args,
            parameters=parameters,
            dedup_source=dedup_source,
            invoker=invoker,
            recipe_contract=recipe_contract,
        )
        effective_staging_workers = int(
            existing_manifest["staging_contract"]["effective_workers"]
        )
    else:
        if args.resume or args.reuse_staged:
            raise FileNotFoundError("a status=staged receipt manifest is required to reuse staging")
        args.staged_input.mkdir(parents=True, exist_ok=True)
        input_files = discover_parquet(args.input)
        progress_root = args.manifest.parent / f".{args.manifest.name}.staging-progress"
        receipts, effective_staging_workers = stage_files_resumable(
            input_files,
            input_root=args.input,
            staged_root=args.staged_input,
            staging_workers=args.staging_workers,
            progress_root=progress_root,
            contract={
                "staging_contract_version": STAGING_CONTRACT_VERSION,
                "input": str(args.input),
                "staged_input": str(args.staged_input),
                "wrapper_sha256": sha256_file(Path(__file__).resolve()),
                "recipe_mode": recipe_mode,
                "dedup_parameters": parameters,
            },
        )
        staged_files = discover_parquet(args.staged_input)
        identity = validate_staged_identity(
            staged_files,
            temporary_directory=args.temporary_directory,
            memory_limit=args.memory_limit,
            threads=args.duckdb_threads,
        )

    command = [
        sys.executable,
        str(invoker.resolve()),
        "--input-root",
        str(args.staged_input),
        "--state-root",
        str(args.state_root),
        "--run-root",
        str(args.run_root),
        "--max-workers",
        str(args.workers),
        "--temporary-directory",
        str(args.temporary_directory),
        "--memory-limit",
        args.memory_limit,
        "--duckdb-threads",
        str(args.duckdb_threads),
        "--greek-diacritic-policy",
        args.greek_diacritic_policy,
        "--minhash-threshold",
        str(args.minhash_threshold),
        "--num-perm",
        str(args.num_perm),
        "--bands",
        str(args.bands),
        "--rows-per-band",
        str(args.rows_per_band),
        "--shingle-mode",
        args.shingle_mode,
        "--shingle-size",
        str(args.shingle_size),
        "--max-bucket-size",
        str(args.max_bucket_size),
    ]
    if args.resume:
        command.append("--resume")
    if args.experimental_parameters:
        command.append("--experimental-parameters")

    totals: dict[str, int] = defaultdict(int)
    for receipt in receipts:
        totals["input_rows"] += int(receipt["input_rows"])
        totals["training_eligible_rows"] += int(receipt["training_eligible_rows"])
        totals["text_hash_verified_rows"] += int(receipt["text_hash_verified_rows"])

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_revision": "v2_content_bound_restart_safe",
        "staged_at": (
            existing_manifest.get("staged_at")
            if existing_manifest is not None
            else utc_now()
        ),
        "completed_at": None,
        "status": "staged",
        "input": str(args.input),
        "staged_input": str(args.staged_input),
        "state_root": str(args.state_root),
        "run_root": str(args.run_root),
        "identity_contract": {
            "dedup_source_dataset": "source_dataset (unchanged)",
            "dedup_source_doc_id": "stable_uid",
            "upstream_source_doc_id": "source_doc_id before staging",
            "input_text_sha256": "verified canonical cleaned_text_sha256",
            **identity,
        },
        "staging_contract": {
            "version": STAGING_CONTRACT_VERSION,
            "executor": "bounded_thread_pool_per_parquet_file",
            "requested_workers": args.staging_workers,
            "effective_workers": effective_staging_workers,
            "max_workers": MAX_STAGING_WORKERS,
            "receipt_validation": "sha256+bytes+rows+row_groups+exact_inventory",
        },
        "dedup_implementation": {
            "path": str(dedup_source.resolve()),
            "sha256": sha256_file(dedup_source),
            "invoker_path": str(invoker.resolve()),
            "invoker_sha256": sha256_file(invoker),
            "recipe_contract_path": str(recipe_contract.resolve()),
            "recipe_contract_sha256": sha256_file(recipe_contract),
            "git_commit": _git_commit(root),
            "reimplemented": False,
        },
        "recipe": {
            "id": PRODUCTION_RECIPE_ID,
            "mode": recipe_mode,
            "approved_production_parameters": APPROVED_PRODUCTION_RECIPE,
        },
        "dedup_parameters": parameters,
        "resources": {
            "dedup_workers": args.workers,
            "duckdb_threads": args.duckdb_threads,
            "duckdb_memory_limit": args.memory_limit,
            "duckdb_temporary_directory": str(args.temporary_directory),
        },
        "command": command,
        "counts": dict(totals),
        "files": receipts,
        "dedup_output": None,
    }

    if existing_manifest is None:
        write_json_atomic(args.manifest, payload)
        staging_manifest_sha256 = sha256_file(args.manifest)
    if args.stage_only:
        print(json.dumps({"ok": True, "status": "staged", "manifest": str(args.manifest)}, sort_keys=True))
        return 0

    try:
        completed_process = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stderr.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise
    try:
        invoker_result = json.loads(completed_process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("lightweight dedup invoker returned invalid JSON") from exc
    if not isinstance(invoker_result, dict) or not invoker_result.get("run_id"):
        raise RuntimeError("lightweight dedup invoker result lacks run_id")
    raw_decisions = args.run_root / "final" / "dedup_decisions.parquet"
    summary = args.run_root / "final" / "run_summary.json"
    if not raw_decisions.is_file() or not summary.is_file():
        raise RuntimeError("existing dedup implementation returned without final decisions/summary")
    bound_decisions = args.run_root / "final" / "dedup_decisions_content_bound.parquet"
    bound_receipt, coverage = build_content_bound_decisions(
        staged_files=discover_parquet(args.staged_input),
        raw_decisions=raw_decisions,
        output=bound_decisions,
        temporary_directory=args.temporary_directory,
        memory_limit=args.memory_limit,
        threads=args.duckdb_threads,
    )
    payload["status"] = "completed"
    payload["completed_at"] = utc_now()
    payload["staging_manifest_sha256"] = staging_manifest_sha256
    payload["dedup_output"] = {
        "raw_decisions": parquet_file_receipt(raw_decisions),
        "content_bound_decisions": bound_receipt,
        "decisions": bound_receipt,
        "content_binding": {
            "schema_version": CONTENT_BOUND_DECISIONS_VERSION,
            "stable_uid_column": "stable_uid",
            "input_text_sha256_column": "input_text_sha256",
            **coverage,
        },
        "summary_path": str(summary.resolve()),
        "summary_sha256": sha256_file(summary),
        "summary_bytes": summary.stat().st_size,
        "invoker_run_id": str(invoker_result["run_id"]),
        "invoker_result_sha256": hashlib.sha256(
            completed_process.stdout.encode("utf-8")
        ).hexdigest(),
    }
    write_json_atomic(args.manifest, payload, immutable=False)
    print(json.dumps({"ok": True, "status": "completed", "manifest": str(args.manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

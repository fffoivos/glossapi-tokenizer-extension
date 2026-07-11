#!/usr/bin/env python3
"""Stage canonical CPT rows and invoke the repository's existing dedup CLI.

This file intentionally contains no duplicate-detection implementation.  It
adapts the full-corpus schema to ``glossapi_corpus_cli.text_dedup`` by using the
canonical ``stable_uid`` as that CLI's ``source_doc_id``.  The upstream ID is
retained as ``upstream_source_doc_id`` and all other provenance columns pass
through unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from finalization_io import (
    atomic_output_path,
    configure_duckdb,
    discover_parquet,
    parquet_file_receipt,
    sha256_file,
    sql_path_list,
    utc_now,
    write_json_atomic,
)


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


def adapt_batch(batch: Any, output_schema: Any) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    table = pa.Table.from_batches([batch])
    training_mask = pc.fill_null(pc.equal(table["eligible_for_training"], True), False)  # noqa: E712
    table = table.filter(training_mask)
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
        "staged": parquet_file_receipt(output_path, relative_to=staged_root),
    }


def validate_staged_identity(staged_files: list[Path], temporary_directory: Path, memory_limit: str, threads: int) -> dict[str, int]:
    import duckdb

    connection = duckdb.connect()
    configure_duckdb(connection, temporary_directory=temporary_directory, memory_limit=memory_limit, threads=threads)
    try:
        paths = sql_path_list(staged_files)
        row = connection.execute(
            f"""
            SELECT
              count(*) AS rows,
              count(*) FILTER (WHERE source_doc_id IS NULL OR source_doc_id = '') AS missing_identity,
              count(*) - count(DISTINCT source_doc_id) AS duplicate_stable_uid,
              count(*) FILTER (WHERE source_doc_id <> stable_uid) AS identity_drift
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
    }
    if any(result[key] for key in ("missing_identity", "duplicate_stable_uid", "identity_drift")):
        raise ValueError(f"staged dedup identity gate failed: {result}")
    return result


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--greek-diacritic-policy", choices=["preserve", "strip"], default="preserve")
    parser.add_argument("--minhash-threshold", type=float, default=0.85)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("--rows-per-band", type=int, default=4)
    parser.add_argument("--shingle-mode", choices=["token", "char"], default="token")
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--max-bucket-size", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {args.manifest}")
    if args.staged_input.exists() and any(args.staged_input.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty staged input: {args.staged_input}")
    args.staged_input.mkdir(parents=True, exist_ok=True)
    input_files = discover_parquet(args.input)
    receipts = [stage_file(path, args.input, args.staged_input) for path in input_files]
    staged_files = discover_parquet(args.staged_input)
    identity = validate_staged_identity(
        staged_files,
        temporary_directory=args.temporary_directory,
        memory_limit=args.memory_limit,
        threads=min(args.workers, 32),
    )
    root = repo_root()
    dedup_source = root / "glossapi_corpus_cli" / "text_dedup.py"
    if not dedup_source.is_file():
        raise FileNotFoundError(f"existing dedup implementation is missing: {dedup_source}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    command = [
        sys.executable,
        "-m",
        "glossapi_corpus_cli.cli",
        "dedup-text",
        "run",
        "--input-root",
        str(args.staged_input.resolve()),
        "--state-root",
        str(args.state_root.resolve()),
        "--run-root",
        str(args.run_root.resolve()),
        "--max-workers",
        str(args.workers),
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
    status = "staged"
    dedup_output: dict[str, Any] | None = None
    if not args.stage_only:
        subprocess.run(command, cwd=root, check=True)
        decisions = args.run_root / "final" / "dedup_decisions.parquet"
        summary = args.run_root / "final" / "run_summary.json"
        if not decisions.is_file() or not summary.is_file():
            raise RuntimeError("existing dedup CLI returned without its final decisions/summary")
        dedup_output = {
            "decisions": parquet_file_receipt(decisions),
            "summary_path": str(summary.resolve()),
            "summary_sha256": sha256_file(summary),
        }
        status = "completed"
    totals: dict[str, int] = defaultdict(int)
    for receipt in receipts:
        totals["input_rows"] += receipt["input_rows"]
        totals["training_eligible_rows"] += receipt["training_eligible_rows"]
    payload = {
        "schema_version": "full_cpt_dedup_wrapper_manifest_v1",
        "completed_at": utc_now(),
        "status": status,
        "input": str(args.input.resolve()),
        "staged_input": str(args.staged_input.resolve()),
        "state_root": str(args.state_root.resolve()),
        "run_root": str(args.run_root.resolve()),
        "identity_contract": {
            "dedup_source_dataset": "source_dataset (unchanged)",
            "dedup_source_doc_id": "stable_uid",
            "upstream_source_doc_id": "source_doc_id before staging",
            **identity,
        },
        "dedup_implementation": {
            "path": str(dedup_source.resolve()),
            "sha256": sha256_file(dedup_source),
            "git_commit": commit,
            "reimplemented": False,
        },
        "dedup_parameters": {
            "greek_diacritic_policy": args.greek_diacritic_policy,
            "minhash_threshold": args.minhash_threshold,
            "num_perm": args.num_perm,
            "bands": args.bands,
            "rows_per_band": args.rows_per_band,
            "shingle_mode": args.shingle_mode,
            "shingle_size": args.shingle_size,
            "max_bucket_size": args.max_bucket_size,
        },
        "command": command,
        "counts": dict(totals),
        "files": receipts,
        "dedup_output": dedup_output,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "status": status, "manifest": str(args.manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

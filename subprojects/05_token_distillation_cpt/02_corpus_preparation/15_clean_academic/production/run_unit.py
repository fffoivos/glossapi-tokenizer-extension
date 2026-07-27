#!/usr/bin/env python3
"""Score one receipt-bound work unit; optionally emit a cleaned apply fragment."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import (
    LEDGER_SCHEMA,
    RECEIPT_SCHEMA,
    atomic_write_json,
    load_json,
    require_schema,
    sha256_file,
    validate_plan,
)

METHOD = "glossapi_rs_bib/heading_lexgate@0.9"
SIZE_COLUMNS = ("chars", "non_whitespace_chars", "utf8_bytes", "approx_word_count")
WORD_RE = re.compile(r"\w+", re.UNICODE)
SPAN_TYPE = pa.struct(
    [
        ("line_start", pa.int64()),
        ("line_end", pa.int64()),
        ("char_start", pa.int64()),
        ("char_end", pa.int64()),
        ("content_chars", pa.int64()),
        ("removed_sha256", pa.string()),
        ("head", pa.string()),
    ]
)
LEDGER_ARROW_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("run_id", pa.string()),
        ("unit_id", pa.string()),
        ("rank", pa.int32()),
        ("row_group", pa.int32()),
        ("row_in_group", pa.int32()),
        ("source_dataset", pa.string()),
        ("source_doc_id", pa.string()),
        ("original_sha256", pa.string()),
        ("chars_before", pa.int64()),
        ("chars_after", pa.int64()),
        ("content_chars_removed", pa.int64()),
        ("separator_chars_removed", pa.int64()),
        ("total_chars_removed", pa.int64()),
        ("removed_fraction", pa.float64()),
        ("lines_before", pa.int64()),
        ("lines_removed", pa.int64()),
        ("had_removal", pa.bool_()),
        ("would_empty", pa.bool_()),
        ("over_50pct", pa.bool_()),
        ("spans", pa.list_(SPAN_TYPE)),
    ]
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _atomic_parquet_path(target: Path) -> tuple[Path, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary, descriptor


def _load_cleaner(glossapi_src: str | None, model_path: Path):
    if glossapi_src:
        sys.path.insert(0, glossapi_src)
    from glossapi.bibliography import BibliographyCleaner

    return BibliographyCleaner(model_path=model_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract, plan, contract_sha, plan_sha = validate_plan(args.contract, args.plan)
    units = {unit["unit_id"]: unit for unit in plan["units"]}
    try:
        unit = units[args.unit_id]
    except KeyError as error:
        raise ValueError(f"unknown unit {args.unit_id}") from error
    if args.mode != contract["mode"]:
        raise ValueError(f"requested mode {args.mode} differs from contract mode")
    if args.mode == "apply" and not unit["apply"]:
        raise ValueError(f"{args.unit_id} is outside the frozen apply scope")

    run_root = Path(contract["run_root"])
    receipt_path = run_root / args.mode / "receipts" / f"{args.unit_id}.json"
    ledger_path = run_root / args.mode / "ledger" / f"{args.unit_id}.parquet"
    output_path = run_root / args.mode / "fragments" / f"{args.unit_id}.parquet"
    lock_path = run_root / args.mode / "locks" / f"{args.unit_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent submissions must not score or publish the same unit twice. The lock
    # is released automatically on normal return, exception, SIGTERM, or process
    # death; a waiter then validates and reuses the first receipt.
    unit_lock = lock_path.open("a")
    fcntl.flock(unit_lock, fcntl.LOCK_EX)
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        require_schema(receipt, RECEIPT_SCHEMA, receipt_path)
        if (
            receipt["contract_sha256"] != contract_sha
            or receipt["plan_sha256"] != plan_sha
            or receipt["unit_id"] != args.unit_id
            or sha256_file(receipt["ledger"]["path"]) != receipt["ledger"]["sha256"]
        ):
            raise ValueError(f"invalid existing completion receipt {receipt_path}")
        print(json.dumps({"status": "passed", "reused": True, "unit_id": args.unit_id}))
        return receipt

    source = Path(unit["source_path"])
    if (
        source.is_symlink()
        or not source.is_file()
        or source.stat().st_size != unit["source_bytes"]
    ):
        raise ValueError(f"source shard identity check failed: {source}")
    model_path = Path(contract["model_artifacts"]["path"]).parent
    cleaner = _load_cleaner(args.glossapi_src, model_path)
    parquet = pq.ParquetFile(source)
    groups = range(unit["row_group_start"], unit["row_group_end"])
    expected_rows = sum(parquet.metadata.row_group(group).num_rows for group in groups)
    if expected_rows != unit["rows"]:
        raise ValueError(f"row-group rows {expected_rows} != plan rows {unit['rows']}")

    temporary_ledger, _ = _atomic_parquet_path(ledger_path)
    temporary_output = None
    ledger_writer = pq.ParquetWriter(
        temporary_ledger, LEDGER_ARROW_SCHEMA, compression="zstd"
    )
    output_writer = None
    if args.mode == "apply":
        temporary_output, _ = _atomic_parquet_path(output_path)
        output_writer = pq.ParquetWriter(
            temporary_output, parquet.schema_arrow, compression="zstd"
        )

    stats = {
        "docs": 0,
        "docs_cleaned": 0,
        "chars_before": 0,
        "content_chars_removed": 0,
        "separator_chars_removed": 0,
        "total_chars_removed": 0,
        "lines_before": 0,
        "lines_removed": 0,
        "spans": 0,
        "would_empty": 0,
        "docs_over_30pct": 0,
        "docs_over_50pct": 0,
    }
    fractions: list[float] = []
    started = time.monotonic()
    try:
        for row_group in groups:
            row_in_group = 0
            table = parquet.read_row_group(row_group)
            for batch in table.to_batches(max_chunksize=args.batch_size):
                rows = batch.to_pylist()
                texts = [row["text"] or "" for row in rows]
                cleaned_batch = cleaner.clean_batch(texts, num_threads=args.threads)
                ledger_rows = []
                for row, original, (cleaned, spans) in zip(rows, texts, cleaned_batch):
                    source_dataset = row.get("source_dataset")
                    source_doc_id = row.get("source_doc_id")
                    if (
                        source_dataset != unit["source_dataset"]
                        or source_doc_id is None
                    ):
                        raise ValueError(
                            "source identity is absent or differs from the plan"
                        )
                    content_removed = sum(span.char_count for span in spans)
                    total_removed = len(original) - len(cleaned)
                    separator_removed = total_removed - content_removed
                    if total_removed < 0 or separator_removed < 0:
                        raise ValueError(
                            f"character waterfall failed for {source_doc_id}"
                        )
                    if cleaned != "\n".join(
                        line
                        for index, line in enumerate(original.split("\n"))
                        if not any(
                            span.line_start <= index < span.line_end for span in spans
                        )
                    ):
                        raise ValueError(
                            f"retained-line identity failed for {source_doc_id}"
                        )
                    lines_before = len(original.split("\n")) if original else 0
                    lines_removed = sum(span.line_count for span in spans)
                    fraction = total_removed / len(original) if original else 0.0
                    would_empty = bool(original.strip()) and not bool(cleaned.strip())
                    ledger_rows.append(
                        {
                            "schema_version": LEDGER_SCHEMA,
                            "run_id": contract["run_id"],
                            "unit_id": args.unit_id,
                            "rank": unit["rank"],
                            "row_group": row_group,
                            "row_in_group": row_in_group,
                            "source_dataset": source_dataset,
                            "source_doc_id": str(source_doc_id),
                            "original_sha256": hashlib.sha256(
                                original.encode("utf-8")
                            ).hexdigest(),
                            "chars_before": len(original),
                            "chars_after": len(cleaned),
                            "content_chars_removed": content_removed,
                            "separator_chars_removed": separator_removed,
                            "total_chars_removed": total_removed,
                            "removed_fraction": fraction,
                            "lines_before": lines_before,
                            "lines_removed": lines_removed,
                            "had_removal": bool(spans),
                            "would_empty": would_empty,
                            "over_50pct": fraction > 0.5,
                            "spans": [
                                {
                                    "line_start": span.line_start,
                                    "line_end": span.line_end,
                                    "char_start": span.char_start,
                                    "char_end": span.char_end,
                                    "content_chars": span.char_count,
                                    "removed_sha256": hashlib.sha256(
                                        span.text_from(original).encode("utf-8")
                                    ).hexdigest(),
                                    "head": span.text_from(original)[:160].replace(
                                        "\n", " "
                                    ),
                                }
                                for span in spans
                            ],
                        }
                    )
                    stats["docs"] += 1
                    stats["docs_cleaned"] += int(bool(spans))
                    stats["chars_before"] += len(original)
                    stats["content_chars_removed"] += content_removed
                    stats["separator_chars_removed"] += separator_removed
                    stats["total_chars_removed"] += total_removed
                    stats["lines_before"] += lines_before
                    stats["lines_removed"] += lines_removed
                    stats["spans"] += len(spans)
                    stats["would_empty"] += int(would_empty)
                    stats["docs_over_30pct"] += int(fraction > 0.3)
                    stats["docs_over_50pct"] += int(fraction > 0.5)
                    if spans:
                        fractions.append(fraction)
                    if args.mode == "apply":
                        if would_empty:
                            raise ValueError(
                                f"apply would empty {source_dataset}/{source_doc_id}"
                            )
                        row["text"] = cleaned
                        replacements = {
                            "chars": len(cleaned),
                            "non_whitespace_chars": sum(
                                not char.isspace() for char in cleaned
                            ),
                            "utf8_bytes": len(cleaned.encode("utf-8")),
                            "approx_word_count": len(WORD_RE.findall(cleaned)),
                        }
                        for column in SIZE_COLUMNS:
                            if column in row and row[column] is not None:
                                row[column] = replacements[column]
                    row_in_group += 1
                ledger_writer.write_table(
                    pa.Table.from_pylist(ledger_rows, schema=LEDGER_ARROW_SCHEMA)
                )
                if output_writer is not None:
                    output_writer.write_table(
                        pa.Table.from_pylist(rows, schema=parquet.schema_arrow)
                    )
        ledger_writer.close()
        ledger_writer = None
        if output_writer is not None:
            output_writer.close()
            output_writer = None
        if stats["docs"] != expected_rows:
            raise ValueError(f"read {stats['docs']} rows, expected {expected_rows}")
        ledger_check = pq.ParquetFile(temporary_ledger)
        if ledger_check.metadata.num_rows != expected_rows:
            raise ValueError("document-ledger row closure failed")
        os.replace(temporary_ledger, ledger_path)
        if temporary_output is not None:
            output_check = pq.ParquetFile(temporary_output)
            if (
                output_check.metadata.num_rows != expected_rows
                or output_check.schema_arrow != parquet.schema_arrow
            ):
                raise ValueError("apply fragment row/schema closure failed")
            os.replace(temporary_output, output_path)
    except BaseException:
        if ledger_writer is not None:
            ledger_writer.close()
        if output_writer is not None:
            output_writer.close()
        temporary_ledger.unlink(missing_ok=True)
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        raise

    elapsed = time.monotonic() - started
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed",
        "mode": args.mode,
        "run_id": contract["run_id"],
        "unit_id": args.unit_id,
        "rank": unit["rank"],
        "source_dataset": unit["source_dataset"],
        "method": METHOD,
        "contract_sha256": contract_sha,
        "plan_sha256": plan_sha,
        "input": {
            "path": str(source),
            "bytes": unit["source_bytes"],
            "sha256": unit["source_sha256"],
            "row_group_start": unit["row_group_start"],
            "row_group_end": unit["row_group_end"],
            "rows": expected_rows,
        },
        "code": contract["code"],
        "model_artifacts": contract["model_artifacts"],
        **stats,
        "removed_fraction_p50": percentile(fractions, 0.5),
        "removed_fraction_p95": percentile(fractions, 0.95),
        "elapsed_seconds": elapsed,
        "docs_per_second": stats["docs"] / elapsed if elapsed else 0.0,
        "ledger": {
            "path": str(ledger_path),
            "rows": expected_rows,
            "bytes": ledger_path.stat().st_size,
            "sha256": sha256_file(ledger_path),
        },
    }
    if args.mode == "apply":
        receipt["output"] = {
            "path": str(output_path),
            "rows": expected_rows,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        }
    atomic_write_json(receipt_path, receipt)
    print(json.dumps({"status": "passed", "unit_id": args.unit_id, **stats}))
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--mode", choices=("dry-run", "apply"), required=True)
    parser.add_argument("--glossapi-src")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Validate a complete run and aggregate exact per-document ledger statistics."""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .contracts import (
    RECEIPT_SCHEMA,
    SUMMARY_SCHEMA,
    atomic_write_json,
    load_json,
    require_schema,
    sha256_file,
    validate_plan,
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _new_stats() -> dict[str, Any]:
    return {
        "docs": 0,
        "docs_cleaned": 0,
        "chars_before": 0,
        "total_chars_removed": 0,
        "lines_before": 0,
        "lines_removed": 0,
        "spans": 0,
        "would_empty": 0,
        "docs_over_30pct": 0,
        "docs_over_50pct": 0,
        "fractions": [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract, plan, contract_sha, plan_sha = validate_plan(args.contract, args.plan)
    receipt_dir = Path(contract["run_root"]) / contract["mode"] / "receipts"
    expected = {unit["unit_id"] for unit in plan["units"]}
    actual = {path.stem for path in receipt_dir.glob("*.json")}
    if actual != expected:
        raise ValueError(
            f"receipt set is not exact: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )

    by_source: dict[str, dict[str, Any]] = defaultdict(_new_stats)
    overall = _new_stats()
    identities: set[tuple[int, str]] = set()
    receipt_hashes = {}
    ledger_hashes = {}
    for unit_id in sorted(expected):
        receipt_path = receipt_dir / f"{unit_id}.json"
        receipt = load_json(receipt_path)
        require_schema(receipt, RECEIPT_SCHEMA, receipt_path)
        if (
            receipt["status"] != "passed"
            or receipt["contract_sha256"] != contract_sha
            or receipt["plan_sha256"] != plan_sha
            or receipt["unit_id"] != unit_id
        ):
            raise ValueError(f"receipt binding failed: {receipt_path}")
        ledger = receipt["ledger"]
        if sha256_file(ledger["path"]) != ledger["sha256"]:
            raise ValueError(f"ledger hash failed: {ledger['path']}")
        table = pq.read_table(
            ledger["path"],
            columns=[
                "rank",
                "source_dataset",
                "source_doc_id",
                "chars_before",
                "total_chars_removed",
                "lines_before",
                "lines_removed",
                "had_removal",
                "would_empty",
                "over_50pct",
                "removed_fraction",
                "spans",
            ],
        )
        if table.num_rows != ledger["rows"]:
            raise ValueError(f"ledger row count failed: {ledger['path']}")
        for row in table.to_pylist():
            identity = (row["rank"], row["source_doc_id"])
            if identity in identities:
                raise ValueError(f"duplicate document ledger identity: {identity}")
            identities.add(identity)
            for stats in (by_source[row["source_dataset"]], overall):
                stats["docs"] += 1
                stats["docs_cleaned"] += int(row["had_removal"])
                stats["chars_before"] += row["chars_before"]
                stats["total_chars_removed"] += row["total_chars_removed"]
                stats["lines_before"] += row["lines_before"]
                stats["lines_removed"] += row["lines_removed"]
                stats["spans"] += len(row["spans"])
                stats["would_empty"] += int(row["would_empty"])
                stats["docs_over_30pct"] += int(row["removed_fraction"] > 0.3)
                stats["docs_over_50pct"] += int(row["over_50pct"])
                if row["had_removal"]:
                    stats["fractions"].append(row["removed_fraction"])
        receipt_hashes[unit_id] = sha256_file(receipt_path)
        ledger_hashes[unit_id] = ledger["sha256"]
    if overall["docs"] != plan["rows"]:
        raise ValueError(f"ledger rows {overall['docs']} != plan rows {plan['rows']}")

    def finalize(stats: dict[str, Any]) -> dict[str, Any]:
        values = stats.pop("fractions")
        stats["docs_cleaned_pct"] = 100 * stats["docs_cleaned"] / max(1, stats["docs"])
        stats["char_removal_pct"] = (
            100 * stats["total_chars_removed"] / max(1, stats["chars_before"])
        )
        stats["removed_fraction_p50"] = percentile(values, 0.5)
        stats["removed_fraction_p95"] = percentile(values, 0.95)
        return stats

    result = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "passed",
        "run_id": contract["run_id"],
        "mode": contract["mode"],
        "contract_sha256": contract_sha,
        "plan_sha256": plan_sha,
        "receipt_count": len(receipt_hashes),
        "document_identities": len(identities),
        "by_source": {
            source: finalize(stats) for source, stats in sorted(by_source.items())
        },
        "overall": finalize(overall),
        "receipt_sha256": receipt_hashes,
        "ledger_sha256": ledger_hashes,
    }
    output = Path(
        args.output or Path(contract["run_root"]) / contract["mode"] / "summary.json"
    )
    atomic_write_json(output, result)
    csv_path = output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=csv_path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(
            handle,
            extrasaction="ignore",
            fieldnames=[
                "source_dataset",
                "docs",
                "docs_cleaned",
                "docs_cleaned_pct",
                "char_removal_pct",
                "removed_fraction_p50",
                "removed_fraction_p95",
                "docs_over_30pct",
                "docs_over_50pct",
                "would_empty",
            ],
        )
        writer.writeheader()
        for source, stats in result["by_source"].items():
            writer.writerow({"source_dataset": source, **stats})
        temporary_csv = Path(handle.name)
    temporary_csv.replace(csv_path)
    print(f"{output}: {overall['docs']} documents from {len(by_source)} sources")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

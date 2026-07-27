#!/usr/bin/env python3
"""Build the deterministic Kallipos and high-removal QA packet from the full ledger."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .contracts import (
    QA_PACKET_SCHEMA,
    QA_REVIEW_SCHEMA,
    SUMMARY_SCHEMA,
    atomic_write_json,
    load_json,
    require_schema,
    sha256_file,
    validate_plan,
)


def _even_middle_sample(rows: list[dict], count: int) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["removed_fraction"],
            hashlib.sha256(row["source_doc_id"].encode()).hexdigest(),
        ),
    )
    if len(ordered) < count:
        raise ValueError(f"only {len(ordered)} eligible rows for a {count}-item sample")
    low = int(0.35 * (len(ordered) - 1))
    high = int(0.65 * (len(ordered) - 1))
    if count == 1:
        indices = [(low + high) // 2]
    else:
        indices = [
            round(low + (high - low) * index / (count - 1)) for index in range(count)
        ]
    if len(set(indices)) != count:
        indices = list(range((len(ordered) - count) // 2, (len(ordered) + count) // 2))
    return [ordered[index] for index in indices]


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract, _plan, contract_sha, plan_sha = validate_plan(args.contract, args.plan)
    summary = load_json(args.summary)
    require_schema(summary, SUMMARY_SCHEMA, args.summary)
    if (
        summary["status"] != "passed"
        or summary["contract_sha256"] != contract_sha
        or summary["plan_sha256"] != plan_sha
    ):
        raise ValueError("summary is not bound to this contract and plan")

    rows: list[dict] = []
    ledger_dir = Path(contract["run_root"]) / contract["mode"] / "ledger"
    ledger_paths = sorted(ledger_dir.glob("*.parquet"))
    expected_ledgers = summary["ledger_sha256"]
    actual_ledgers = {path.stem for path in ledger_paths}
    if actual_ledgers != set(expected_ledgers):
        raise ValueError(
            "ledger set changed after aggregation: "
            f"missing={sorted(set(expected_ledgers) - actual_ledgers)}, "
            f"extra={sorted(actual_ledgers - set(expected_ledgers))}"
        )
    for ledger_path in ledger_paths:
        if sha256_file(ledger_path) != expected_ledgers[ledger_path.stem]:
            raise ValueError(f"ledger changed after aggregation: {ledger_path}")
        rows.extend(pq.read_table(ledger_path).to_pylist())
    kallipos = [
        row
        for row in rows
        if row["source_dataset"] == "Apothetirio_Kallipos" and row["had_removal"]
    ]
    selected: dict[tuple[int, str], tuple[dict, set[str]]] = {}

    def add(row: dict, reason: str) -> None:
        key = (row["rank"], row["source_doc_id"])
        if key not in selected:
            selected[key] = (row, set())
        selected[key][1].add(reason)

    for row in _even_middle_sample(
        kallipos, contract["policy"]["qa_gate"]["kallipos_sample_size"]
    ):
        add(row, "kallipos_median")
    for row in rows:
        if row["source_dataset"] == "openarchives.gr" and row["over_50pct"]:
            add(row, "openarchives_over_50pct")
        if row["would_empty"]:
            add(row, "would_empty")

    by_location: dict[tuple[int, int], list[tuple[dict, set[str]]]] = defaultdict(list)
    for row, reasons in selected.values():
        by_location[(row["rank"], row["row_group"])].append((row, reasons))
    items = []
    for (rank, row_group), candidates in sorted(by_location.items()):
        source_path = Path(contract["release_root"]) / "data" / f"{rank:06d}.parquet"
        source_rows = pq.ParquetFile(source_path).read_row_group(row_group).to_pylist()
        for ledger_row, reasons in candidates:
            source_row = source_rows[ledger_row["row_in_group"]]
            original = source_row["text"] or ""
            if (
                str(source_row["source_doc_id"]) != ledger_row["source_doc_id"]
                or hashlib.sha256(original.encode()).hexdigest()
                != ledger_row["original_sha256"]
            ):
                raise ValueError("QA rehydration identity check failed")
            span_context = []
            for span in ledger_row["spans"]:
                start, end = span["char_start"], span["char_end"]
                removed = original[start:end]
                if (
                    hashlib.sha256(removed.encode()).hexdigest()
                    != span["removed_sha256"]
                ):
                    raise ValueError("QA removed-span hash check failed")
                span_context.append(
                    {
                        **span,
                        "before_context": original[max(0, start - 1500) : start],
                        "removed_excerpt": removed[:8000],
                        "removed_excerpt_truncated": len(removed) > 8000,
                        "after_context": original[end : end + 1500],
                    }
                )
            item_id = f"{rank:06d}:{ledger_row['source_doc_id']}"
            items.append(
                {
                    "item_id": item_id,
                    "reasons": sorted(reasons),
                    "rank": rank,
                    "row_group": row_group,
                    "row_in_group": ledger_row["row_in_group"],
                    "source_dataset": ledger_row["source_dataset"],
                    "source_doc_id": ledger_row["source_doc_id"],
                    "chars_before": ledger_row["chars_before"],
                    "total_chars_removed": ledger_row["total_chars_removed"],
                    "removed_fraction": ledger_row["removed_fraction"],
                    "would_empty": ledger_row["would_empty"],
                    "spans": span_context,
                }
            )
    packet = {
        "schema_version": QA_PACKET_SCHEMA,
        "status": "ready_for_review",
        "run_id": contract["run_id"],
        "contract_sha256": contract_sha,
        "plan_sha256": plan_sha,
        "summary_sha256": sha256_file(args.summary),
        "gate": contract["policy"]["qa_gate"],
        "item_count": len(items),
        "items": sorted(items, key=lambda item: item["item_id"]),
    }
    output = Path(
        args.output or Path(contract["run_root"]) / contract["mode"] / "qa_packet.json"
    )
    atomic_write_json(output, packet)
    review = {
        "schema_version": QA_REVIEW_SCHEMA,
        "status": "incomplete",
        "run_id": contract["run_id"],
        "packet_sha256": sha256_file(output),
        "reviewer": None,
        "reviewed_utc": None,
        "decisions": [
            {
                "item_id": item["item_id"],
                "classification": None,
                "primarily_bibliography": None,
                "rationale": None,
            }
            for item in packet["items"]
        ],
    }
    atomic_write_json(output.with_name("qa_review.json"), review)
    print(f"{output}: {len(items)} items")
    return packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

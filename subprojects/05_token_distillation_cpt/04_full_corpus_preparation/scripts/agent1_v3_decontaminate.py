#!/usr/bin/env python3
"""GreekMMLU decontamination for the ordered Agent 1 v3 lane.

Unlike the legacy final-stage tool, this runs after deduplication and before
anonymization.  High-confidence question+answer contamination is dropped;
all otherwise non-destructive match evidence is quarantined rather than
quietly retained or deleted.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from decontaminate_full_corpus import load_benchmark_index, match_document
from finalization_io import (
    atomic_output_path,
    discover_parquet,
    parquet_file_receipt,
    sha256_text,
    utc_now,
    write_json_atomic,
)


LEDGER_SCHEMA_VERSION = "agent1_full_corpus_v3_greekmmlu_ledger_v1"
MANIFEST_SCHEMA_VERSION = "agent1_full_corpus_v3_decontamination_manifest_v1"


def action_for_match(action: str, reason: str, evidence: list[dict[str, Any]]) -> tuple[str, str]:
    if action == "drop":
        return "drop", reason
    if evidence:
        return "quarantine", "greekmmlu_ambiguous_match_evidence"
    return "keep", reason


def ledger_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("representation_id", pa.string()),
            ("input_text_sha256", pa.string()),
            ("action", pa.string()),
            ("reason", pa.string()),
            ("benchmark_matches_json", pa.string()),
        ]
    )


def claimed_hash(row: dict[str, Any], actual: str) -> None:
    for name in ("text_sha256", "cleaned_text_sha256"):
        claimed = row.get(name)
        if claimed is not None and str(claimed) != actual:
            raise ValueError(f"{row.get('stable_uid')}: {name} drift")


def process_file(
    input_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    dropped_root: Path,
    quarantine_root: Path,
    ledger_root: Path,
    benchmark_index: Any,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    relative = input_path.relative_to(input_root)
    parquet = pq.ParquetFile(input_path)
    required = {"stable_uid", "text"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{input_path}: missing required columns {sorted(missing)}")
    outputs = {
        "output": output_root / relative,
        "dropped": dropped_root / relative,
        "quarantine": quarantine_root / relative,
        "ledger": ledger_root / relative,
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = {name: atomic_output_path(path) for name, path in outputs.items()}
    writers = {
        "output": pq.ParquetWriter(temporary["output"], parquet.schema_arrow, compression="zstd"),
        "dropped": pq.ParquetWriter(temporary["dropped"], parquet.schema_arrow, compression="zstd"),
        "quarantine": pq.ParquetWriter(temporary["quarantine"], parquet.schema_arrow, compression="zstd"),
        "ledger": pq.ParquetWriter(temporary["ledger"], ledger_schema(), compression="zstd"),
    }
    counts: Counter[str] = Counter()
    try:
        for batch in parquet.iter_batches(batch_size=512, use_threads=False):
            rows = batch.to_pylist()
            partitions: dict[str, list[dict[str, Any]]] = {"keep": [], "drop": [], "quarantine": []}
            ledger_rows: list[dict[str, Any]] = []
            for row in rows:
                text = str(row.get("text") or "")
                text_hash = sha256_text(text)
                claimed_hash(row, text_hash)
                raw_action, raw_reason, evidence = match_document(text, benchmark_index)
                action, reason = action_for_match(raw_action, raw_reason, evidence)
                representation_id = str(row.get("representation_id") or row.get("stable_uid"))
                partitions[action].append(row)
                ledger_rows.append(
                    {
                        "stable_uid": str(row["stable_uid"]),
                        "representation_id": representation_id,
                        "input_text_sha256": text_hash,
                        "action": action,
                        "reason": reason,
                        "benchmark_matches_json": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                    }
                )
                counts["input"] += 1
                counts[action] += 1
                if evidence:
                    counts["evidence_rows"] += 1
            for name in ("keep", "drop", "quarantine"):
                if partitions[name]:
                    writers[{"keep": "output", "drop": "dropped", "quarantine": "quarantine"}[name]].write_table(
                        pa.Table.from_pylist(partitions[name], schema=parquet.schema_arrow)
                    )
            if ledger_rows:
                writers["ledger"].write_table(pa.Table.from_pylist(ledger_rows, schema=ledger_schema()))
        for writer in writers.values():
            writer.close()
        for name, path in outputs.items():
            os.replace(temporary[name], path)
    except BaseException:
        for writer in writers.values():
            writer.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return {
        "relative_path": relative.as_posix(),
        "counts": dict(counts),
        **{name: parquet_file_receipt(path, relative_to=root) for (name, path), root in (
            (("output", outputs["output"]), output_root),
            (("dropped", outputs["dropped"]), dropped_root),
            (("quarantine", outputs["quarantine"]), quarantine_root),
            (("ledger", outputs["ledger"]), ledger_root),
        )},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dropped", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers != 1:
        raise ValueError("v3 decontamination is deliberately one-process until its benchmark index has a receipt-bound worker serialization")
    if args.manifest.exists():
        raise FileExistsError(f"immutable manifest already exists: {args.manifest}")
    for root in (args.output, args.dropped, args.quarantine, args.ledger):
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"refusing non-empty output root: {root}")
        root.mkdir(parents=True, exist_ok=True)
    index, benchmark = load_benchmark_index(
        args.queries_jsonl,
        args.benchmark_manifest,
        k=8,
        min_coverage=0.85,
        minhash_threshold=0.85,
        min_matched_grams=4,
        max_gap_tokens=40,
    )
    receipts = [
        process_file(
            path,
            input_root=args.input,
            output_root=args.output,
            dropped_root=args.dropped,
            quarantine_root=args.quarantine,
            ledger_root=args.ledger,
            benchmark_index=index,
        )
        for path in discover_parquet(args.input)
    ]
    totals: Counter[str] = Counter()
    for receipt in receipts:
        totals.update({key: int(value) for key, value in receipt["counts"].items()})
    if totals["input"] != totals["keep"] + totals["drop"] + totals["quarantine"]:
        raise RuntimeError("decontamination action ledger does not close")
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "passed",
        "completed_at": utc_now(),
        "benchmark": benchmark,
        "policy": {
            "high_confidence_actions": "drop",
            "ambiguous_match_actions": "quarantine",
            "answer_only_action": "audit_only",
        },
        "counts": dict(totals),
        "files": receipts,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": dict(totals)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

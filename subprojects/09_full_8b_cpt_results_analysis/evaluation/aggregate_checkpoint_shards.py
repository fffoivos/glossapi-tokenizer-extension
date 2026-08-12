#!/usr/bin/env python3
"""Verify, merge and summarize all independently scored checkpoint shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from run_checkpoint_suite import summarize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    contract, manifest = json.loads(args.contract.read_text()), json.loads(args.manifest.read_text())
    examples_path = Path(manifest["examples"]["path"])
    if sha256(examples_path) != manifest["examples"]["sha256"]:
        raise ValueError("frozen example hash drift")
    expected_rows = read_jsonl(examples_path)
    expected_keys = {(row["benchmark"], row["example_id"]) for row in expected_rows}
    if len(expected_keys) != len(expected_rows):
        raise ValueError("frozen examples do not have unique benchmark/example identities")

    aggregate_receipts = []
    for checkpoint in contract["checkpoint_scope"]:
        label = checkpoint["label"]
        shard_receipts = sorted((args.root / label).glob("*/receipt.json"))
        if len(shard_receipts) != 5:
            raise ValueError(f"expected five shard receipts for {label}, got {len(shard_receipts)}")
        rows, bound_receipts = [], []
        for receipt_path in shard_receipts:
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "completed" or receipt.get("model", {}).get("label") != label:
                raise ValueError(f"invalid shard receipt {receipt_path}")
            predictions_path = Path(receipt["artifacts"]["predictions"]["path"])
            if sha256(predictions_path) != receipt["artifacts"]["predictions"]["sha256"]:
                raise ValueError(f"prediction hash drift for {receipt_path}")
            rows.extend(read_jsonl(predictions_path))
            bound_receipts.append({"path": str(receipt_path.resolve()), "sha256": sha256(receipt_path)})
        keys = [(row["benchmark"], row["example_id"]) for row in rows]
        duplicates = [key for key, count in Counter(keys).items() if count != 1]
        if duplicates:
            raise ValueError(f"duplicate shard identities for {label}: {duplicates[:5]}")
        if set(keys) != expected_keys:
            raise ValueError(f"shard union differs from frozen examples for {label}")
        measured_counts = Counter(row["benchmark"] for row in rows)
        if dict(measured_counts) != manifest["counts"]:
            raise ValueError(f"benchmark counts differ for {label}")

        combined = args.root / label / "combined"
        combined.mkdir()
        rows.sort(key=lambda row: (row["benchmark"], row["example_id"]))
        predictions_path = combined / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        metrics = summarize(rows)
        metrics_path = combined / "metrics.csv"
        fields = sorted({key for row in metrics for key in row})
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metrics)
        receipt = {
            "schema_version": "apertus_full8_native_greek_checkpoint_aggregate_v1",
            "status": "completed",
            "model": label,
            "counts": dict(measured_counts),
            "shard_receipts": bound_receipts,
            "artifacts": {
                "predictions": {"path": str(predictions_path.resolve()), "sha256": sha256(predictions_path)},
                "metrics": {"path": str(metrics_path.resolve()), "sha256": sha256(metrics_path)},
            },
        }
        receipt_path = combined / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        aggregate_receipts.append({"path": str(receipt_path.resolve()), "sha256": sha256(receipt_path), "model": label})

    matrix = {
        "schema_version": "apertus_full8_native_greek_3cp_matrix_v2",
        "status": "completed",
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "contract_sha256": sha256(args.contract),
        "manifest_sha256": sha256(args.manifest),
        "checkpoint_receipts": aggregate_receipts,
    }
    (args.root / "matrix_receipt.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "checkpoints": len(aggregate_receipts), "examples_per_checkpoint": len(expected_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

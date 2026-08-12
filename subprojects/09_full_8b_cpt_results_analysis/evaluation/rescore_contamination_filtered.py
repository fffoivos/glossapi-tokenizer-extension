#!/usr/bin/env python3
"""Recompute checkpoint metrics after strict contamination exclusions."""

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
        return [json.loads(line) for line in handle if line.strip()]


def write_metrics(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--exclusions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    matrix_path = args.matrix_root / "matrix_receipt.json"
    matrix = json.loads(matrix_path.read_text())
    if matrix.get("status") != "completed":
        raise ValueError("matrix is not complete")
    exclusion_rows = read_jsonl(args.exclusions_jsonl)
    exclusions = {(str(row["benchmark"]), str(row["example_id"])) for row in exclusion_rows}
    if len(exclusions) != len(exclusion_rows):
        raise ValueError("duplicate exclusion identities")

    checkpoint_receipts = []
    delta_rows = []
    for binding in matrix["checkpoint_receipts"]:
        receipt_path = Path(binding["path"])
        if sha256(receipt_path) != binding["sha256"]:
            raise ValueError(f"checkpoint receipt drift: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        predictions_path = Path(receipt["artifacts"]["predictions"]["path"])
        if sha256(predictions_path) != receipt["artifacts"]["predictions"]["sha256"]:
            raise ValueError(f"prediction drift: {predictions_path}")
        rows = read_jsonl(predictions_path)
        prediction_keys = {(str(row["benchmark"]), str(row["example_id"])) for row in rows}
        unknown = exclusions - prediction_keys
        # Exclusions cover the shared benchmark panel, so every checkpoint must
        # contain every excluded identity.
        if unknown:
            raise ValueError(f"exclusions absent from {binding['model']}: {sorted(unknown)[:5]}")
        clean = [row for row in rows if (str(row["benchmark"]), str(row["example_id"])) not in exclusions]
        full_metrics = summarize(rows)
        clean_metrics = summarize(clean)
        checkpoint_dir = args.output_dir / binding["model"]
        checkpoint_dir.mkdir()
        full_path = checkpoint_dir / "full_metrics.csv"
        clean_path = checkpoint_dir / "strict_filtered_metrics.csv"
        write_metrics(full_path, full_metrics)
        write_metrics(clean_path, clean_metrics)
        full_by_key = {(row["benchmark"], row["subject"]): row for row in full_metrics}
        for clean_row in clean_metrics:
            key = (clean_row["benchmark"], clean_row["subject"])
            full_row = full_by_key[key]
            delta_rows.append(
                {
                    "model": binding["model"],
                    "benchmark": key[0],
                    "subject": key[1],
                    "full_n": full_row["n"],
                    "filtered_n": clean_row["n"],
                    "excluded_n": full_row["n"] - clean_row["n"],
                    "full_accuracy": full_row["accuracy"],
                    "filtered_accuracy": clean_row["accuracy"],
                    "accuracy_delta": clean_row["accuracy"] - full_row["accuracy"],
                    "full_choice_nll": full_row["choice_nll"],
                    "filtered_choice_nll": clean_row["choice_nll"],
                    "choice_nll_delta": (
                        None
                        if full_row["choice_nll"] is None or clean_row["choice_nll"] is None
                        else clean_row["choice_nll"] - full_row["choice_nll"]
                    ),
                }
            )
        checkpoint_receipts.append(
            {
                "model": binding["model"],
                "source_receipt_sha256": binding["sha256"],
                "full_metrics_sha256": sha256(full_path),
                "strict_filtered_metrics_sha256": sha256(clean_path),
                "rows_full": len(rows),
                "rows_filtered": len(clean),
            }
        )

    delta_path = args.output_dir / "full_vs_strict_filtered.csv"
    write_metrics(delta_path, delta_rows)
    counts = Counter(row["benchmark"] for row in exclusion_rows)
    output_receipt = {
        "schema_version": "apertus_full8_contamination_filtered_rescore_v1",
        "status": "passed",
        "matrix_receipt": {"path": str(matrix_path), "sha256": sha256(matrix_path)},
        "exclusions": {
            "path": str(args.exclusions_jsonl),
            "sha256": sha256(args.exclusions_jsonl),
            "counts_by_benchmark": dict(sorted(counts.items())),
            "rows": len(exclusion_rows),
        },
        "policy": "exclude only evaluation identities with a strong two-surface corpus match",
        "checkpoints": checkpoint_receipts,
        "comparison": {"path": str(delta_path), "sha256": sha256(delta_path)},
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(output_receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "excluded": len(exclusion_rows), "checkpoints": len(checkpoint_receipts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

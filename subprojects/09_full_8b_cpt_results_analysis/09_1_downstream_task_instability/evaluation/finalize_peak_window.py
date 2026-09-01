#!/usr/bin/env python3
"""Join four new native-Greek evaluations with the existing best checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ORDER = [
    "iter_0007152",
    "iter_0008344",
    "iter_0009536",
    "iter_0010728",
    "iter_0011920",
]
NEW_LABELS = [label for label in ORDER if label != "iter_0009536"]
BENCHMARKS = [
    "demosqa",
    "medical_mcqa",
    "asep_mcqa",
    "gpcr",
    "oyxoy_nli",
    "oyxoy_nli_exact_set",
    "oyxoy_wsd_definition",
    "oyxoy_wic",
    "oyxoy_metaphor",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_aggregate_metrics(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row["benchmark"]: row
        for row in rows
        if row.get("subject") == "__all__" and row.get("benchmark") in BENCHMARKS
    }
    if set(selected) != set(BENCHMARKS):
        raise ValueError(
            f"{path}: benchmark coverage drift: "
            f"missing={sorted(set(BENCHMARKS)-set(selected))}"
        )
    return selected


def verify_matrix(root: Path, expected_labels: list[str]) -> dict[str, Any]:
    receipt_path = root / "matrix_receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("status") != "completed":
        raise ValueError(f"matrix is not complete: {root}")
    labels = [row["model"] for row in receipt.get("checkpoint_receipts", [])]
    if labels != expected_labels:
        raise ValueError(f"checkpoint order drift in {root}: {labels}")
    for row in receipt["checkpoint_receipts"]:
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"checkpoint receipt hash drift: {path}")
    return {
        "path": str(receipt_path.resolve()),
        "sha256": sha256(receipt_path),
        "schema_version": receipt.get("schema_version"),
    }


def verify_filtered(root: Path, expected_labels: list[str]) -> dict[str, Any]:
    receipt_path = root / "receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("status") != "passed":
        raise ValueError(f"filtered rescore is not passed: {root}")
    labels = [row["model"] for row in receipt.get("checkpoints", [])]
    if labels != expected_labels:
        raise ValueError(f"filtered checkpoint order drift in {root}: {labels}")
    for row in receipt["checkpoints"]:
        label = row["model"]
        path = root / label / "strict_filtered_metrics.csv"
        if sha256(path) != row["strict_filtered_metrics_sha256"]:
            raise ValueError(f"filtered metrics hash drift: {path}")
    return {"path": str(receipt_path.resolve()), "sha256": sha256(receipt_path)}


def verify_subset_receipt(path: Path) -> dict[str, Any]:
    receipt = read_json(path)
    if receipt.get("status") != "passed" or not all(receipt.get("checks", {}).values()):
        raise ValueError("clean-subset rebind receipt is not passed")
    subset = receipt.get("clean_subset", {})
    if subset.get("exclusions", {}).get("rows") != 10076:
        raise ValueError("clean-subset exclusion count drift")
    if sum(subset.get("retained_by_benchmark", {}).values()) != 73894:
        raise ValueError("clean-subset retained count drift")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "clean_subset": subset,
    }


def numeric(row: dict[str, str]) -> dict[str, Any]:
    def maybe_float(key: str) -> float | None:
        value = row.get(key)
        return None if value in (None, "", "None") else float(value)

    return {
        "n": int(row["n"]),
        "accuracy": float(row["accuracy"]),
        "choice_nll": maybe_float("choice_nll"),
        "correct_answer_bpb": maybe_float("correct_answer_bpb"),
        "binary_macro_f1": maybe_float("binary_macro_f1"),
        "balanced_accuracy": maybe_float("balanced_accuracy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-matrix-root", type=Path, required=True)
    parser.add_argument("--reference-matrix-root", type=Path, required=True)
    parser.add_argument("--reference-filtered-root", type=Path, required=True)
    parser.add_argument("--subset-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    bindings = {
        "new_matrix": verify_matrix(args.new_matrix_root, NEW_LABELS),
        "reference_matrix": verify_matrix(
            args.reference_matrix_root,
            ["iter_0000000", "iter_0009536", "iter_0018284"],
        ),
        "reference_filtered": verify_filtered(
            args.reference_filtered_root,
            ["iter_0000000", "iter_0009536", "iter_0018284"],
        ),
        "clean_subset": verify_subset_receipt(args.subset_receipt),
    }

    table: list[dict[str, Any]] = []
    for label in ORDER:
        if label == "iter_0009536":
            metrics_path = args.reference_filtered_root / label / "strict_filtered_metrics.csv"
            provenance = "reused_authoritative_3cp_evaluation"
        else:
            metrics_path = args.new_matrix_root / label / "combined/metrics.csv"
            provenance = "new_peak_window_evaluation"
        clean = read_aggregate_metrics(metrics_path)
        iteration = int(label.split("_")[-1])
        table.append(
            {
                "label": label,
                "iteration": iteration,
                "token_slots": iteration * 4_194_304,
                "provenance": provenance,
                "benchmarks": {
                    benchmark: numeric(clean[benchmark])
                    for benchmark in BENCHMARKS
                },
                "artifacts": {
                    "clean_subset_metrics": {
                        "path": str(metrics_path.resolve()),
                        "sha256": sha256(metrics_path),
                    },
                },
            }
        )

    payload = {
        "schema_version": "apertus_full8_native_greek_peak_window_results_v1",
        "status": "completed",
        "checkpoint_order": ORDER,
        "best_checkpoint": "iter_0009536",
        "benchmarks": BENCHMARKS,
        "evaluation_population": "clean_subset_only",
        "table": table,
        "bindings": bindings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"ok": True, "checkpoints": len(table), "benchmarks": len(BENCHMARKS)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

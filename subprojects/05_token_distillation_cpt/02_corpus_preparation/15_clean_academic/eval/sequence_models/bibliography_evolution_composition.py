#!/usr/bin/env python3
"""Evaluate one predeclared pairwise composition of frozen predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_entry_blocks import evaluate_prediction
from .bibliography_entry_models import load_table


SCHEMA_VERSION = "bibliography-evolution-pairwise-composition-v1"


def combine_parent_barriers(
    left: Any, right: Any, shape: tuple[int, ...]
) -> dict[str, np.ndarray]:
    combined: dict[str, np.ndarray] = {}
    for name in ("hard_wall", "upward_stop", "downward_stop"):
        if (
            name not in left or name not in right
            or left[name].shape != shape or right[name].shape != shape
        ):
            raise ValueError("pairwise parent barrier artifacts do not align")
        combined[name] = left[name].astype(bool) | right[name].astype(bool)
    return combined


def enforce_combined_barriers(
    prediction: np.ndarray, barriers: dict[str, np.ndarray]
) -> np.ndarray:
    result = prediction.astype(bool, copy=True)
    result[barriers["hard_wall"]] = False
    for right in range(1, len(result)):
        left = right - 1
        if result[left] and result[right] and barriers["upward_stop"][right]:
            result[left] = False
        if result[left] and result[right] and barriers["downward_stop"][left]:
            result[right] = False
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _qualified_documents(table: Any, path: Path) -> set[int]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    ids = packet.get("document_ids")
    if not isinstance(ids, list) or len(ids) != 268 or len(ids) != len(set(ids)):
        raise ValueError("qualified development inventory must contain 268 unique ids")
    wanted = set(str(value) for value in ids)
    index = {
        str(document["document_id"]): number
        for number, document in enumerate(table.documents)
    }
    if not wanted.issubset(index):
        raise ValueError("qualified development inventory is not in the table")
    return {index[value] for value in wanted}


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="validation")
    left = np.load(args.left_prediction, allow_pickle=False).astype(bool)
    right = np.load(args.right_prediction, allow_pickle=False).astype(bool)
    if left.shape != right.shape or left.shape != (len(table.targets),):
        raise ValueError("pairwise predictions do not align to the validation table")
    left_barrier = np.load(args.left_barrier_artifact, allow_pickle=False)
    right_barrier = np.load(args.right_barrier_artifact, allow_pickle=False)
    # A composition must respect the union of both parents' restrictions;
    # using either parent's wall set alone can reintroduce a line forbidden by
    # the other parent.
    combined = combine_parent_barriers(left_barrier, right_barrier, left.shape)
    if args.operation == "union":
        prediction = left | right
    elif args.operation == "intersection":
        prediction = left & right
    elif args.operation == "left_minus_right":
        prediction = left & ~right
    else:  # pragma: no cover - argparse owns choices
        raise AssertionError(args.operation)
    prediction = enforce_combined_barriers(prediction, combined)
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    prediction_path = output / "prediction.npy"
    with prediction_path.open("xb") as handle:
        np.save(handle, prediction, allow_pickle=False)
    barrier_output = output / "combined_barriers.npz"
    with barrier_output.open("xb") as handle:
        np.savez(
            handle,
            hard_wall=combined["hard_wall"],
            upward_stop=combined["upward_stop"],
            downward_stop=combined["downward_stop"],
        )
    documents = _qualified_documents(table, Path(args.qualified_documents))
    metrics = evaluate_prediction(table, prediction, document_subset=documents)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_pairwise_development_evaluation",
        "validation_opened": True,
        "final_test_opened": False,
        "operation": args.operation,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "inputs": {
            "left_prediction": _sha256(Path(args.left_prediction)),
            "right_prediction": _sha256(Path(args.right_prediction)),
            "qualified_documents": _sha256(Path(args.qualified_documents)),
            "left_barrier_artifact": _sha256(Path(args.left_barrier_artifact)),
            "right_barrier_artifact": _sha256(Path(args.right_barrier_artifact)),
        },
        "metrics": metrics,
        "prediction_sha256": _sha256(prediction_path),
    }
    _write_json_new(output / "report.json", result)
    outputs = {
        path.relative_to(output).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    _write_json_new(output / "receipt.json", {**result, "outputs": outputs})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--left-prediction", required=True)
    parser.add_argument("--right-prediction", required=True)
    parser.add_argument("--qualified-documents", required=True)
    parser.add_argument("--left-barrier-artifact", required=True)
    parser.add_argument("--right-barrier-artifact", required=True)
    parser.add_argument("--operation", choices=("union", "intersection", "left_minus_right"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

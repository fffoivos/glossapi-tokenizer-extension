#!/usr/bin/env python3
"""Create an immutable weighted ensemble of aligned development OOF models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_nextgen_models import SCHEMA_VERSION as MODEL_SCHEMA
from .contract import sha256_file


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    roots = [Path(value).resolve() for value in args.model_dirs]
    weights = np.asarray(args.weights, dtype=np.float64)
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if len(roots) < 2 or len(weights) != len(roots):
        raise ValueError("provide one weight for each of at least two models")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1.0e-8):
        raise ValueError("ensemble weights must be non-negative and sum to one")
    reports = [
        json.loads((root / "report.json").read_text(encoding="utf-8"))
        for root in roots
    ]
    if any(
        report.get("schema_version") != MODEL_SCHEMA
        or report.get("test_opened") is not False
        for report in reports
    ):
        raise ValueError("ensemble inputs must be sealed development OOF models")
    arrays = [
        np.load(root / "oof_probability.npy", mmap_mode="r", allow_pickle=False)
        for root in roots
    ]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("ensemble probabilities are not aligned")
    probability = np.zeros(arrays[0].shape, dtype=np.float64)
    for weight, array in zip(weights, arrays, strict=True):
        if not np.isfinite(array).all():
            raise ValueError("ensemble input contains non-finite values")
        probability += weight * array
    probability = probability.astype(np.float32)
    output.mkdir(parents=True)
    with (output / "oof_probability.npy").open("xb") as handle:
        np.save(handle, probability, allow_pickle=False)
    report = {
        "schema_version": MODEL_SCHEMA,
        "status": "passed_aligned_development_oof_ensemble",
        "kind": args.name,
        "validation_opened": False,
        "test_opened": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "members": [
            {
                "path": str(root),
                "kind": member["kind"],
                "weight": float(weight),
                "receipt_sha256": sha256_file(root / "receipt.json"),
                "probability_sha256": sha256_file(root / "oof_probability.npy"),
            }
            for root, member, weight in zip(roots, reports, weights, strict=True)
        ],
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(
        output / "receipt.json",
        {
            **report,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dirs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

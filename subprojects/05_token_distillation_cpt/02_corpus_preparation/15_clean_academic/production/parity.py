#!/usr/bin/env python3
"""Require exact 0.9 line-mask parity against the pinned 210,704-line reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .contracts import PARITY_SCHEMA, atomic_write_json, sha256_file


def run(args: argparse.Namespace) -> dict:
    reference_raw = np.load(args.reference)
    candidate_raw = np.load(args.candidate)
    reference = _probabilities(reference_raw)
    candidate = _probabilities(candidate_raw)
    if reference.shape != candidate.shape:
        equal = 0
        reference_positive = int((reference >= args.threshold).sum())
        candidate_positive = int((candidate >= args.threshold).sum())
    else:
        reference_mask = reference >= args.threshold
        candidate_mask = candidate >= args.threshold
        equal = int((reference_mask == candidate_mask).sum())
        reference_positive = int(reference_mask.sum())
        candidate_positive = int(candidate_mask.sum())
    total = int(reference.size)
    passed = (
        reference.shape == candidate.shape
        and total == args.expected_lines
        and equal == total
        and reference_positive == args.expected_positives
        and candidate_positive == args.expected_positives
    )
    result = {
        "schema_version": PARITY_SCHEMA,
        "status": "passed" if passed else "failed",
        "threshold": args.threshold,
        "expected_lines": args.expected_lines,
        "lines": total,
        "equal_masks": equal,
        "expected_positives": args.expected_positives,
        "reference_positives": reference_positive,
        "candidate_positives": candidate_positive,
        "reference": {
            "path": str(Path(args.reference).resolve()),
            "sha256": sha256_file(args.reference),
        },
        "candidate": {
            "path": str(Path(args.candidate).resolve()),
            "sha256": sha256_file(args.candidate),
        },
        "cohort": {
            "path": str(Path(args.cohort).resolve()),
            "sha256": sha256_file(args.cohort),
        },
        "train_commit": args.train_commit,
        "glossapi_commit": args.glossapi_commit,
        "binary_sha256": sha256_file(args.binary),
        "artifact_manifest_sha256": sha256_file(args.artifact_manifest),
    }
    atomic_write_json(args.output, result)
    print(
        f"{result['status']}: {equal}/{total} masks; "
        f"positives {candidate_positive}/{args.expected_positives}"
    )
    return result


def _probabilities(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] in (1, 2):
        return array[:, 0]
    raise ValueError(f"unsupported probability array shape {array.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--train-commit", required=True)
    parser.add_argument("--glossapi-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--expected-lines", type=int, default=210704)
    parser.add_argument("--expected-positives", type=int, default=19117)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    raise SystemExit(result["status"] != "passed")

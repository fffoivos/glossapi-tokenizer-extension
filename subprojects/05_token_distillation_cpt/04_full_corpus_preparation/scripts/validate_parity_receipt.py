#!/usr/bin/env python3
"""Validate exact Rust parity on a pinned LLM-silver comparison corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite_nonnegative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def positive_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    validation = policy.get("validation", {})
    expected_corpus = validation.get("structural_parity_corpus_sha256")
    expected_documents = validation.get("required_parity_documents")
    maximum_delta = finite_nonnegative(validation.get("maximum_probability_delta"))
    errors = []
    if (
        receipt.get("schema_version") != "struct_rust_parity_receipt_v1"
        or receipt.get("status") != "passed"
    ):
        errors.append("receipt schema/status is not passed")
    if (
        receipt.get("input_snapshot_method")
        != "private_job_local_o_nofollow_copy_rehash_before_publish"
        or receipt.get("inputs_rehashed_before_publication") is not True
    ):
        errors.append("receipt does not prove snapshotted and rehashed inputs")
    if validation.get("structural_parity_evidence") != "LLM_silver":
        errors.append("cleaning policy does not declare parity evidence as LLM_silver")
    if not isinstance(expected_corpus, str) or len(expected_corpus) != 64:
        errors.append("cleaning policy does not pin structural_parity_corpus_sha256")
    elif receipt.get("corpus_sha256", receipt.get("gold_sha256")) != expected_corpus:
        errors.append("receipt comparison-corpus hash does not match cleaning policy")
    if receipt.get("evidence_status") != "LLM_silver":
        errors.append("parity receipt does not declare LLM_silver evidence")
    heldout_documents = positive_integer(receipt.get("heldout_documents"))
    if positive_integer(expected_documents) is None:
        errors.append("cleaning policy parity document count is invalid")
    elif heldout_documents != expected_documents:
        errors.append(
            f"receipt does not cover all {expected_documents} parity documents"
        )
    if receipt.get("binary_sha256") != sha256_file(args.binary):
        errors.append("receipt does not match detector binary")
    tolerance = finite_nonnegative(receipt.get("tolerance"))
    if tolerance is None:
        errors.append("receipt tolerance is not finite and non-negative")
    elif maximum_delta is None or tolerance > maximum_delta:
        errors.append("receipt tolerance is looser than cleaning policy")
    for head in ("bib", "toc"):
        positive = positive_integer(
            receipt.get("positive_document_counts", {}).get(head)
        )
        result = receipt.get("heads", {}).get(head, {})
        documents = positive_integer(result.get("documents"))
        mismatches = result.get("span_mismatches")
        delta = finite_nonnegative(result.get("max_probability_difference"))
        if positive is None or (
            heldout_documents is not None and positive > heldout_documents
        ):
            errors.append(f"receipt lacks valid positive coverage for {head}")
        if documents is None or documents != heldout_documents:
            errors.append(f"receipt {head} document coverage differs from top level")
        if (
            isinstance(mismatches, bool)
            or not isinstance(mismatches, int)
            or mismatches != 0
        ):
            errors.append(f"receipt contains span mismatches for {head}")
        if delta is None or tolerance is None or delta > tolerance:
            errors.append(
                f"receipt {head} probability delta is invalid or exceeds tolerance"
            )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "binary_sha256": receipt["binary_sha256"],
                "corpus_sha256": expected_corpus,
                "evidence_status": "LLM_silver",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

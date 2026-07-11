#!/usr/bin/env python3
"""Validate exact Rust parity on a pinned LLM-silver comparison corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    errors = []
    if receipt.get("schema_version") != "struct_rust_parity_receipt_v1" or receipt.get("status") != "passed":
        errors.append("receipt schema/status is not passed")
    if validation.get("structural_parity_evidence") != "LLM_silver":
        errors.append("cleaning policy does not declare parity evidence as LLM_silver")
    if not isinstance(expected_corpus, str) or len(expected_corpus) != 64:
        errors.append("cleaning policy does not pin structural_parity_corpus_sha256")
    elif receipt.get("corpus_sha256", receipt.get("gold_sha256")) != expected_corpus:
        errors.append("receipt comparison-corpus hash does not match cleaning policy")
    if receipt.get("evidence_status") != "LLM_silver":
        errors.append("parity receipt does not declare LLM_silver evidence")
    if receipt.get("heldout_documents") != expected_documents:
        errors.append(f"receipt does not cover all {expected_documents} parity documents")
    if receipt.get("binary_sha256") != sha256_file(args.binary):
        errors.append("receipt does not match detector binary")
    if any(int(receipt.get("positive_document_counts", {}).get(head, 0)) <= 0 for head in ("bib", "toc")):
        errors.append("receipt lacks positive coverage for one or both heads")
    if any(int(receipt.get("heads", {}).get(head, {}).get("span_mismatches", -1)) != 0 for head in ("bib", "toc")):
        errors.append("receipt contains span mismatches")
    tolerance = float(receipt.get("tolerance", 1.0))
    if tolerance > 1e-3:
        errors.append("receipt tolerance is looser than 1e-3")
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

#!/usr/bin/env python3
"""Freeze a source-separated, cross-deduplicated neutral Modern-Greek corpus."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def verify_file(row: dict) -> Path:
    path = Path(row["path"])
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row["bytes"])
        or sha256_file(path) != row["sha256"]
    ):
        raise ValueError(f"file receipt drift: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedup-receipt", type=Path, required=True)
    parser.add_argument("--pool-corpus-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dedup = read(args.dedup_receipt)
    pool = read(args.pool_corpus_receipt)
    if (
        dedup.get("schema_version") != "apertus_mini_neutral_external_dedup_v1"
        or dedup.get("status") != "passed"
    ):
        raise ValueError("neutral-external dedup receipt is not passed")
    if pool.get("schema_version") != "apertus_mini_schedule_pool_corpus_v1" or pool.get("status") != "completed":
        raise ValueError("training pool corpus receipt is incomplete")
    expected_pool = dedup.get("training_reference", {}).get("pool_corpus_receipt", {})
    if (
        expected_pool.get("path") != str(args.pool_corpus_receipt.resolve())
        or expected_pool.get("bytes") != args.pool_corpus_receipt.stat().st_size
        or expected_pool.get("sha256") != sha256_file(args.pool_corpus_receipt)
    ):
        raise ValueError("neutral dedup is not bound to the frozen training corpus")
    identity = pool.get("global_identity_proof", {})
    if identity.get("modern_greek_exact_content_duplicates_or_collisions") != 0:
        raise ValueError("training Modern-Greek exact-content inventory is not unique")
    separation = dedup.get("dataset_separation", {})
    source_separated = (
        separation.get("publishers_or_domains_absent_from_training") is True
        or separation.get("source_time_window_absent_from_training") is True
    )
    if (
        separation.get("document_cluster_split") is not True
        or not source_separated
        or separation.get("candidate_documents_never_used_for_training") is not True
        or separation.get("evaluation_use_authorized") is not True
    ):
        raise ValueError("external source/time/rights separation is not proven")
    exact = dedup.get("exact_dedup", {})
    if (
        exact.get("algorithm") != "sha256_utf8_text"
        or exact.get("candidate_internal_duplicate_rows") != 0
        or exact.get("candidate_to_training_match_rows") != 0
    ):
        raise ValueError("neutral exact cross-dedup failed")
    near = dedup.get("minhash_dedup", {})
    expected_near = {
        "token_shingle_size": 5,
        "permutations": 128,
        "bands": 32,
        "rows_per_band": 4,
        "threshold": 0.85,
        "candidate_internal_pairs_at_or_above_threshold": 0,
        "candidate_to_training_pairs_at_or_above_threshold": 0,
    }
    if any(near.get(key) != value for key, value in expected_near.items()):
        raise ValueError("neutral 0.85 MinHash cross-dedup failed or drifted")
    snapshots = dedup.get("source_snapshot_receipts", [])
    if not snapshots:
        raise ValueError("neutral corpus has no source snapshot receipts")
    for row in snapshots:
        verify_file(row)
    candidate = dedup.get("candidate_output", {})
    candidate_path = verify_file(candidate)
    clusters = set()
    content = set()
    rows = 0
    with candidate_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{candidate_path}:{line_number}: expected object")
            cluster = str(row.get("cluster_id", ""))
            text = row.get("text")
            source = str(row.get("source_id", ""))
            if not cluster or not source or not isinstance(text, str) or not text.strip():
                raise ValueError(f"{candidate_path}:{line_number}: incomplete external document")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if cluster in clusters or digest in content:
                raise ValueError("candidate output is not independently cluster/exact unique")
            clusters.add(cluster)
            content.add(digest)
            rows += 1
    if rows != int(candidate.get("rows", -1)) or rows < 1:
        raise ValueError("candidate output row count drift")
    payload = {
        "schema_version": "apertus_mini_neutral_external_corpus_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_jsonl": candidate,
        "documents": rows,
        "unique_document_clusters": len(clusters),
        "unique_exact_contents": len(content),
        "document_cluster_split": True,
        "global_exact_dedup_against_training": True,
        "global_minhash_dedup_against_training": True,
        "minhash_threshold": 0.85,
        "publishers_or_domains_absent_from_training": separation.get("publishers_or_domains_absent_from_training") is True,
        "source_time_window_absent_from_training": separation.get("source_time_window_absent_from_training") is True,
        "source_separation_rule": "publisher_or_domain_or_time_window",
        "candidate_documents_never_used_for_training": True,
        "source_snapshot_receipts": snapshots,
        "dedup_receipt": {
            "path": str(args.dedup_receipt.resolve()),
            "bytes": args.dedup_receipt.stat().st_size,
            "sha256": sha256_file(args.dedup_receipt),
        },
        "pool_corpus_receipt": expected_pool,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"ok": True, "documents": rows, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Finalize shard-local contamination evidence into publishable HF tables."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--queries-summary", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    args = parse_args()
    final_output_dir = args.output_dir
    if final_output_dir.exists():
        raise FileExistsError(final_output_dir)
    staging_dir = final_output_dir.with_name(final_output_dir.name + f".staging.{os.getpid()}")
    staging_dir.mkdir(parents=True)
    args.output_dir = staging_dir
    queries_sha256 = sha256_file(args.queries_jsonl)
    manifest_sha256 = sha256_file(args.corpus_manifest)
    publication_sha256 = sha256_file(args.publication_receipt)
    corpus_manifest = json.loads(args.corpus_manifest.read_text())
    publication = json.loads(args.publication_receipt.read_text())
    if publication.get("manifest", {}).get("sha256") != manifest_sha256:
        raise ValueError("publication receipt is not bound to the corpus manifest")

    matches: list[dict[str, Any]] = []
    shard_receipts: list[dict[str, Any]] = []
    for item in corpus_manifest["files"]:
        stem = Path(item["path"]).stem
        receipt_path = args.scan_root / "shards" / f"{stem}.receipt.json"
        match_path = args.scan_root / "shards" / f"{stem}.matches.jsonl"
        if not receipt_path.is_file() or not match_path.is_file():
            raise FileNotFoundError(f"missing scan artifacts for {item['path']}")
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("status") != "passed"
            or receipt.get("queries_sha256") != queries_sha256
            or receipt.get("corpus_manifest_sha256") != manifest_sha256
            or receipt.get("publication_receipt_sha256") != publication_sha256
            or receipt.get("input", {}).get("path") != item["path"]
            or receipt.get("input", {}).get("sha256") != item["sha256"]
            or receipt.get("output", {}).get("sha256") != sha256_file(match_path)
        ):
            raise ValueError(f"invalid scan receipt for {item['path']}")
        shard_receipts.append(
            {"path": f"shards/{receipt_path.name}", "sha256": sha256_file(receipt_path)}
        )
        with match_path.open(encoding="utf-8") as handle:
            matches.extend(json.loads(line) for line in handle if line.strip())
    matches.sort(
        key=lambda row: (
            row["benchmark"],
            row["evaluation_unit_id"],
            row["source_dataset"],
            row["source_doc_id"],
            row["dataset_shard"],
            row["dataset_row_index"],
        )
    )

    queries = [json.loads(line) for line in args.queries_jsonl.read_text().splitlines() if line.strip()]
    by_unit: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for match in matches:
        by_unit[(match["benchmark"], match["evaluation_unit_id"])].append(match)
    summaries = []
    excluded_example_ids: list[dict[str, str]] = []
    for query in queries:
        key = (query["benchmark"], query["evaluation_unit_id"])
        unit_matches = by_unit.get(key, [])
        strong = [row for row in unit_matches if row["recommended_exclusion"]]
        recommended = bool(strong)
        if recommended:
            for example_id in query["discount_example_ids"]:
                excluded_example_ids.append(
                    {
                        "benchmark": query["benchmark"],
                        "example_id": str(example_id),
                        "evaluation_unit_id": query["evaluation_unit_id"],
                    }
                )
        summaries.append(
            {
                "benchmark": query["benchmark"],
                "evaluation_unit_id": query["evaluation_unit_id"],
                "example_id": query["example_id"],
                "discount_example_ids": query["discount_example_ids"],
                "query_kind": query["query_kind"],
                "subject": query.get("subject"),
                "matched_documents": len(unit_matches),
                "strong_matched_documents": len(strong),
                "candidate_only_documents": len(unit_matches) - len(strong),
                "matched_source_datasets": sorted({row["source_dataset"] for row in unit_matches}),
                "recommended_exclusion": recommended,
            }
        )
    excluded_example_ids.sort(key=lambda row: (row["benchmark"], row["example_id"]))

    match_path = args.output_dir / "qa_document_line_matches.parquet"
    summary_path = args.output_dir / "benchmark_question_summary.parquet"
    exclusion_path = args.output_dir / "recommended_excluded_example_ids.jsonl"
    if matches:
        pq.write_table(pa.Table.from_pylist(matches), match_path, compression="zstd")
    else:
        pq.write_table(pa.table({"benchmark": pa.array([], type=pa.string())}), match_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(summaries), summary_path, compression="zstd")
    exclusion_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in excluded_example_ids),
        encoding="utf-8",
    )

    benchmark_counts: dict[str, dict[str, int]] = {}
    for row in summaries:
        block = benchmark_counts.setdefault(
            row["benchmark"],
            {"evaluation_units": 0, "matched_units": 0, "recommended_excluded_units": 0, "recommended_excluded_scored_examples": 0},
        )
        block["evaluation_units"] += 1
        block["matched_units"] += int(row["matched_documents"] > 0)
        block["recommended_excluded_units"] += int(row["recommended_exclusion"])
        if row["recommended_exclusion"]:
            block["recommended_excluded_scored_examples"] += len(row["discount_example_ids"])

    audit = {
        "schema_version": "greek_benchmark_contamination_audit_v1",
        "status": "passed",
        "dataset": {
            "repository_id": publication["repo_id"],
            "revision": publication["commit_sha"],
            "rows": corpus_manifest["counts"]["rows"],
            "shards": corpus_manifest["counts"]["shards"],
            "manifest_sha256": manifest_sha256,
            "publication_receipt_sha256": publication_sha256,
        },
        "queries": {
            "sha256": queries_sha256,
            "summary_sha256": sha256_file(args.queries_summary),
            "evaluation_units": len(queries),
            "scored_examples": sum(len(query["discount_example_ids"]) for query in queries),
        },
        "method": {
            "normalization": "NFKC, combining-mark removal, casefold, Unicode word tokens",
            "question_rule": "8-token k-gram; exact full normalized stem for 3-7-token questions",
            "strong_rule": "question/source anchor plus correct-answer/paired-source anchor in the proximity window",
            "discount_policy": "only strong matches are recommended for post-hoc exclusion; question-only matches remain candidates",
            "line_numbering": "1-based lines within the published document text",
        },
        "counts": {
            "match_rows": len(matches),
            "recommended_excluded_scored_examples": len(excluded_example_ids),
            "benchmarks": benchmark_counts,
        },
        "shard_receipts": shard_receipts,
    }
    audit_path = args.output_dir / "audit_receipt.json"
    write_json(audit_path, audit)
    readme = f"""---
license: other
---

# Benchmark contamination evidence

This directory is an auxiliary audit for `{publication['repo_id']}` at immutable
revision `{publication['commit_sha']}`. It does not alter the dataset rows.

`qa_document_line_matches.parquet` identifies benchmark evaluation units and
the published source document, shard, row, and 1-based text lines that matched.
`benchmark_question_summary.parquet` includes every audited evaluation unit,
including those with no match. `recommended_excluded_example_ids.jsonl` is the
strict post-hoc score filter.

The strict recommendation requires two nearby human-authored surfaces: a
question/source anchor and the correct-answer or paired-source anchor. A
question-only match is published as a candidate but is **not** automatically
discounted. OYXOY matching uses its original premises, hypotheses, usage
examples, and definitions rather than evaluator-authored Greek prompt text.

All results are receipt-bound in `audit_receipt.json`. The inaccessible gated
Protipa benchmark is not represented in this version of the audit.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    files = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.name == "publish_manifest.json":
            continue
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    publish_manifest = {
        "schema_version": "greek_benchmark_contamination_hf_payload_v1",
        "status": "passed",
        "repo_id": publication["repo_id"],
        "dataset_revision_audited": publication["commit_sha"],
        "target_path": "benchmark_contamination/native_greek_suite_v1",
        "files": files,
    }
    write_json(args.output_dir / "publish_manifest.json", publish_manifest)
    os.replace(args.output_dir, final_output_dir)
    print(json.dumps(audit["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

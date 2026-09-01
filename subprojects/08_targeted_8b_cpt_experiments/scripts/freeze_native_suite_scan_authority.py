#!/usr/bin/env python3
"""Freeze the reproducible native-suite query authority for training scans."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


EXPECTED_QUERY_SHA256 = "ef9d601b8f91c6845818b9584c6634a13337c77b07e3f101f755a4884634c0eb"
EXPECTED_EXAMPLES_SHA256 = "51e1dc1565e44d891173a50b787bbe0a90916cf6e8d36c53b9c61106f65604df"
EXPECTED_ITEMS = 80_446
EXPECTED_SCORED_EXAMPLES = 83_970
EXPECTED_BENCHMARKS = {
    "asep_mcqa", "demosqa", "gpcr", "medical_mcqa", "oyxoy_metaphor",
    "oyxoy_nli", "oyxoy_wic", "oyxoy_wsd_definition",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--queries-summary", type=Path, required=True)
    parser.add_argument("--frozen-examples", type=Path, required=True)
    parser.add_argument("--builder-script", type=Path, required=True)
    parser.add_argument("--scanner-script", type=Path, required=True)
    parser.add_argument("--published-audit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable authority receipt exists: {args.output}")
    for path in (
        args.queries_jsonl, args.queries_summary, args.frozen_examples,
        args.builder_script, args.scanner_script, args.published_audit_receipt,
    ):
        require(path.is_file(), f"native-suite authority input missing: {path}")
    queries = file_binding(args.queries_jsonl)
    examples = file_binding(args.frozen_examples)
    require(queries["sha256"] == EXPECTED_QUERY_SHA256, "native-suite query SHA-256 drift")
    require(examples["sha256"] == EXPECTED_EXAMPLES_SHA256, "frozen native-suite examples drift")
    summary = read_json(args.queries_summary)
    require(summary.get("schema") == "greek-benchmark-decontam-query-summary-v2", "query-summary schema drift")
    require(summary.get("output_sha256") == EXPECTED_QUERY_SHA256, "query-summary output binding drift")
    require(summary.get("frozen_examples_sha256") == EXPECTED_EXAMPLES_SHA256, "query-summary examples binding drift")
    require(int(summary.get("total_items", -1)) == EXPECTED_ITEMS, "native-suite query count drift")
    require(int(summary.get("total_scored_examples", -1)) == EXPECTED_SCORED_EXAMPLES, "native-suite scored-example count drift")
    require(set(summary.get("benchmarks", {})) == EXPECTED_BENCHMARKS, "native-suite benchmark inventory drift")
    require(sum(1 for line in args.queries_jsonl.open(encoding="utf-8") if line.strip()) == EXPECTED_ITEMS, "query physical-row drift")
    published = read_json(args.published_audit_receipt)
    require(published.get("status") == "passed", "published native-suite audit did not pass")
    require(published.get("queries", {}).get("sha256") == EXPECTED_QUERY_SHA256, "published audit query binding drift")

    payload = {
        "schema_version": "apertus_native_suite_scan_authority_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "queries": {**queries, "rows": EXPECTED_ITEMS},
        "queries_summary": file_binding(args.queries_summary),
        "frozen_examples": examples,
        "builder": file_binding(args.builder_script),
        "scanner": file_binding(args.scanner_script),
        "published_audit_receipt": file_binding(args.published_audit_receipt),
        "benchmark_ids": sorted(EXPECTED_BENCHMARKS),
        "scored_examples": EXPECTED_SCORED_EXAMPLES,
        "protipa_included": False,
        "matching_contract": {
            "k": 8,
            "minimum_short_question_tokens": 3,
            "max_gap_tokens": 50,
            "max_gap_tokens_short": 5,
            "strong_rule": "question_or_source_anchor_plus_correct_or_paired_source_anchor",
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

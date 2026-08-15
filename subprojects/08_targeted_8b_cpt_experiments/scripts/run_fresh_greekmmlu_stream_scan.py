#!/usr/bin/env python3
"""Run and receipt the exact historical GreekMMLU scan on one rebuilt stream."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-name", required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--query-receipt", type=Path, required=True)
    parser.add_argument("--scanner-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    require(args.stream_name in {
        "hplt", "openarchives", "hplt_post", "openarchives_post",
        "foreign_replay", "old_greek_replay", "replay_selected", "replay_selected_post",
    }, "unknown stream")
    require(args.workers >= 1, "workers must be positive")
    require(not args.output_root.exists(), f"immutable scan root exists: {args.output_root}")
    require(not args.output_receipt.exists(), f"immutable receipt exists: {args.output_receipt}")
    query_receipt = read_json(args.query_receipt)
    require(query_receipt.get("status") == "passed", "query receipt did not pass")
    require(query_receipt["queries"]["sha256"] == file_binding(args.queries_jsonl)["sha256"], "query hash drift")
    args.output_root.mkdir(parents=True)
    clean = args.output_root / "clean.jsonl"
    dropped = args.output_root / "dropped.jsonl"
    command = [
        sys.executable,
        str(args.scanner_script),
        "--queries-jsonl", str(args.queries_jsonl),
        "--corpus-jsonl", str(args.corpus_jsonl),
        "--output-dir", str(args.output_root / "scanner"),
        "--benchmark", "greekmmlu",
        "--k", "8",
        "--max-gap-tokens", "50",
        "--max-gap-tokens-short", "5",
        "--direction", "after",
        "--primary-rule", "correct_only",
        "--max-q-kgram-digit-fraction", "0.5",
        "--n-workers", str(args.workers),
    ]
    if not args.audit_only:
        command.extend(["--filter-output-clean", str(clean), "--filter-output-dropped", str(dropped)])
    subprocess.run(command, check=True)
    audits = list((args.output_root / "scanner").glob("greekmmlu_dclm_audit_*.json"))
    require(len(audits) == 1, "scanner did not produce exactly one audit")
    audit = read_json(audits[0])
    require(audit["k"] == 8, "scanner k drift")
    require(audit["max_gap_tokens"] == 50, "scanner long-gap drift")
    require(audit["max_gap_tokens_short"] == 5, "scanner short-gap drift")
    require(audit["direction"] == "after", "scanner direction drift")
    require(audit["headline"]["rule"] == "correct_only", "scanner primary-rule drift")
    filt = audit.get("filter")
    if args.audit_only:
        require(filt is None, "audit-only scan unexpectedly rewrote the corpus")
    else:
        require(isinstance(filt, dict), "filter summary is missing")
        require(filt["input_rows"] == filt["clean_rows"] + filt["dropped_rows"], "scan row accounting drift")
    if args.stream_name.endswith("_post"):
        require(args.audit_only, "post-filter GreekMMLU scan must be audit-only")
        require(int(audit["headline"]["item_doc_pairs"]) == 0, "post-filter GreekMMLU scan was not clean")
    payload = {
        "schema_version": "apertus_fresh_greekmmlu_stream_scan_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stream": args.stream_name,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "input": file_binding(args.corpus_jsonl),
        "queries": file_binding(args.queries_jsonl),
        "query_receipt": file_binding(args.query_receipt),
        "scanner": file_binding(args.scanner_script),
        "command_contract": {
            "k": 8,
            "max_gap_tokens": 50,
            "max_gap_tokens_short": 5,
            "direction": "after",
            "primary_rule": "correct_only",
        },
        "audit": file_binding(audits[0]),
        "audit_only": args.audit_only,
        "clean": None if args.audit_only else {**file_binding(clean), "rows": int(filt["clean_rows"])},
        "dropped": None if args.audit_only else {**file_binding(dropped), "rows": int(filt["dropped_rows"])},
        "counts": {"item_doc_pairs": int(audit["headline"]["item_doc_pairs"])} if args.audit_only else filt,
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

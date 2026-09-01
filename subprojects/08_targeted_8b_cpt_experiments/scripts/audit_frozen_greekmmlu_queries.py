#!/usr/bin/env python3
"""Audit a deterministically regenerated GreekMMLU decontamination query file."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable receipt exists: {args.output}")
    contract = read_json(args.contract)
    require(contract.get("schema_version") == "apertus_greekmmlu_query_regeneration_v1", "query contract schema drift")
    require(contract.get("status") == "frozen", "query regeneration contract is not frozen")
    expected = contract["dataset"]
    sources = {}
    for name, digest in contract["source_files"].items():
        candidates = [args.source_root / name, args.source_root / "eval" / name]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        require(path is not None, f"query source missing: {name}")
        require(sha256_file(path) == digest, f"query source drift: {name}")
        sources[name] = file_binding(path)
    seen = set()
    rows = 0
    canonical = hashlib.sha256()
    with args.queries.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(row.get("schema") == "greek-mcq-decontam-query-v1", f"query schema drift at line {line_number}")
            require(row.get("benchmark") == "greekmmlu", f"non-GreekMMLU query at line {line_number}")
            require(row.get("dataset_repo_id") == expected["repo_id"], f"dataset id drift at line {line_number}")
            require(row.get("dataset_revision") == expected["revision"], f"dataset revision drift at line {line_number}")
            require(row.get("split") == expected["split"], f"split drift at line {line_number}")
            example_id = str(row.get("example_id", ""))
            require(example_id and example_id not in seen, f"duplicate or missing example id: {example_id!r}")
            seen.add(example_id)
            choices = row.get("choices")
            answer = row.get("answer_index")
            require(isinstance(choices, list) and isinstance(answer, int) and 0 <= answer < len(choices), f"answer drift at line {line_number}")
            canonical.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            canonical.update(b"\0")
            rows += 1
    require(rows == expected["expected_questions"], f"GreekMMLU query count drift: {rows}")
    summary = read_json(args.summary)
    require(summary.get("total_items") == rows, "query summary item-count drift")
    require(summary.get("output_jsonl_sha256") == sha256_file(args.queries), "query summary hash drift")
    payload = {
        "schema_version": "apertus_frozen_greekmmlu_queries_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "contract": file_binding(args.contract),
        "sources": sources,
        "queries": {**file_binding(args.queries), "rows": rows, "unique_example_ids": len(seen)},
        "summary": file_binding(args.summary),
        "canonical_row_digest": canonical.hexdigest(),
        "named_difference": contract["named_difference"],
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

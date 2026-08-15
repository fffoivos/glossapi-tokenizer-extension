#!/usr/bin/env python3
"""Filter exact strong native-suite matches from normalized replay without deduplication."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pyarrow.parquet as pq

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--exclusions-parquet", type=Path, required=True)
    parser.add_argument("--exclusions-receipt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--dropped-jsonl", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_jsonl, args.dropped_jsonl, args.output_receipt):
        require(not output.exists(), f"immutable native filter output exists: {output}")
    upstream = read_json(args.input_receipt)
    require(upstream.get("status") == "passed", "normalized replay input did not pass")
    require(upstream.get("output", {}).get("sha256") == sha256_file(args.input_jsonl), "normalized replay binding drift")
    exclusion_receipt = read_json(args.exclusions_receipt)
    require(exclusion_receipt.get("status") == "passed", "native exclusion receipt did not pass")
    expected = exclusion_receipt["exclusions"]
    require(expected["sha256"] == sha256_file(args.exclusions_parquet), "native exclusion SHA drift")
    table = pq.read_table(args.exclusions_parquet, columns=["source_dataset", "source_doc_id", "document_text_sha256"])
    exclusions = set(zip(table["source_dataset"].to_pylist(), table["source_doc_id"].to_pylist(), table["document_text_sha256"].to_pylist(), strict=True))
    require(len(exclusions) == int(expected["rows"]), "native exclusion tuple count drift")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temps = []
    descriptors = []
    for output in (args.output_jsonl, args.dropped_jsonl):
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
        descriptors.append(descriptor); temps.append(Path(name))
    counts: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = {}
    try:
        with args.input_jsonl.open(encoding="utf-8") as source, os.fdopen(descriptors[0], "w", encoding="utf-8") as kept, os.fdopen(descriptors[1], "w", encoding="utf-8") as dropped:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                source_dataset = str(row.get("source_dataset", ""))
                source_doc_id = str(row.get("source_doc_id", ""))
                require(isinstance(text, str) and text and source_dataset and source_doc_id, f"invalid normalized replay row: {line_number}")
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                require(row.get("document_text_sha256") == digest, f"normalized text SHA drift: {line_number}")
                bucket = per_source.setdefault(source_dataset, Counter())
                counts["input_rows"] += 1; bucket["input_rows"] += 1
                if (source_dataset, source_doc_id, digest) in exclusions:
                    dropped.write(json.dumps({"source_dataset": source_dataset, "source_doc_id": source_doc_id, "document_text_sha256": digest, "exclusion_reason": "native_suite_strong_content_match"}, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["dropped_rows"] += 1; bucket["dropped_rows"] += 1
                else:
                    kept.write(line if line.endswith("\n") else line + "\n")
                    counts["kept_rows"] += 1; bucket["kept_rows"] += 1
            kept.flush(); os.fsync(kept.fileno()); dropped.flush(); os.fsync(dropped.fileno())
        require(counts["input_rows"] == int(upstream["output"]["rows"]), "native filter input row-count drift")
        require(counts["input_rows"] == counts["kept_rows"] + counts["dropped_rows"], "native filter accounting drift")
        for temporary, output in zip(temps, (args.output_jsonl, args.dropped_jsonl), strict=True):
            os.link(temporary, output); temporary.unlink()
    except BaseException:
        for descriptor in descriptors:
            try: os.close(descriptor)
            except OSError: pass
        for temporary in temps: temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema_version": "apertus_replay_native_suite_filter_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"), "nodes": int(os.environ.get("SLURM_NNODES", "0"))},
        "executing_code_bundle": executing_code_bundle(),
        "input": file_binding(args.input_jsonl), "input_receipt": file_binding(args.input_receipt),
        "exclusions": file_binding(args.exclusions_parquet), "exclusions_receipt": file_binding(args.exclusions_receipt),
        "output": {**file_binding(args.output_jsonl), "rows": counts["kept_rows"]},
        "dropped": {**file_binding(args.dropped_jsonl), "rows": counts["dropped_rows"]},
        "counts": dict(counts), "per_source": {name: dict(counter) for name, counter in sorted(per_source.items())},
        "invariants": {"row_order_preserved": True, "nonmatching_row_multiplicity_preserved": True, "text_transformed": False, "additional_deduplication": False},
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

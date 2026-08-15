#!/usr/bin/env python3
"""Remove only frozen validation-panel exact-text matches from rebuilt replay."""

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
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--validation-exclusions", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--dropped-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_jsonl, args.dropped_jsonl, args.output_manifest):
        require(not output.exists(), f"immutable replay exclusion output exists: {output}")
    upstream = read_json(args.input_manifest)
    require(Path(upstream["output"]).resolve() == args.input_jsonl.resolve(), "replay mix manifest output drift")
    validation = read_json(args.validation_receipt)
    require(validation.get("schema_version") == "apertus_hard_h_to_g_reused_validation_panels_v1", "validation receipt schema drift")
    require(validation.get("status") == "passed", "validation receipt did not pass")
    expected = validation["training_exclusions"]
    require(sha256_file(args.validation_exclusions) == expected["sha256"], "validation exclusions SHA-256 drift")
    exclusion_table = pq.read_table(args.validation_exclusions, columns=["document_text_sha256"])
    exclusions = {str(value) for value in exclusion_table["document_text_sha256"].to_pylist()}
    require(len(exclusions) == int(expected["rows"]), "validation exclusion count drift")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    descriptors: list[int] = []
    for output in (args.output_jsonl, args.dropped_jsonl):
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
        descriptors.append(descriptor)
        temporary_paths.append(Path(name))
    counts: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = {}
    try:
        with args.input_jsonl.open(encoding="utf-8") as source, \
                os.fdopen(descriptors[0], "w", encoding="utf-8") as kept, \
                os.fdopen(descriptors[1], "w", encoding="utf-8") as dropped:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                source_name = str(row.get("source", ""))
                doc_id = str(row.get("doc_id", ""))
                require(isinstance(text, str) and text and source_name and doc_id, f"invalid replay row: {line_number}")
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                bucket = per_source.setdefault(source_name, Counter())
                counts["input_rows"] += 1
                bucket["input_rows"] += 1
                if digest in exclusions:
                    dropped.write(json.dumps({
                        "source": source_name,
                        "doc_id": doc_id,
                        "document_text_sha256": digest,
                        "exclusion_reason": "reused_validation_panel_exact_text",
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["dropped_rows"] += 1
                    bucket["dropped_rows"] += 1
                else:
                    kept.write(line if line.endswith("\n") else line + "\n")
                    counts["kept_rows"] += 1
                    bucket["kept_rows"] += 1
            kept.flush(); os.fsync(kept.fileno())
            dropped.flush(); os.fsync(dropped.fileno())
        require(counts["input_rows"] == int(upstream["actual_rows"]), "replay input row-count drift")
        require(counts["input_rows"] == counts["kept_rows"] + counts["dropped_rows"], "replay validation exclusion accounting drift")
        for temporary, output in zip(temporary_paths, (args.output_jsonl, args.dropped_jsonl), strict=True):
            os.link(temporary, output)
            temporary.unlink()
    except BaseException:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise

    payload = {
        "schema_version": "apertus_replay_validation_exclusion_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "input": file_binding(args.input_jsonl),
        "input_manifest": file_binding(args.input_manifest),
        "validation_receipt": file_binding(args.validation_receipt),
        "validation_exclusions": file_binding(args.validation_exclusions),
        "output": str(args.output_jsonl.resolve()),
        "output_binding": file_binding(args.output_jsonl),
        "dropped_ledger": file_binding(args.dropped_jsonl),
        "actual_rows": counts["kept_rows"],
        "counts": dict(counts),
        "per_source": {
            name: {**dict(counter), "rows": counter["kept_rows"]}
            for name, counter in sorted(per_source.items())
        },
        "invariants": {
            "row_order_preserved": True,
            "nonmatching_row_multiplicity_preserved": True,
            "text_or_metadata_transformed": False,
            "additional_deduplication": False,
            "only_exact_validation_text_matches_removed": True,
        },
    }
    write_json_atomic(args.output_manifest, payload)
    print(args.output_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

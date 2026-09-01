#!/usr/bin/env python3
"""Materialize normalized replay as immutable Parquet shards for the canonical scanner."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


SCHEMA = pa.schema([
    ("source_dataset", pa.string()),
    ("source_doc_id", pa.string()),
    ("text", pa.string()),
    ("source_metadata_json", pa.string()),
])


def write_shard(path: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, path)
    return {"path": path.name, "bytes": path.stat().st_size, "rows": len(rows), "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument(
        "--zero-greekmmlu-receipt",
        type=Path,
        help="required for post-Stage-B materialization; must bind this input and prove zero GreekMMLU matches",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    require(args.shards >= 1, "shard count must be positive")
    require(not args.output_root.exists(), f"immutable scan corpus exists: {args.output_root}")
    require(not args.output_receipt.exists(), f"immutable receipt exists: {args.output_receipt}")
    upstream = read_json(args.input_receipt)
    require(upstream.get("status") == "passed", "normalized replay input receipt did not pass")
    input_binding = file_binding(args.input_jsonl)
    upstream_output = upstream.get("output") if isinstance(upstream.get("output"), dict) else upstream.get("clean")
    require(isinstance(upstream_output, dict), "upstream receipt has no bound output")
    require(upstream_output.get("sha256") == input_binding["sha256"], "normalized replay input binding drift")
    expected_rows = int(upstream_output["rows"])
    require(expected_rows > 0, "normalized replay is empty")
    zero_greekmmlu_binding = None
    if args.zero_greekmmlu_receipt is not None:
        zero_greekmmlu = read_json(args.zero_greekmmlu_receipt)
        require(
            zero_greekmmlu.get("schema_version") == "apertus_fresh_greekmmlu_stream_scan_v1",
            "post-Stage-B GreekMMLU receipt schema drift",
        )
        require(zero_greekmmlu.get("status") == "passed", "post-Stage-B GreekMMLU receipt did not pass")
        require(zero_greekmmlu.get("stream") == "replay_selected_post", "post-Stage-B GreekMMLU stream drift")
        require(zero_greekmmlu.get("audit_only") is True, "post-Stage-B GreekMMLU scan rewrote the stream")
        require(
            zero_greekmmlu.get("input", {}).get("sha256") == input_binding["sha256"],
            "post-Stage-B GreekMMLU receipt input drift",
        )
        require(
            int(zero_greekmmlu.get("counts", {}).get("item_doc_pairs", -1)) == 0,
            "post-Stage-B GreekMMLU scan was not clean",
        )
        zero_greekmmlu_binding = file_binding(args.zero_greekmmlu_receipt)
    rows_per_shard = math.ceil(expected_rows / args.shards)

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent))
    files: list[dict[str, object]] = []
    buffer: list[dict[str, str]] = []
    rows = 0
    try:
        with args.input_jsonl.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                source_dataset = row.get("source_dataset")
                source_doc_id = row.get("source_doc_id")
                require(isinstance(text, str) and text, f"{line_number}: missing text")
                require(source_dataset not in (None, "") and source_doc_id not in (None, ""), f"{line_number}: missing identity")
                metadata = json.dumps({
                    "adapter_source": row.get("adapter_source"),
                    "adapter_row_index": row.get("adapter_row_index"),
                    "normalized_doc_id": row.get("doc_id"),
                    "document_text_sha256": row.get("document_text_sha256"),
                }, ensure_ascii=False, sort_keys=True)
                buffer.append({
                    "source_dataset": str(source_dataset),
                    "source_doc_id": str(source_doc_id),
                    "text": text,
                    "source_metadata_json": metadata,
                })
                rows += 1
                if len(buffer) == rows_per_shard:
                    files.append(write_shard(staging / f"part-{len(files):05d}.parquet", buffer))
                    buffer = []
        if buffer:
            files.append(write_shard(staging / f"part-{len(files):05d}.parquet", buffer))
        require(rows == expected_rows, f"normalized replay row drift: {rows} != {expected_rows}")
        require(1 <= len(files) <= args.shards, "unexpected realized shard count")
        manifest = {
            "schema_version": "apertus_replay_training_scan_corpus_manifest_v1",
            "status": "passed",
            "files": files,
            "counts": {"rows": rows, "shards": len(files)},
        }
        manifest_path = staging / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        publication = {
            "schema_version": "apertus_local_training_scan_publication_binding_v1",
            "status": "passed",
            "repo_id": "local/apertus-selected-replay",
            "commit_sha": hashlib.sha256((input_binding["sha256"] + "\0native-suite-scan-v1").encode()).hexdigest(),
            "manifest": file_binding(manifest_path),
        }
        publication_path = staging / "publication_receipt.json"
        write_json_atomic(publication_path, publication)
        os.replace(staging, args.output_root)
    except BaseException:
        for path in sorted(staging.glob("*")):
            path.unlink(missing_ok=True)
        staging.rmdir()
        raise

    payload = {
        "schema_version": "apertus_replay_training_scan_corpus_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "input": input_binding,
        "input_receipt": file_binding(args.input_receipt),
        "zero_greekmmlu_receipt": zero_greekmmlu_binding,
        "root": str(args.output_root.resolve()),
        "manifest": file_binding(args.output_root / "manifest.json"),
        "publication_receipt": file_binding(args.output_root / "publication_receipt.json"),
        "counts": {"rows": rows, "shards": len(files)},
        "files": files,
        "invariants": {
            "row_order_preserved": True,
            "row_multiplicity_preserved": True,
            "text_transformed": False,
            "additional_deduplication": False,
        },
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

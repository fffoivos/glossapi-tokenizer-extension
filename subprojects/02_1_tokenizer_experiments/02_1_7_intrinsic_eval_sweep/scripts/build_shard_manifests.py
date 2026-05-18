"""Pre-shard train.parquet + train_manifest into K paired parquet files.

Per FIRING_COUNT_PLAN.md v2.4 §5 reviewer-finding fix (#5):
  gcloud storage cp does not support row-group slice copies of a single
  parquet. Each worker getting its own physical file is simpler and more
  reliable than range-read coordination across K instances.

Strategy:
  1. Read train.parquet metadata (no full download) — get row count,
     row-group boundaries, total rows.
  2. Read manifest (csv or parquet) — must have same row count.
  3. Split into K shards by row index, balanced by row count
     (greedy bin-pack across row groups so each shard ends on a row-
     group boundary if practical).
  4. For each shard N: pyarrow-read the assigned row groups from
     train.parquet (text column only) and the corresponding manifest
     rows (source_dataset column only) → write paired
     {shard_NN_text.parquet, shard_NN_manifest.parquet}.
  5. Upload each pair to GCS:
       {gcs_out_prefix}/shards/shard_NN_text.parquet
       {gcs_out_prefix}/shards/shard_NN_manifest.parquet
  6. Write shard manifest summary at
       {gcs_out_prefix}/shards/shards_index.json

Input paths can be local or gs:// (uses pyarrow.fs).
Output uses `gcloud storage cp` (so the runner must be authenticated).

Memory: ~5 GB per shard staged locally before upload. K=8 means ~5 GB
peak. Use --local-tmp to redirect.

Failure semantics: fail-hard on any I/O error. Idempotent — re-running
overwrites GCS outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


def open_parquet(path: str) -> pq.ParquetFile:
    """Open a parquet file — local path or gs://. Uses pyarrow.fs for GCS."""
    if path.startswith("gs://"):
        import pyarrow.fs as pafs
        fs = pafs.GcsFileSystem()
        bucket_and_key = path[len("gs://"):]
        f = fs.open_input_file(bucket_and_key)
        return pq.ParquetFile(f)
    return pq.ParquetFile(path)


def read_manifest_sources(path: str) -> list[str]:
    """Read the entire source_dataset column into a list of strings.
    Supports .csv and .parquet.

    For very large manifests this materializes everything in memory.
    On a 14.4M-row manifest, that's ~500MB-1GB of strings (most are
    repeated). Acceptable for this one-time sharding step.
    """
    if path.startswith("gs://") and path.endswith(".csv"):
        # Stream CSV from GCS via gsutil cat into a tempfile to use pyarrow
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
            local = fh.name
        try:
            subprocess.run(["gcloud", "storage", "cp", path, local], check=True)
            return read_manifest_sources(local)
        finally:
            try: os.remove(local)
            except: pass

    if path.endswith(".parquet"):
        pf = open_parquet(path)
        if "source_dataset" not in pf.schema_arrow.names:
            raise ValueError(
                f"{path} has no 'source_dataset' column "
                f"(schema: {pf.schema_arrow.names[:6]})"
            )
        all_sources = []
        for rb in pf.iter_batches(batch_size=200_000, columns=["source_dataset"]):
            all_sources.extend(rb.column("source_dataset").to_pylist())
        return all_sources

    if path.endswith(".csv"):
        # PyArrow CSV reader is much faster than the stdlib csv module
        tbl = pacsv.read_csv(path)
        if "source_dataset" not in tbl.column_names:
            raise ValueError(
                f"{path} has no 'source_dataset' column "
                f"(columns: {tbl.column_names})"
            )
        return tbl.column("source_dataset").to_pylist()

    raise ValueError(f"unknown manifest format: {path}")


def assign_row_groups_to_shards(text_pf: pq.ParquetFile, k: int) -> list[list[int]]:
    """Greedy bin-pack: assign row groups to K shards by row count.

    Returns list of K lists of row group indices.
    """
    rg_rows = [text_pf.metadata.row_group(i).num_rows for i in range(text_pf.num_row_groups)]
    # Sort row groups by size desc, then greedily assign each to the
    # smallest shard. Classic LPT scheduling.
    order = sorted(range(len(rg_rows)), key=lambda i: -rg_rows[i])
    shard_assignments: list[list[int]] = [[] for _ in range(k)]
    shard_loads = [0] * k
    for rg in order:
        target = min(range(k), key=lambda s: shard_loads[s])
        shard_assignments[target].append(rg)
        shard_loads[target] += rg_rows[rg]
    # Sort each shard's row group list (so shard reads in original order)
    for s in shard_assignments:
        s.sort()
    return shard_assignments


def get_row_group_text_row_range(text_pf: pq.ParquetFile,
                                  rg_indices: list[int]) -> list[tuple[int, int]]:
    """Return list of (global_start_row, global_end_row) for each rg in indices.

    Used to compute which slice of the manifest belongs to this shard's
    row groups.
    """
    # Build cumulative row count per row group
    cum = 0
    rg_starts = []
    for i in range(text_pf.num_row_groups):
        rg_starts.append(cum)
        cum += text_pf.metadata.row_group(i).num_rows
    return [(rg_starts[i], rg_starts[i] + text_pf.metadata.row_group(i).num_rows)
            for i in rg_indices]


def write_shard(shard_idx: int, total_shards: int, text_pf: pq.ParquetFile,
                manifest_sources: list[str], rg_indices: list[int],
                local_tmp: str, gcs_out_prefix: str) -> dict:
    """Write one shard's text + manifest paired parquets to GCS."""
    tag = f"shard_{shard_idx:02d}_of_{total_shards:02d}"
    local_text = os.path.join(local_tmp, f"{tag}_text.parquet")
    local_manifest = os.path.join(local_tmp, f"{tag}_manifest.parquet")

    # Read text column from assigned row groups
    text_table = text_pf.read_row_groups(rg_indices, columns=["text"])
    pq.write_table(text_table, local_text, compression="zstd")

    # Compute the corresponding manifest row range(s) and slice
    ranges = get_row_group_text_row_range(text_pf, rg_indices)
    manifest_slice = []
    for start, end in ranges:
        manifest_slice.extend(manifest_sources[start:end])
    # Sanity: matches text row count
    if len(manifest_slice) != text_table.num_rows:
        raise RuntimeError(
            f"row count mismatch: text={text_table.num_rows} "
            f"manifest_slice={len(manifest_slice)}"
        )
    manifest_table = pa.table({
        "source_dataset": pa.array(manifest_slice, type=pa.string())
    })
    pq.write_table(manifest_table, local_manifest, compression="zstd")

    # Upload
    text_gcs = f"{gcs_out_prefix}/shards/{tag}_text.parquet"
    manifest_gcs = f"{gcs_out_prefix}/shards/{tag}_manifest.parquet"
    subprocess.run(
        ["gcloud", "storage", "cp", local_text, text_gcs],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["gcloud", "storage", "cp", local_manifest, manifest_gcs],
        check=True, capture_output=True, text=True,
    )
    text_size = os.path.getsize(local_text)
    manifest_size = os.path.getsize(local_manifest)
    # Clean up local
    try: os.remove(local_text)
    except: pass
    try: os.remove(local_manifest)
    except: pass
    return {
        "shard": shard_idx,
        "total_shards": total_shards,
        "rg_indices": rg_indices,
        "rows": text_table.num_rows,
        "text_size_bytes": text_size,
        "manifest_size_bytes": manifest_size,
        "text_gcs": text_gcs,
        "manifest_gcs": manifest_gcs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-parquet", required=True,
                    help="train.parquet (local path or gs://)")
    ap.add_argument("--manifest", required=True,
                    help="train_manifest.csv or .parquet (local or gs://)")
    ap.add_argument("--k", type=int, required=True,
                    help="number of shards (= number of workers)")
    ap.add_argument("--gcs-out-prefix", required=True,
                    help="e.g. gs://testbucketglossapi/firing_counts_TS")
    ap.add_argument("--local-tmp", default="/tmp")
    args = ap.parse_args()

    t0 = time.time()

    print(f"[shard] opening {args.text_parquet} ...", flush=True)
    text_pf = open_parquet(args.text_parquet)
    n_rows = text_pf.metadata.num_rows
    n_rg = text_pf.num_row_groups
    print(f"  rows: {n_rows:,}  row_groups: {n_rg}", flush=True)

    print(f"[shard] reading manifest {args.manifest} ...", flush=True)
    manifest_sources = read_manifest_sources(args.manifest)
    print(f"  manifest rows: {len(manifest_sources):,}", flush=True)
    if len(manifest_sources) != n_rows:
        print(
            f"FATAL: text rows ({n_rows:,}) != manifest rows "
            f"({len(manifest_sources):,})", file=sys.stderr
        )
        return 1

    print(f"[shard] assigning {n_rg} row groups across {args.k} shards ...",
          flush=True)
    shard_assignments = assign_row_groups_to_shards(text_pf, args.k)
    for i, rgs in enumerate(shard_assignments):
        rows_in_shard = sum(text_pf.metadata.row_group(rg).num_rows for rg in rgs)
        print(f"  shard {i}: {len(rgs)} row_groups, {rows_in_shard:,} rows",
              flush=True)

    Path(args.local_tmp).mkdir(parents=True, exist_ok=True)

    # Build all shards (sequential — pyarrow read_row_groups is GIL-released
    # internally and writing/uploading is I/O-bound; parallelism gains here
    # are marginal).
    shard_records = []
    for i in range(args.k):
        print(f"[shard] writing shard {i} ...", flush=True)
        rec = write_shard(
            shard_idx=i, total_shards=args.k,
            text_pf=text_pf, manifest_sources=manifest_sources,
            rg_indices=shard_assignments[i],
            local_tmp=args.local_tmp,
            gcs_out_prefix=args.gcs_out_prefix,
        )
        shard_records.append(rec)
        print(f"  ✓ shard {i}: {rec['rows']:,} rows, "
              f"text {rec['text_size_bytes']//1024//1024} MB, "
              f"manifest {rec['manifest_size_bytes']//1024} KB", flush=True)

    # Index file
    index = {
        "source_text_parquet": args.text_parquet,
        "source_manifest": args.manifest,
        "total_rows": n_rows,
        "total_row_groups": n_rg,
        "k_shards": args.k,
        "wall_seconds": round(time.time() - t0, 1),
        "shards": shard_records,
    }
    local_index = os.path.join(args.local_tmp, "shards_index.json")
    Path(local_index).write_text(json.dumps(index, indent=2))
    index_gcs = f"{args.gcs_out_prefix}/shards/shards_index.json"
    subprocess.run(
        ["gcloud", "storage", "cp", local_index, index_gcs],
        check=True, capture_output=True, text=True,
    )
    print(f"\n[shard] all done. index → {index_gcs}", flush=True)
    print(f"  total wall: {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[shard FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Coordinator step 05: stream worker shard outputs into per-source parquet
tables WITHOUT loading the entire union into memory.

Strategy:
- Download shard parquets from GCS bucket into PER-WORKER subdirs of a staging
  dir. Workers can legitimately produce files with the same basename; flattening
  into one dir causes silent overwrites and corrupted parquets.
- For each (corpus_id, source_id) group, iterate the shard files in that
  group and APPEND-write to artifacts/sources/<corpus>_<source>.parquet via
  pyarrow.ParquetWriter (streaming).
- Each shard is read once with pyarrow batched scan; never concat-materialised
  into a single in-memory frame.

Per review r4: previous version eagerly concat'd via polars; would OOM on
~20-40 GB of worker outputs.

Per 2026-05-18 lessons: download MUST go to per-worker subdirs; recursive glob
locates shards regardless of layout. Architecturally this script is best run
on a same-region joins-worker so the download isn't bandwidth-limited from
home — see scripts/coordinator/spin_up_joins_worker.sh.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SUB = Path(__file__).resolve().parents[2]
RUN_ID = (SUB / "manifests/CURRENT_RUN_ID").read_text().strip()
MANI = SUB / f"manifests/run_{RUN_ID}"
PIN = json.loads((MANI / "text_dedup_pin.json").read_text())
BUCKET = PIN["bucket"]
N_WORKERS = int(PIN.get("worker_count", 8))
ART_ROOT = SUB / "artifacts" / RUN_ID / "sources"
ART_ROOT.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = SUB / "_worker_downloads" / RUN_ID
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

CANON_SCHEMA = pa.schema([
    ("doc_key", pa.string()),
    ("source_dataset", pa.string()),
    ("source_doc_id", pa.string()),
    ("text_length", pa.int64()),
    ("token_count", pa.int64()),
    ("strict_exact_hash", pa.string()),
    ("relaxed_exact_hash", pa.string()),
    ("minhash_sig", pa.binary()),
    ("lsh_band_hashes", pa.binary()),
    ("corpus_id", pa.string()),
    ("source_id", pa.string()),
])


def main() -> int:
    print(f"[concat] downloading worker outputs from {BUCKET} (per-worker subdirs) ...")
    # Discover which worker_N/ subprefixes actually have output (some indices may
    # be absent if the run had fewer workers than N_WORKERS).
    list_r = subprocess.run(
        ["gcloud", "storage", "ls", f"{BUCKET}/"],
        capture_output=True, text=True,
    )
    if list_r.returncode:
        print("[concat] list stderr:", list_r.stderr[-500:], file=sys.stderr)
        return 2
    worker_prefixes = sorted(
        ln.strip() for ln in list_r.stdout.splitlines()
        if ln.strip().endswith("/") and "/worker_" in ln
    )
    print(f"[concat] {len(worker_prefixes)} worker subprefixes: {worker_prefixes}")

    # Download each worker's outputs to its own local subdir, in parallel.
    procs = []
    for prefix in worker_prefixes:
        # prefix like gs://.../worker_3/
        wname = prefix.rstrip("/").rsplit("/", 1)[-1]
        wdir = DOWNLOAD_DIR / wname
        wdir.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(
            ["gcloud", "storage", "cp", "-r", f"{prefix}*.parquet", str(wdir) + "/"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        procs.append((wname, p))
    cp_failed: list[tuple[str, int, str]] = []
    for wname, p in procs:
        rc = p.wait()
        tail = (p.stdout.read() if p.stdout else "")[-300:]
        if rc:
            cp_failed.append((wname, rc, tail))
            print(f"[concat] cp FAILED for {wname} rc={rc}: {tail}", file=sys.stderr)
    if cp_failed:
        print(f"[concat] FATAL: {len(cp_failed)} per-worker downloads failed; "
              f"workers={[w for w, _, _ in cp_failed]}", file=sys.stderr)
        return 5

    files = sorted(DOWNLOAD_DIR.rglob("*.parquet"))
    print(f"[concat] {len(files)} parquet shards on disk (recursive)")

    # Group shards by (corpus_id, source_id) — read first batch of each for labels.
    # ZERO TOLERANCE for unreadable parquets: any corruption indicates a bug in
    # the worker write path (e.g. 2026-05-18: basename collision races dropped
    # 95% of shards). Better to fail loudly + investigate than emit a
    # partial-data report that looks superficially clean.
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    bad: list[tuple[Path, str]] = []
    for f in files:
        try:
            tbl = pq.read_table(f, columns=["corpus_id", "source_id"], use_threads=True)
            if tbl.num_rows == 0:
                bad.append((f, "empty (0 rows)"))
                continue
            corpus = tbl.column("corpus_id")[0].as_py()
            source = tbl.column("source_id")[0].as_py()
            groups[(corpus, source)].append(f)
        except Exception as e:
            bad.append((f, str(e)[:200]))
    if bad:
        print(f"[concat] FATAL: {len(bad)} parquet(s) unreadable/empty:", file=sys.stderr)
        for f, msg in bad[:10]:
            print(f"  - {f}: {msg}", file=sys.stderr)
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more", file=sys.stderr)
        return 6

    summary = []
    for (corpus, source), group_files in groups.items():
        out = ART_ROOT / f"{corpus}_{source}.parquet"
        writer = None
        total_rows = 0
        for sf in group_files:
            pf = pq.ParquetFile(str(sf))
            for batch in pf.iter_batches(batch_size=20_000):
                tbl = pa.Table.from_batches([batch]).cast(CANON_SCHEMA, safe=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(out), CANON_SCHEMA, compression="zstd")
                writer.write_table(tbl)
                total_rows += batch.num_rows
        if writer is not None:
            writer.close()
        summary.append({"corpus_id": corpus, "source_id": source,
                        "shard_files": len(group_files), "rows": total_rows,
                        "out_path": str(out)})
        print(f"[concat] {corpus}/{source}: {len(group_files)} shards -> {total_rows:,} rows -> {out}")

    summary_path = ART_ROOT.parent / "concat_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

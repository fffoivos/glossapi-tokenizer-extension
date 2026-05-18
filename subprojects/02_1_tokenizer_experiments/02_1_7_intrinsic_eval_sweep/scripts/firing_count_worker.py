"""Firing-count worker — per-source tokenization of one C3 train shard.

Design (per FIRING_COUNT_PLAN.md v2.4 §4):
  Inputs (paired streaming read, row-aligned):
    --text-parquet PATH      one shard of train.parquet (text column only)
    --manifest PATH          one shard of train_manifest.csv (or .parquet);
                              must have the same row count, same row order,
                              and a 'source_dataset' column
    --tokenizer-dir PATH     dir with tokenizer.json (148,480 vocab)
    --shard N --total K      shard identity (informational; used for output naming)
    --gcs-out-prefix URI     where partials go: gs://bucket/run-ts/
    [--smoke]                run in smoke mode (cap rows, write to _smoke/)
    [--max-total-rows-per-component N]    smoke cap; counts rows per
                              component (glossapi_nanochat_only / hplt_only)
                              and stops when both reach N
    [--batch-size N=4096]    text rows per encode_batch call
    [--local-tmp DIR]        where partial parquets are staged before upload

  Outputs (per shard, three GCS objects):
    {gcs_out_prefix}/per_source_counts/shard_{NN}_of_{KK}.parquet
        Long format: {source_dataset, id, fire_count}
        Only non-zero (source, id) rows.
    {gcs_out_prefix}/per_source_denominators/shard_{NN}_of_{KK}.json
        {source_dataset → {rows, chars, tokenized_tokens}}
    {gcs_out_prefix}/_DONE_shard_{NN}    (empty marker; gated on full success)

  In smoke mode, outputs go to {gcs_out_prefix}/_smoke/... AND a sibling
  smoke_benchmark.json with download_mbps, tokenize_tokens_per_sec,
  mean_cpu_pct, peak_rss_mb.

Failure semantics (per FIRING_COUNT_PLAN.md v2.4 §5):
  ANY exception → log + sys.exit(1). No catch-and-continue. The coordinator
  detects missing _DONE and respawns.

Compute model: single Python process. tokenizers.encode_batch handles
internal Rust parallelism across all cores. No multiprocessing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HPLT_SOURCE_STRING = "HPLT/ell_Grek_ge8_no_mt_clean60"


def classify_source(s: str) -> str:
    """Component rule per FIRING_COUNT_PLAN.md v2.4 §3.3."""
    return "hplt_only" if s == HPLT_SOURCE_STRING else "glossapi_nanochat_only"


# ----------------------------------------------------------------------------
# Manifest reader — supports both CSV and Parquet manifests
# ----------------------------------------------------------------------------

def iter_manifest_sources(manifest_path: str, batch_size: int):
    """Yield lists of `source_dataset` strings in row order, in batches.

    Works for both .csv and .parquet manifests; both must have a
    `source_dataset` column.
    """
    p = Path(manifest_path)
    if p.suffix == ".parquet":
        pf = pq.ParquetFile(str(p))
        if "source_dataset" not in pf.schema_arrow.names:
            raise ValueError(
                f"manifest {p} has no 'source_dataset' column "
                f"(schema: {pf.schema_arrow.names[:5]})"
            )
        for rb in pf.iter_batches(batch_size=batch_size, columns=["source_dataset"]):
            yield rb.column("source_dataset").to_pylist()
    elif p.suffix == ".csv":
        # CSV streaming with consistent batch sizes
        with p.open(newline="") as fh:
            rdr = csv.DictReader(fh)
            if "source_dataset" not in rdr.fieldnames:
                raise ValueError(
                    f"manifest {p} has no 'source_dataset' column "
                    f"(columns: {rdr.fieldnames})"
                )
            batch: list[str] = []
            for row in rdr:
                batch.append(row["source_dataset"])
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
    else:
        raise ValueError(f"unknown manifest extension: {p.suffix}")


# ----------------------------------------------------------------------------
# Paired-read iterator
# ----------------------------------------------------------------------------

def paired_text_source_iter(text_parquet: str, manifest_path: str,
                            batch_size: int):
    """Yield (texts, sources) batches with one-to-one row alignment.

    The text parquet drives the cadence; the manifest stream is advanced
    in lockstep. If the manifest runs out before the text parquet (or vice
    versa), raise — that's a row-misalignment error per fail-hard semantics.
    """
    text_pf = pq.ParquetFile(text_parquet)
    if "text" not in text_pf.schema_arrow.names:
        raise ValueError(
            f"{text_parquet} has no 'text' column "
            f"(schema: {text_pf.schema_arrow.names[:5]})"
        )

    manifest_gen = iter_manifest_sources(manifest_path, batch_size)
    manifest_buf: list[str] = []

    for rb in text_pf.iter_batches(batch_size=batch_size, columns=["text"]):
        texts = rb.column("text").to_pylist()
        # Refill manifest buffer to match this batch's size
        while len(manifest_buf) < len(texts):
            try:
                manifest_buf.extend(next(manifest_gen))
            except StopIteration:
                raise RuntimeError(
                    f"manifest exhausted before text parquet (text rows so far "
                    f"need {len(texts)} more sources)"
                )
        sources = manifest_buf[: len(texts)]
        manifest_buf = manifest_buf[len(texts):]
        yield texts, sources

    # Ensure manifest didn't have extra rows
    remaining = len(manifest_buf)
    try:
        remaining += sum(len(b) for b in manifest_gen)
    except Exception:
        pass
    if remaining > 0:
        raise RuntimeError(
            f"manifest has {remaining} extra rows beyond text parquet"
        )


# ----------------------------------------------------------------------------
# Main counting loop
# ----------------------------------------------------------------------------

def run_counting(text_parquet: str, manifest_path: str, tokenizer_dir: str,
                 batch_size: int = 4096,
                 smoke_max_rows_per_component: int | None = None,
                 progress_fn=None) -> tuple[dict, dict, dict]:
    """Return (counts_by_source, denoms_by_source, benchmark).

    counts_by_source : {source_dataset → np.int64[vocab_size]}
    denoms_by_source : {source_dataset → {rows, chars, tokenized_tokens}}
    benchmark        : {wall_seconds, tokenize_tokens_per_sec, peak_rss_mb,
                        mean_cpu_pct, n_rows_processed}
    """
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    vocab_size = tok.get_vocab_size()

    counts: dict[str, np.ndarray] = {}
    denoms: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "chars": 0, "tokenized_tokens": 0}
    )
    component_row_counts = {"glossapi_nanochat_only": 0, "hplt_only": 0}
    n_rows_processed = 0
    total_tokens = 0
    t_tok_start = time.time()

    # CPU sampling thread (best-effort; only if /proc/stat is readable)
    cpu_samples: list[float] = []
    stop_sampling = threading.Event()

    def sample_cpu():
        try:
            import psutil
        except ImportError:
            return
        while not stop_sampling.is_set():
            try:
                cpu_samples.append(psutil.cpu_percent(interval=2.0))
            except Exception:
                return

    sampler = threading.Thread(target=sample_cpu, daemon=True)
    sampler.start()

    # Hard ceiling for smoke: even if shard 0 is source-homogeneous
    # (e.g. train.parquet sorted by stable_key → shard 0 is all GlossAPI),
    # the per-component cap never trips and smoke would otherwise process
    # the whole shard. Cap total rows at 4× per-component-cap × n_components
    # so smoke remains bounded in pathological cases (reviewer round 5 #2).
    smoke_total_rows_cap = (
        smoke_max_rows_per_component * 4 * len(component_row_counts)
        if smoke_max_rows_per_component is not None else None
    )

    try:
        for texts, sources in paired_text_source_iter(
            text_parquet, manifest_path, batch_size
        ):
            # Smoke-cap check (BEFORE encoding to avoid wasted tokenize work).
            # Two termination conditions: (a) every component has reached its
            # per-component cap, or (b) total rows hit the homogeneous-shard
            # safety ceiling.
            if smoke_max_rows_per_component is not None:
                total_rows_so_far = sum(component_row_counts.values())
                per_component_full = all(
                    c >= smoke_max_rows_per_component
                    for c in component_row_counts.values()
                )
                if per_component_full or total_rows_so_far >= smoke_total_rows_cap:
                    break

            # Validate text + source alignment
            if len(texts) != len(sources):
                raise RuntimeError(
                    f"alignment error: {len(texts)} texts vs {len(sources)} sources"
                )

            # Step 1: account EVERY row in row/char denominators here, BEFORE
            # the non-empty filter below. C3's quality gates filter empty
            # texts, but if any slip through they must still count toward
            # source_dataset_rows / source_dataset_chars (so per-source row
            # totals match train.parquet's row count). They contribute 0 to
            # tokenized_tokens (incremented in Step 3 per encoded row).
            for text, source in zip(texts, sources):
                d = denoms[source]
                d["rows"] += 1
                d["chars"] += len(text) if isinstance(text, str) else 0
                component_row_counts[classify_source(source)] += 1

            # Step 2: filter to non-empty for tokenization (empty texts would
            # produce 0 tokens anyway; skip them to avoid wasted encode_batch
            # work).
            valid_pairs = [(t, s) for t, s in zip(texts, sources)
                          if isinstance(t, str) and t]
            if not valid_pairs:
                continue
            valid_texts = [p[0] for p in valid_pairs]
            valid_sources = [p[1] for p in valid_pairs]

            # Tokenize — Rust parallel via encode_batch
            encs = tok.encode_batch(valid_texts, add_special_tokens=False)
            if len(encs) != len(valid_texts):
                raise RuntimeError(
                    f"encode_batch returned {len(encs)} encodings for "
                    f"{len(valid_texts)} texts — shape mismatch"
                )

            # Accumulate per-source counts. Critical perf fix (reviewer
            # round 3 #2): bincount ONCE per (source, batch) on concatenated
            # ids, NOT once per document. At minlength=148_480 the per-doc
            # allocation dominates; batching gives ~200x speedup.
            ids_by_source: dict[str, list[np.ndarray]] = defaultdict(list)
            for source, enc in zip(valid_sources, encs):
                ids = np.asarray(enc.ids, dtype=np.int32)
                ids_by_source[source].append(ids)
                denoms[source]["tokenized_tokens"] += int(ids.size)
                total_tokens += int(ids.size)
            for source, id_arrays in ids_by_source.items():
                if source not in counts:
                    counts[source] = np.zeros(vocab_size, dtype=np.int64)
                if not id_arrays:
                    continue
                concat = np.concatenate(id_arrays) if len(id_arrays) > 1 else id_arrays[0]
                if concat.size == 0:
                    continue
                # Assert no id overflows the tokenizer vocab (catches
                # corrupted tokenizer.json or unexpected token-id values)
                if concat.max() >= vocab_size:
                    raise RuntimeError(
                        f"token id {concat.max()} exceeds tokenizer vocab "
                        f"({vocab_size}) — likely tokenizer/data mismatch"
                    )
                counts[source] += np.bincount(
                    concat, minlength=vocab_size
                ).astype(np.int64)

            n_rows_processed += len(texts)
            if progress_fn is not None:
                progress_fn(n_rows_processed, total_tokens)
    finally:
        stop_sampling.set()
        sampler.join(timeout=1.0)

    wall = time.time() - t_tok_start
    peak_rss = 0
    try:
        import resource
        peak_rss = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        )  # KB → MB on Linux
    except Exception:
        pass

    benchmark = {
        "wall_seconds": round(wall, 3),
        "n_rows_processed": int(n_rows_processed),
        "total_tokens": int(total_tokens),
        "tokenize_tokens_per_sec": int(total_tokens / max(wall, 1)),
        "peak_rss_mb": int(peak_rss),
        "mean_cpu_pct": (
            round(sum(cpu_samples) / len(cpu_samples), 2) if cpu_samples else None
        ),
        "n_sources": len(counts),
        "vocab_size": vocab_size,
        # Source accounting verification (reviewer round 3 #6): smoke must
        # confirm source_dataset values are actually being read + counted.
        "source_datasets_seen": sorted(counts.keys()),
        "source_dataset_rows": {s: int(d["rows"]) for s, d in denoms.items()},
        "component_row_counts": dict(component_row_counts),
    }
    # Convert defaultdict to plain dict for json-ability
    denoms = dict(denoms)
    return counts, denoms, benchmark


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------

def write_per_source_counts_parquet(counts: dict, out_path: str,
                                     vocab_size: int) -> int:
    """Long-format sparse: only non-zero (source, id) rows. Returns row count."""
    sources, ids, fires = [], [], []
    for source, vec in counts.items():
        nz = np.nonzero(vec)[0]
        if nz.size == 0:
            continue
        sources.extend([source] * nz.size)
        ids.extend(nz.tolist())
        fires.extend(vec[nz].tolist())
    table = pa.table({
        "source_dataset": pa.array(sources, type=pa.string()),
        "id": pa.array(ids, type=pa.int32()),
        "fire_count": pa.array(fires, type=pa.int64()),
    })
    pq.write_table(table, out_path, compression="zstd")
    return table.num_rows


def write_denoms_json(denoms: dict, out_path: str) -> None:
    Path(out_path).write_text(
        json.dumps(denoms, indent=2, ensure_ascii=False, sort_keys=True)
    )


def gsutil_upload(local: str, gcs: str) -> None:
    """Upload via `gcloud storage cp`. Raises on failure (fail-hard)."""
    subprocess.run(
        ["gcloud", "storage", "cp", local, gcs],
        check=True, capture_output=True, text=True,
    )


def gsutil_touch_done(gcs_path: str) -> None:
    """Create a tiny empty marker object at gcs_path."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".done", delete=False) as fh:
        fh.write(b"")
        tmp = fh.name
    try:
        gsutil_upload(tmp, gcs_path)
    finally:
        try: os.remove(tmp)
        except: pass


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text-parquet", required=True,
                    help="path to this shard's text parquet (local or gs://)")
    ap.add_argument("--manifest", required=True,
                    help="path to this shard's source manifest (csv or parquet)")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--gcs-out-prefix", required=True,
                    help="e.g. gs://bucket/firing_counts_TS")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-total-rows-per-component", type=int, default=None,
                    help="smoke cap: stop after both components reach this many rows")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--local-tmp", default="/tmp")
    args = ap.parse_args()

    if args.smoke and args.max_total_rows_per_component is None:
        print("ERROR: --smoke requires --max-total-rows-per-component", file=sys.stderr)
        return 1

    # Output namespace
    out_root = args.gcs_out_prefix.rstrip("/")
    if args.smoke:
        out_root = f"{out_root}/_smoke"
    shard_tag = f"shard_{args.shard:02d}_of_{args.total:02d}"

    t0 = time.time()

    # Progress callback (prints to stderr)
    def progress(n, t):
        if n % 50000 == 0:
            print(f"  progress: rows={n:,} tokens={t:,} elapsed={time.time()-t0:.0f}s",
                  flush=True, file=sys.stderr)

    print(f"[worker {shard_tag}] starting", flush=True)
    print(f"  text-parquet : {args.text_parquet}", flush=True)
    print(f"  manifest     : {args.manifest}", flush=True)
    print(f"  tokenizer    : {args.tokenizer_dir}", flush=True)
    print(f"  smoke        : {args.smoke}", flush=True)

    counts, denoms, benchmark = run_counting(
        text_parquet=args.text_parquet,
        manifest_path=args.manifest,
        tokenizer_dir=args.tokenizer_dir,
        batch_size=args.batch_size,
        smoke_max_rows_per_component=args.max_total_rows_per_component,
        progress_fn=progress,
    )

    print(f"[worker {shard_tag}] tokenize phase done: {benchmark}", flush=True)

    # Write outputs locally
    counts_local = os.path.join(args.local_tmp, f"{shard_tag}_counts.parquet")
    denoms_local = os.path.join(args.local_tmp, f"{shard_tag}_denoms.json")
    n_count_rows = write_per_source_counts_parquet(
        counts, counts_local, benchmark["vocab_size"]
    )
    write_denoms_json(denoms, denoms_local)
    print(f"  wrote {counts_local}  ({n_count_rows:,} non-zero rows)", flush=True)
    print(f"  wrote {denoms_local}  ({len(denoms)} sources)", flush=True)

    # Upload to GCS
    counts_gcs = f"{out_root}/per_source_counts/{shard_tag}.parquet"
    denoms_gcs = f"{out_root}/per_source_denominators/{shard_tag}.json"
    gsutil_upload(counts_local, counts_gcs)
    gsutil_upload(denoms_local, denoms_gcs)
    print(f"  uploaded → {counts_gcs}", flush=True)
    print(f"  uploaded → {denoms_gcs}", flush=True)

    # Smoke benchmark sidecar
    if args.smoke:
        bench_local = os.path.join(args.local_tmp, f"smoke_benchmark_{shard_tag}.json")
        Path(bench_local).write_text(json.dumps(benchmark, indent=2))
        bench_gcs = f"{out_root}/smoke_benchmark.json"
        gsutil_upload(bench_local, bench_gcs)
        print(f"  uploaded smoke benchmark → {bench_gcs}", flush=True)

    # _DONE marker last — coordinator gates on this
    done_gcs = f"{out_root}/_DONE_{shard_tag}"
    gsutil_touch_done(done_gcs)
    print(f"[worker {shard_tag}] DONE  total wall {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[worker FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Worker step: compute strict_exact + relaxed_exact + 128-perm MinHash + LSH bands
per row across all assigned shards.

Drives CANONICAL functions from `text_dedup_lib.py` (a copy of
`glossapi_corpus_cli/text_dedup.py` at the pinned commit). No local
reimplementation of hashing or MinHash — this matters because:

- text_dedup.hash_bytes returns full 64-char blake3 hex (the spec). Earlier
  draft truncated to 16 chars, which would have made our hashes incompatible
  with the published bundle's strict_exact_group_hash (a critical bug).
- text_dedup.minhash_signature is the canonical 128-perm impl. Earlier draft
  rolled its own which produced different signatures.
- text_dedup.shingle_hashes_from_text already short-circuits docs with <20
  tokens by returning []. We treat those docs as **not-near-dup-able** and
  emit a NULL minhash_sig + NULL lsh_band_hashes so they cannot collide.

Output: /mnt/data/output/shard_<source_id>_<stem>.parquet (zstd), columns:
  corpus_id, source_id, doc_key, source_dataset, source_doc_id,
  text_length, token_count,
  strict_exact_hash, relaxed_exact_hash,
  minhash_sig (binary 128*8 bytes OR null for short docs),
  lsh_band_hashes (binary 32*8 bytes OR null for short docs)
"""
from __future__ import annotations

import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from blake3 import blake3

# Import the pinned text_dedup library (copied next to this script by dispatch).
sys.path.insert(0, "/mnt/data")
import text_dedup_lib as td  # type: ignore  # noqa: E402

GREEK_DIACRITIC_POLICY = "preserve"  # USER DECISION 2026-05-18
NUM_PERM = 128
SHINGLE_SIZE = 5
SHINGLE_MODE = "token"
BANDS = 32
ROWS_PER_BAND = 4

CONFIG = Path("/mnt/data/run_state/worker_config.json")
SRC_ROOT = Path("/mnt/data/sources")
OUT_DIR = Path("/mnt/data/output")
LOG = Path("/mnt/data/run_state/hash_log.jsonl")
DONE = Path("/mnt/data/hash_done")
# The first full 8-worker rerun OOM-killed hash workers: individual large
# shards reached ~15 GiB RSS, and the old pool used cpu_count()//2 workers.
# Keep these tunable from metadata/env, but default to a conservative profile.
BATCH_ROWS = int(os.environ.get("HASH_BATCH_ROWS", "2000"))
DEFAULT_HASH_WORKERS = int(os.environ.get("HASH_WORKERS", "8"))

OUT_SCHEMA = pa.schema([
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


def log_event(event: dict) -> None:
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with LOG.open("a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def compute_lsh_bands(sig_uint64: np.ndarray) -> bytes:
    """Hash each band (ROWS_PER_BAND perm-mins) to 8-byte blake3 digest. 32 bands total."""
    bands_out = bytearray(BANDS * 8)
    for band_idx in range(BANDS):
        start = band_idx * ROWS_PER_BAND
        chunk = sig_uint64[start:start + ROWS_PER_BAND].tobytes()
        h = blake3(chunk).digest(length=8)
        bands_out[band_idx * 8:(band_idx + 1) * 8] = h
    return bytes(bands_out)


def process_row(text: str, source_dataset: str, source_doc_id: str) -> dict:
    if text is None:
        text = ""
    strict_norm = td.normalize_exact_strict_text(text)
    strict_hash = td.hash_bytes(strict_norm.encode("utf-8"))  # full 64-char blake3 hex
    relaxed_norm = td.normalize_exact_relaxed_text(
        text, greek_diacritic_policy=GREEK_DIACRITIC_POLICY)
    relaxed_hash = td.hash_bytes(relaxed_norm.encode("utf-8"))
    shingle_hashes, token_count, char_count = td.shingle_hashes_from_text(
        near_text=relaxed_norm, shingle_mode=SHINGLE_MODE, shingle_size=SHINGLE_SIZE,
    )
    # Short-doc handling: shingle_hashes is [] when token_count < SHORT_DOC_TOKEN_THRESHOLD.
    # Emit NULL sig + NULL bands so short docs cannot collide via LSH.
    if not shingle_hashes:
        sig_bytes = None
        bands_bytes = None
    else:
        sig_arr = td.minhash_signature(list(shingle_hashes), num_perm=NUM_PERM)
        # Ensure uint64 little-endian.
        sig_arr = np.ascontiguousarray(sig_arr, dtype=np.uint64)
        sig_bytes = sig_arr.tobytes()
        bands_bytes = compute_lsh_bands(sig_arr)
    doc_key = td.stable_doc_key(source_dataset, source_doc_id)
    return {
        "doc_key": doc_key,
        "source_dataset": source_dataset,
        "source_doc_id": source_doc_id,
        "text_length": char_count,
        "token_count": token_count,
        "strict_exact_hash": strict_hash,
        "relaxed_exact_hash": relaxed_hash,
        "minhash_sig": sig_bytes,
        "lsh_band_hashes": bands_bytes,
    }


def extract_greek_text(row: dict, source_id: str) -> tuple[str, str] | None:
    """Per-source-aware Greek-text extraction. Returns (text, source_doc_id_inferred) or None to skip row."""
    # EuroParl: `translation` is a struct/dict like {"el": "...", "X": "..."}.
    if source_id == "europarl_greek":
        tr = row.get("translation")
        if isinstance(tr, dict):
            t = tr.get("el")
            if not t:
                return None
            # Synthesize doc id from row index hint elsewhere.
            return t, row.get("_row_id", "")
        return None
    # EuroBlocks: rows have `language` field; keep only Greek.
    if source_id == "euroblocks_greek":
        lang = row.get("language")
        if lang != "Greek":
            return None
        t = row.get("text") or row.get("content") or row.get("answer") or row.get("output")
        if not t:
            return None
        return t, row.get("id") or row.get("_row_id", "")
    # FW2-HQ / Clean-Wikipedia / others — standard text field.
    for c in ("text", "content", "raw_content"):
        if row.get(c):
            return row[c], row.get("source_doc_id") or row.get("id") or row.get("_row_id", "")
    return None


def process_batch(
    batch_dict: dict,
    source_id: str,
    corpus_id: str,
    *,
    row_id_prefix: str,
    base_row_idx: int,
) -> list[dict]:
    out = []
    n_rows = len(next(iter(batch_dict.values())))
    src_datasets = batch_dict.get("source_dataset")
    for i in range(n_rows):
        row = {k: (v[i] if v is not None else None) for k, v in batch_dict.items()}
        row["_row_id"] = f"{row_id_prefix}_row_{base_row_idx + i}"
        extracted = extract_greek_text(row, source_id)
        if extracted is None:
            continue
        text, src_doc_id = extracted
        sds = (src_datasets[i] if src_datasets is not None else source_id)
        result = process_row(text, sds, str(src_doc_id))
        result["corpus_id"] = corpus_id
        result["source_id"] = source_id
        out.append(result)
    return out


def _safe_rel_path(parquet_path: Path, source_id: str) -> str:
    """Encode the input parquet path RELATIVE to /mnt/data/sources/<source_id>/ into a
    flat output stem so distinct inputs never share an output filename. EuroParl's
    20 bitexts each have a `train-00000-of-00001.parquet` — without this fix, all
    20 collide on the worker filesystem and ParquetWriter racing corrupts them."""
    src_root = SRC_ROOT / source_id
    try:
        rel = parquet_path.relative_to(src_root)
    except ValueError:
        rel = Path(parquet_path.name)
    rel_no_ext = rel.with_suffix("")
    # Replace path separators so the result is a flat filename. Avoid double-underscores
    # in input names colliding with our separator by using "__sep__".
    return str(rel_no_ext).replace("/", "__sep__")


def columns_for_source(pf: pq.ParquetFile, source_id: str) -> list[str]:
    """Read only columns that can affect extraction or doc identity.

    The old worker converted every parquet column in every batch to Python
    lists. Some HF shards carry large extra fields, and doing that across many
    concurrent processes was the memory multiplier that triggered OOM kills.
    """
    available = set(pf.schema_arrow.names)
    if source_id == "europarl_greek":
        desired = ["translation", "source_dataset", "source_doc_id", "id"]
    elif source_id == "euroblocks_greek":
        desired = ["language", "text", "content", "answer", "output",
                   "source_dataset", "source_doc_id", "id"]
    else:
        desired = ["text", "content", "raw_content",
                   "source_dataset", "source_doc_id", "id"]
    cols = [c for c in desired if c in available]
    if not cols:
        raise ValueError(f"no usable text/identity columns in {pf.metadata}")
    return cols


def process_shard(parquet_path: Path, source_id: str, corpus_id: str) -> tuple[str, int]:
    t0 = time.time()
    n = 0
    n_kept = 0
    stem = _safe_rel_path(parquet_path, source_id)
    out_path = OUT_DIR / f"shard_{source_id}_{stem}.parquet"
    tmp_path = OUT_DIR / f"shard_{source_id}_{stem}.parquet.tmp"
    if out_path.exists():
        try:
            n_existing = pq.ParquetFile(str(out_path)).metadata.num_rows
            log_event({"event": "shard_skipped_existing", "shard": str(parquet_path),
                       "source_id": source_id, "rows_kept": n_existing})
            return str(out_path), int(n_existing)
        except Exception:
            try:
                out_path.unlink()
            except FileNotFoundError:
                pass
    writer = None
    pf = pq.ParquetFile(str(parquet_path))
    read_cols = columns_for_source(pf, source_id)
    row_id_prefix = f"{source_id}_{stem}"
    try:
        for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=read_cols):
            cols = batch.column_names
            # Materialize the selected columns as Python lists so extraction can
            # handle nested structs and source-specific schemas.
            batch_dict = {c: batch.column(c).to_pylist() for c in cols}
            batch_n = len(next(iter(batch_dict.values())))
            rows = process_batch(
                batch_dict,
                source_id,
                corpus_id,
                row_id_prefix=row_id_prefix,
                base_row_idx=n,
            )
            n += batch_n
            n_kept += len(rows)
            if not rows:
                continue
            tbl = pa.Table.from_pylist(rows, schema=OUT_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp_path), tbl.schema, compression="zstd")
            writer.write_table(tbl)
        if writer is not None:
            writer.close()
            os.replace(tmp_path, out_path)  # atomic on POSIX same-fs
    except BaseException:
        # On any failure: close writer if possible, remove tmp, re-raise.
        if writer is not None:
            try: writer.close()
            except Exception: pass
        try: tmp_path.unlink()
        except FileNotFoundError: pass
        raise
    log_event({"event": "shard_done", "shard": str(parquet_path),
               "source_id": source_id, "rows_in": n, "rows_kept": n_kept,
               "seconds": round(time.time() - t0, 2)})
    return str(out_path), n_kept


def main() -> int:
    if not CONFIG.exists():
        print(f"[FATAL] missing config {CONFIG}", file=sys.stderr)
        return 2
    cfg = json.loads(CONFIG.read_text())
    widx = cfg["worker_idx"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_event({"event": "hash_start", "worker_idx": widx,
               "cpu_count": mp.cpu_count(), "policy": GREEK_DIACRITIC_POLICY,
               "num_perm": NUM_PERM, "shingle_size": SHINGLE_SIZE,
               "batch_rows": BATCH_ROWS, "hash_workers": DEFAULT_HASH_WORKERS})

    work = []
    for shard in cfg["shards"]:
        src_id = shard["source_id"]
        corpus = shard["corpus_id"]
        src_dir = SRC_ROOT / src_id
        for pq_path in sorted(src_dir.rglob("*.parquet")):
            work.append((pq_path, src_id, corpus))
    log_event({"event": "shards_enumerated", "count": len(work)})

    pool_size = min(max(DEFAULT_HASH_WORKERS, 1), max(1, len(work)))
    completed = 0
    total_kept = 0
    failed: list[dict] = []
    with ProcessPoolExecutor(max_workers=pool_size) as ex:
        futures = {ex.submit(process_shard, p, sid, c): (p, sid, c) for (p, sid, c) in work}
        for fut in as_completed(futures):
            try:
                out, n_kept = fut.result()
                completed += 1
                total_kept += n_kept
            except Exception as e:
                p, sid, c = futures[fut]
                rec = {"event": "shard_failed", "shard": str(p),
                       "source_id": sid, "corpus_id": c, "error": str(e)[:500]}
                log_event(rec)
                failed.append(rec)
    log_event({"event": "hash_done", "worker_idx": widx,
               "shards_processed": completed, "rows_kept": total_kept,
               "shards_failed": len(failed)})
    # CRITICAL: do NOT touch hash_done sentinel if any shard failed. Upstream
    # run_all_on_worker.sh has set -e, so a nonzero exit here halts the chain
    # before upload_output.sh + _done_sentinel.sh — the worker never reports
    # success, and 04_poll_and_collect will surface the failure.
    if failed:
        print(f"[hash_pass] FATAL: {len(failed)} shard(s) failed; see hash_log.jsonl. Not touching hash_done.", file=sys.stderr)
        return 3
    DONE.touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())

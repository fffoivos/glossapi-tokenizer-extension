#!/usr/bin/env python3
"""Coordinator step 07: MinHash near-dup overlap via LSH bands — STREAMING.

For each Apertus source A and each HF-pool source C:
1. For each of 32 bands, vectorize-extract just THAT band's hashes for both
   sides (no full 32x explode in memory at once).
2. Inner-join on band_hash → candidate (a_doc, c_doc) pairs.
3. Accumulate candidates into a set (dedup across bands).
4. After all bands processed, compute estimated Jaccard from full MinHash
   sigs in numpy and filter to >= 0.85.

Compared to the 2026-05-18 version that exploded full sources at once:
- Peak memory drops by ~32x (one band at a time, not 32).
- A full clean run will have ~10M apertus rows + ~10M HF-pool rows; the
  one-shot explode = 320M rows × ~50 bytes/row ≈ 16 GB just for the band
  table — risky on a c4-highcpu-32 (64 GB RAM). One band at a time = 500 MB.

Output: artifacts/<RUN_ID>/overlap/near/<A>_x_<C>.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

SUB = Path(__file__).resolve().parents[2]
RUN_ID = (SUB / "manifests/CURRENT_RUN_ID").read_text().strip()
SRC = SUB / "artifacts" / RUN_ID / "sources"
OUT = SUB / "artifacts" / RUN_ID / "overlap" / "near"
OUT.mkdir(parents=True, exist_ok=True)

NUM_PERM = 128
BANDS = 32
THRESHOLD = 0.85


def extract_band(df: pl.DataFrame, band_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (doc_keys_array, band_hashes_uint64) for ONE band only.

    `lsh_band_hashes` is a 256-byte binary column (32 bands × 8 bytes).
    Slice bytes [band_idx*8 : (band_idx+1)*8] from each row.
    """
    df = df.filter(pl.col("lsh_band_hashes").is_not_null())
    n = df.height
    if n == 0:
        return np.array([], dtype=object), np.array([], dtype=np.uint64)
    raw = np.frombuffer(b"".join(df["lsh_band_hashes"].to_list()),
                         dtype=np.uint8).reshape(n, BANDS * 8)
    band_bytes = raw[:, band_idx * 8:(band_idx + 1) * 8]
    # View 8 bytes as one uint64 per row.
    band_u64 = band_bytes.copy().view(np.uint64).reshape(-1)
    return df["doc_key"].to_numpy(), band_u64


def join_pair_near(a_path: Path, c_path: Path, out_path: Path) -> int:
    a_df = pl.read_parquet(a_path).select([
        "doc_key", "source_dataset", "source_doc_id", "lsh_band_hashes", "minhash_sig"
    ])
    c_df = pl.read_parquet(c_path).select([
        "doc_key", "source_dataset", "source_doc_id", "lsh_band_hashes", "minhash_sig"
    ])
    print(f"[near]   a_rows={a_df.height:,}  c_rows={c_df.height:,}")

    candidate_pairs: set[tuple[str, str]] = set()
    for bi in range(BANDS):
        a_keys, a_hashes = extract_band(a_df, bi)
        c_keys, c_hashes = extract_band(c_df, bi)
        if a_hashes.size == 0 or c_hashes.size == 0:
            continue
        # Polars inner-join on the single-band hash.
        band_df_a = pl.DataFrame({"a_doc_key": a_keys, "band_hash": a_hashes})
        band_df_c = pl.DataFrame({"c_doc_key": c_keys, "band_hash": c_hashes})
        joined = band_df_a.join(band_df_c, on="band_hash", how="inner")
        if joined.is_empty():
            continue
        for ak, ck in zip(joined["a_doc_key"].to_list(), joined["c_doc_key"].to_list()):
            candidate_pairs.add((ak, ck))
        # Free per-band arrays.
        del a_keys, a_hashes, c_keys, c_hashes, band_df_a, band_df_c, joined

    print(f"[near]   total unique candidate pairs across {BANDS} bands: {len(candidate_pairs):,}")
    if not candidate_pairs:
        return 0

    # Map doc_key -> {sig, source_dataset, source_doc_id}.
    a_meta = {dk: (sd, sdi, sig) for dk, sd, sdi, sig in zip(
        a_df["doc_key"].to_list(), a_df["source_dataset"].to_list(),
        a_df["source_doc_id"].to_list(), a_df["minhash_sig"].to_list()) if sig is not None}
    c_meta = {dk: (sd, sdi, sig) for dk, sd, sdi, sig in zip(
        c_df["doc_key"].to_list(), c_df["source_dataset"].to_list(),
        c_df["source_doc_id"].to_list(), c_df["minhash_sig"].to_list()) if sig is not None}

    pair_list = [p for p in candidate_pairs if p[0] in a_meta and p[1] in c_meta]
    if not pair_list:
        return 0
    a_sigs = np.array([np.frombuffer(a_meta[p[0]][2], dtype=np.uint64) for p in pair_list])
    c_sigs = np.array([np.frombuffer(c_meta[p[1]][2], dtype=np.uint64) for p in pair_list])
    jac = (a_sigs == c_sigs).sum(axis=1).astype(np.float64) / NUM_PERM
    keep = jac >= THRESHOLD
    if not keep.any():
        return 0
    rows = []
    for i, (ak, ck) in enumerate(pair_list):
        if not keep[i]:
            continue
        a_sd, a_sdi, _ = a_meta[ak]
        c_sd, c_sdi, _ = c_meta[ck]
        rows.append({
            "a_doc_key": ak, "a_source_dataset": a_sd, "a_source_doc_id": a_sdi,
            "c_doc_key": ck, "c_source_dataset": c_sd, "c_source_doc_id": c_sdi,
            "estimated_jaccard": float(jac[i]),
        })
    pl.DataFrame(rows).write_parquet(out_path, compression="zstd")
    return len(rows)


def main() -> int:
    a_files = sorted(SRC.glob("apertus_*.parquet"))
    c_files = sorted(SRC.glob("hf_source_pool_*.parquet"))
    if not a_files or not c_files:
        print(f"[near] missing sources under {SRC}", file=sys.stderr)
        return 2
    rows = []
    for ap in a_files:
        for cp in c_files:
            pair = f"{ap.stem.replace('apertus_', '')}_x_{cp.stem.replace('hf_source_pool_', '')}"
            op = OUT / f"{pair}.parquet"
            print(f"[near] {pair}")
            n = join_pair_near(ap, cp, op)
            rows.append({"pair": pair, "near_matches": n})
            print(f"[near] {pair}  matches={n}")
    summary = pl.DataFrame(rows)
    summary.write_parquet(OUT.parent.parent / "near_overlap_summary.parquet")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Tokenize the Phase B v4 Greek slices with the Apertus tokenizer and
count per-token-id occurrences.

Output:
  arrays/token_freq_<slice>.npy       (131072,) int64 per slice
  arrays/token_freq_total.npy         (131072,) int64 sum across slices
  arrays/token_freq_meta.json

Used by:
  - build_loo_target_ids.py  (filter Greek tokens with count >= threshold)
  - phase0_freq_stratification.py  (§2.7.1)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MODEL_ID = "swiss-ai/Apertus-8B-2509"

DEFAULT_SLICES = [
    "/home/foivos/runs/apertus_greek_phase_b_v4_20260512/hplt_el.parquet",
    "/home/foivos/runs/apertus_greek_phase_b_v4_20260512/glossapi_el_modern.parquet",
]

OUT_DIR = Path("/home/foivos/runs/apertus_embedding_init_test_20260512/arrays")


def count_slice(parquet_path: Path, tok, vocab_size: int) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=np.int64)
    table = pq.read_table(str(parquet_path))
    texts = table.column("text").to_pylist()
    t0 = time.time()
    last_log = t0
    n = len(texts)
    for i, text in enumerate(texts):
        if not text:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        # Use bincount on a length-V vector for speed.
        if ids:
            arr = np.asarray(ids, dtype=np.int64)
            np.add.at(counts, arr, 1)
        if time.time() - last_log >= 30:
            print(f"  [tok] {parquet_path.name}: {i+1}/{n} docs "
                  f"({time.time()-t0:.0f}s)", flush=True)
            last_log = time.time()
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else 131072
    # Apertus reports 131072 via the model config, not the tokenizer attr.
    vocab_size = max(vocab_size, 131072)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = np.zeros(vocab_size, dtype=np.int64)
    per_slice = {}
    t_all = time.time()

    for s in args.slices:
        p = Path(s)
        if not p.exists():
            print(f"[skip] missing {p}", flush=True)
            continue
        t0 = time.time()
        print(f"[tok] {p.name}", flush=True)
        counts = count_slice(p, tok, vocab_size)
        np.save(OUT_DIR / f"token_freq_{p.stem}.npy", counts)
        total += counts
        per_slice[p.stem] = {
            "path": str(p),
            "total_tokens": int(counts.sum()),
            "unique_tokens": int((counts > 0).sum()),
            "wall_seconds": int(time.time() - t0),
        }
        print(f"[tok] {p.stem}: {counts.sum():,} tokens, "
              f"{(counts > 0).sum()} unique  ({time.time()-t0:.0f}s)", flush=True)

    np.save(OUT_DIR / "token_freq_total.npy", total)
    meta = {
        "vocab_size": vocab_size,
        "slices": per_slice,
        "total_tokens": int(total.sum()),
        "unique_tokens_overall": int((total > 0).sum()),
        "wall_seconds": int(time.time() - t_all),
    }
    (OUT_DIR / "token_freq_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {meta['total_tokens']:,} tokens across "
          f"{meta['unique_tokens_overall']} unique ids", flush=True)


if __name__ == "__main__":
    main()

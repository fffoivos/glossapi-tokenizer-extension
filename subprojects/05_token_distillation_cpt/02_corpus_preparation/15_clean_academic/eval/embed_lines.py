#!/usr/bin/env python3
"""Frozen line embeddings for the DL tier. Embeds every present line of every doc with a frozen
multilingual encoder (default multilingual-e5-small, 384-d — fast on CPU, native Greek+Latin+polytonic),
in the SAME (doc, line) order as line_lr.build_matrix, so the embedding matrix aligns row-for-row with
the cached feature matrix. Cached to units/SPAN_emb/<encoder>.npz (gitignored, regenerable). Frozen
encoder — only the downstream head trains.

Run: python embed_lines.py [encoder]   (default intfloat/multilingual-e5-small)
"""
import os, sys, json
import numpy as np
import span_seq_data as D
HERE = os.path.dirname(os.path.abspath(__file__))
EMB_DIR = f"{HERE}/units/SPAN_emb"


def cache_path(enc):
    return f"{EMB_DIR}/{enc.split('/')[-1]}.npz"


def load_cached(enc):
    z = np.load(cache_path(enc))
    return z["emb"], list(z["docids"]), z["counts"]


def main():
    enc = sys.argv[1] if len(sys.argv) > 1 else "intfloat/multilingual-e5-small"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # >0 = timing probe on first N lines
    os.makedirs(EMB_DIR, exist_ok=True)
    data = D.load()
    # collect line texts in the canonical (doc, line) order; CHAR-TRUNCATE giant OCR/table lines
    texts, docids, counts = [], [], []
    for doc_id, d in data.docs.items():
        docids.append(doc_id); counts.append(len(d["lines"]))
        for _, t in d["lines"]:
            texts.append("passage: " + t[:300])   # e5 prefix; bib lines are short, cap runaway lines
    import time
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(8)
    model = SentenceTransformer(enc, device="cpu")
    model.max_seq_length = 96                       # KEY: cap tokens so long lines don't blow up batches
    if limit:
        t0 = time.time(); model.encode(texts[:limit], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        r = limit / (time.time() - t0)
        print(f"PROBE: {r:.0f} lines/sec -> {len(texts)} lines ~ {len(texts)/r/60:.1f} min")
        return
    print(f"embedding {len(texts):,} lines with {enc} (max_seq=96) ...")
    emb = model.encode(texts, batch_size=128, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.savez(cache_path(enc), emb=emb, docids=np.array(docids, dtype="U64"),
             counts=np.array(counts, np.int32))
    print(f"wrote {cache_path(enc)}  shape {emb.shape}")


if __name__ == "__main__":
    main()

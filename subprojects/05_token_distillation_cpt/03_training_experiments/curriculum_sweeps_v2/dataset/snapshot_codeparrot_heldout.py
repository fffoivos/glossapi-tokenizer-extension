#!/usr/bin/env python3
"""Snapshot a held-out codeparrot slice with a STABLE synthetic id.

Resolves the "codeparrot has no stable doc id" blocker enough for this sweep:
codeparrot-clean is HF-streamed (text_column='content', no local parquet, no
id), so id-drop is not the real guard. Here we stream a deterministic slice well
past the expected training draw, mint id = sha1(content)[:16] for provenance,
write val_forget_code.jsonl, and fold the ids into forget_holdout_ids.parquet
(col `doc_id`). The no-overlap guarantee comes from the disjoint offset, not
from source id exclusion.

Deterministic: takes a contiguous window [OFFSET, OFFSET+budget) of the stream so it is reproducible.
This v2 run includes code forgetting, so run this before build_forgetting_vals.py.
"""
import argparse, hashlib, json, os
import pyarrow.parquet as pq, pyarrow as pa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--char-budget", type=int, default=2_000_000_000)  # ~0.5B tok
    ap.add_argument("--skip-docs", type=int, default=2_000_000,
                    help="skip this many docs first so the held-out window is disjoint from the training draw")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)
    out_jsonl = os.path.join(a.output_dir, "val_forget_code.jsonl")
    chars = kept = 0
    ids = []
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for i, row in enumerate(ds):
            if i < a.skip_docs:
                continue
            t = row.get("content") or row.get("text")
            if not t:
                continue
            doc_id = hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]
            fout.write(json.dumps({"text": t, "doc_id": doc_id, "source_dataset": "codeparrot"}, ensure_ascii=False) + "\n")
            chars += len(t); kept += 1; ids.append(doc_id)
            if chars >= a.char_budget:
                break
    print(f"  code: {kept:,} docs, {chars/1e9:.2f}B chars -> {out_jsonl}", flush=True)

    # merge code ids into forget_holdout_ids.parquet (build_forgetting_vals.py merges any pre-existing)
    out = os.path.join(a.output_dir, "forget_holdout_ids.parquet")
    if os.path.exists(out):
        ids = [str(x) for x in pq.read_table(out).column("doc_id").to_pylist()] + ids
    pq.write_table(pa.table({"doc_id": ids}), out, compression="zstd")
    print(f"DONE code snapshot; forget ids now {len(ids):,} -> {out}", flush=True)

if __name__ == "__main__":
    main()

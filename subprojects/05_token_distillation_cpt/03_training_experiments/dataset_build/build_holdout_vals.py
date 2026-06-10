#!/usr/bin/env python3
"""Build 3 held-out validation sets (~0.5B tokens each) from SELECTED:
  hplt          source_dataset ~ ^HPLT/ell_Grek
  openarchives  source_dataset ~ ^openarchives\\.gr
  greek_phd     source_dataset ~ ^greek_phd
Writes val_<name>.jsonl per set, plus val_holdout_ids.parquet (col `doc_id` =
the held-out source_doc_ids for hplt+openarchives) so the train mix can exclude
them via drop_doc_keys_parquet. greek_phd is not in the train mix, so its ids
need not be excluded (still recorded for provenance).

Deterministic: takes the FIRST docs (by row order) per source up to the char
budget. A val set is only used for --eval-iters batches, so ~0.5B is generous.
"""
import argparse, json, os, re
import pyarrow as pa
import pyarrow.parquet as pq

SETS = {
    "hplt":         r"^HPLT/ell_Grek",
    "openarchives": r"^openarchives\.gr",
    "greek_phd":    r"^greek_phd",
}
EXCLUDE_FROM_TRAIN = {"hplt", "openarchives"}  # these overlap the train mix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--char-budget", type=int, default=2_000_000_000,  # ~0.5B tok
                    help="chars per val set (≈ chars/4 tokens)")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    pats = {k: re.compile(v) for k, v in SETS.items()}
    fout = {k: open(os.path.join(a.output_dir, f"val_{k}.jsonl"), "w", encoding="utf-8") for k in SETS}
    chars = {k: 0 for k in SETS}
    kept = {k: 0 for k in SETS}
    done = set()
    hold_ids = []  # (doc_id) for EXCLUDE_FROM_TRAIN sets

    f = pq.ParquetFile(a.selected)
    for b in f.iter_batches(columns=["source_dataset", "source_doc_id", "text"], batch_size=20_000):
        if len(done) == len(SETS):
            break
        d = b.to_pydict()
        for sd, sdi, t in zip(d["source_dataset"], d["source_doc_id"], d["text"]):
            if not sd or not t:
                continue
            for k, pat in pats.items():
                if k in done:
                    continue
                if pat.match(sd):
                    fout[k].write(json.dumps({"text": t, "source_doc_id": sdi,
                                              "source_dataset": sd}, ensure_ascii=False) + "\n")
                    chars[k] += len(t); kept[k] += 1
                    if k in EXCLUDE_FROM_TRAIN:
                        hold_ids.append(sdi)
                    if chars[k] >= a.char_budget:
                        done.add(k); fout[k].close()
                        print(f"  {k}: budget reached — {kept[k]:,} docs, {chars[k]/1e9:.2f}B chars", flush=True)
                    break
    for k in SETS:
        if k not in done:
            fout[k].close()
            print(f"  {k}: source exhausted — {kept[k]:,} docs, {chars[k]/1e9:.2f}B chars", flush=True)

    out = os.path.join(a.output_dir, "val_holdout_ids.parquet")
    pq.write_table(pa.table({"doc_id": hold_ids}), out, compression="zstd")
    print(f"DONE: val sets in {a.output_dir}; holdout ids ({len(hold_ids):,}) -> {out}", flush=True)
    for k in SETS:
        print(f"  val_{k}.jsonl: {kept[k]:,} docs, ~{chars[k]/4/1e9:.2f}B tok est")

if __name__ == "__main__":
    main()

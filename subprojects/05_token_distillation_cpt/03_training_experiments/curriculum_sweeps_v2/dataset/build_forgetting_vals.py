#!/usr/bin/env python3
"""Build the OLD-data "forgetting" held-out validation sets (~0.5B tok each) from the
Apertus-original/replay pools — the contamination-free forgetting signal.

Mirrors build_holdout_vals.py (deterministic first-N-by-row-order to a char budget) but each
set comes from its OWN source parquet(s), with its own id column. Writes val_forget_<name>.jsonl
per set + a single forget_holdout_ids.parquet (column `doc_id`) holding every held-out id so the
replay_only recipe can exclude them via drop_doc_keys_parquet.

The stored id VALUE per source must match the doc_key_field wired in make_phase_recipes.py:
  FineWeb pools (english/de/ru/zh) -> `id`
  greek_replay                     -> `source_doc_id`
  code                             -> handled separately by snapshot_codeparrot_heldout.py (`doc_id`)

UNSEEN GUARANTEE: each set is carved first-N-by-row-order from the SAME parquet shard the replay_only
TRAINING recipe reads (so the held-out docs are genuinely in the training pool), and its `id`s are
dropped from training via drop_doc_keys_parquet. english additionally applies the training source's
score>=3 filter (bulk_13b.json replay_t1_english_edu) so the held-out sub-population matches what is
trained; de/ru/zh fw2hq shards are pre-filtered HQ (no score column to match). code is the lone
exception — it has no stable id, so it is held out by a disjoint OFFSET instead (snapshot_codeparrot
--skip-docs), not by the id-drop.

ROBUST TO MISSING POOLS: a source whose parquet(s) don't exist is skipped with a warning; the rest
still build (idempotent: re-running rewrites val_forget_<name>.jsonl + merges ids).
"""
import argparse, glob, json, os
import pyarrow.parquet as pq, pyarrow as pa

SC = "/iopsstor/scratch/cscs/fffoivos"
# name -> (parquet glob, id_column, text_column, min_score)  -- glob points at the SAME shard the
# replay_only training recipe reads (bulk_13b.json local_parquet), so the id-drop is a real forgetting signal.
SOURCES = {
    "english":   (f"{SC}/cpt_corpus/replay/eng_fineweb_edu/sample/10BT/*.parquet",   "id",            "text", 3.0),
    "de":        (f"{SC}/cpt_corpus/replay/deu_Latn_fw2hq/deu_Latn/*.parquet",       "id",            "text", None),
    "ru":        (f"{SC}/cpt_corpus/replay/rus_Cyrl_fw2hq/rus_Cyrl/*.parquet",       "id",            "text", None),
    "zh":        (f"{SC}/cpt_corpus/replay/cmn_Hani_fw2hq/cmn_Hani/*.parquet",       "id",            "text", None),
    "old_greek": (f"{SC}/cpt_corpus/greek_replay/greek_replay.parquet",              "source_doc_id", "text", None),
}
# code is built by snapshot_codeparrot_heldout.py (HF stream, no stable id) and merged into forget_holdout_ids.parquet.

def build_one(name, pattern, id_col, text_col, min_score, out_dir, char_budget):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  [SKIP] {name}: no parquet at {pattern} (download the pool, then re-run)", flush=True)
        return []
    out_jsonl = os.path.join(out_dir, f"val_forget_{name}.jsonl")
    chars = kept = 0
    ids = []
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for fp in files:
            if chars >= char_budget:
                break
            pf = pq.ParquetFile(fp)
            cols = set(pf.schema_arrow.names)
            ic = id_col if id_col in cols else ("source_doc_id" if "source_doc_id" in cols else ("id" if "id" in cols else None))
            sc = "score" if (min_score is not None and "score" in cols) else None
            for b in pf.iter_batches(columns=[c for c in (ic, text_col, sc) if c], batch_size=20_000):
                if chars >= char_budget:
                    break
                d = b.to_pydict()
                texts = d.get(text_col) or d.get("content") or []
                idv = d.get(ic) if ic else [None] * len(texts)
                scores = d.get(sc) if sc else [None] * len(texts)
                for t, i, s in zip(texts, idv, scores):
                    if not t:
                        continue
                    if min_score is not None and s is not None and s < min_score:
                        continue   # match the training source's score>=min filter (english)
                    doc_id = str(i) if i is not None else f"{name}_{kept}"
                    fout.write(json.dumps({"text": t, "doc_id": doc_id, "source_dataset": name}, ensure_ascii=False) + "\n")
                    chars += len(t); kept += 1
                    ids.append(doc_id)
                    if chars >= char_budget:
                        break
    print(f"  {name}: {kept:,} docs, {chars/1e9:.2f}B chars (~{chars/4/1e9:.2f}B tok est) -> {out_jsonl}", flush=True)
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--char-budget", type=int, default=2_000_000_000)  # ~0.5B tok
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    all_ids = []
    # merge with any pre-existing code ids (snapshot_codeparrot writes them first)
    existing = os.path.join(a.output_dir, "forget_holdout_ids.parquet")
    if os.path.exists(existing):
        all_ids.extend(str(x) for x in pq.read_table(existing).column("doc_id").to_pylist())
        print(f"  merging {len(all_ids):,} pre-existing ids (e.g. code snapshot)", flush=True)
    for name, (pattern, id_col, text_col, min_score) in SOURCES.items():
        all_ids.extend(build_one(name, pattern, id_col, text_col, min_score, a.output_dir, a.char_budget))

    out = os.path.join(a.output_dir, "forget_holdout_ids.parquet")
    pq.write_table(pa.table({"doc_id": all_ids}), out, compression="zstd")
    print(f"DONE: forgetting val sets in {a.output_dir}; {len(all_ids):,} held-out ids -> {out}", flush=True)

if __name__ == "__main__":
    main()

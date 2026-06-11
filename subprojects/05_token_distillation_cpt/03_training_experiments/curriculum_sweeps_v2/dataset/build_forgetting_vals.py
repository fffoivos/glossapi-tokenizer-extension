#!/usr/bin/env python3
"""Build OLD-data validation sets from Apertus pretraining source families.

Mirrors build_holdout_vals.py (deterministic first-N-by-row-order to a char budget) but each
set comes from its OWN source parquet(s), with its own id column. Writes val_forget_<name>.jsonl
per set + a single forget_holdout_ids.parquet (column `doc_id`) holding every held-out id so the
replay_only recipe can exclude them via drop_doc_keys_parquet.

The stored id VALUE per source must match the doc_key_field wired in make_phase_recipes.py:
  FineWeb pools (english/de/ru/zh) -> `id`
  greek_replay                     -> `source_doc_id`
  code                             -> StarCoderData staged subset (`doc_id`)

UNSEEN GUARANTEE FOR THIS CPT RUN: each row-id-backed set is carved from the SAME local replay
pool the replay_only recipe trains on, and its ids are dropped via drop_doc_keys_parquet. That is
separate from Apertus provenance: english/de/ru/zh are Apertus pretraining source families, but this
builder does not have an item-level manifest proving a particular document was consumed by Apertus.
old_greek is stricter: it comes from the nanochat ∩ Apertus-overlap overlay. code is a staged
StarCoderData subset, matching an Apertus pretraining source family.

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
    "code":      (f"{SC}/cpt_corpus/replay/starcoderdata_v2/*/*.parquet",            "doc_id",        "content", None),
    "old_greek": (f"{SC}/cpt_corpus/greek_replay/greek_replay.parquet",              "source_doc_id", "text", None),
}

PROVENANCE = {
    "english": {
        "apertus_source_status": "apertus_pretraining_source_family",
        "apertus_source": "FineWeb-Edu Score-3/Score-2 family; v2 uses Score>=3 to match the replay source.",
        "strict_item_level_seen_by_apertus": False,
    },
    "de": {
        "apertus_source_status": "apertus_pretraining_source_family",
        "apertus_source": "FineWeb2-HQ high-resource language deu_Latn.",
        "strict_item_level_seen_by_apertus": False,
    },
    "ru": {
        "apertus_source_status": "apertus_pretraining_source_family",
        "apertus_source": "FineWeb2-HQ high-resource language rus_Cyrl.",
        "strict_item_level_seen_by_apertus": False,
    },
    "zh": {
        "apertus_source_status": "apertus_pretraining_source_family",
        "apertus_source": "FineWeb2-HQ high-resource language cmn_Hani.",
        "strict_item_level_seen_by_apertus": False,
    },
    "old_greek": {
        "apertus_source_status": "apertus_overlap_overlay",
        "apertus_source": "Apertus-original Greek replay: nanochat rows intersecting the Apertus-overlap drop overlay.",
        "strict_item_level_seen_by_apertus": True,
    },
    "code": {
        "apertus_source_status": "apertus_pretraining_source_family",
        "apertus_source": "StarCoderData subset staged from bigcode/starcoderdata.",
        "strict_item_level_seen_by_apertus": False,
    },
}

def iter_rows(files, id_col, text_col, min_score):
    for fp in files:
        pf = pq.ParquetFile(fp)
        cols = set(pf.schema_arrow.names)
        ic = id_col if id_col in cols else ("source_doc_id" if "source_doc_id" in cols else ("id" if "id" in cols else None))
        sc = "score" if (min_score is not None and "score" in cols) else None
        for b in pf.iter_batches(columns=[c for c in (ic, text_col, sc) if c], batch_size=20_000):
            d = b.to_pydict()
            texts = d.get(text_col) or d.get("content") or []
            idv = d.get(ic) if ic else [None] * len(texts)
            scores = d.get(sc) if sc else [None] * len(texts)
            for t, i, s in zip(texts, idv, scores):
                if not t:
                    continue
                if min_score is not None and s is not None and s < min_score:
                    continue   # match the training source's score>=min filter (english)
                yield t, i

def measure_pool(files, id_col, text_col, min_score):
    docs = chars = 0
    for t, _ in iter_rows(files, id_col, text_col, min_score):
        docs += 1
        chars += len(t)
    return docs, chars

def build_one(name, pattern, id_col, text_col, min_score, out_dir, char_budget, max_val_fraction):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  [SKIP] {name}: no parquet at {pattern} (download the pool, then re-run)", flush=True)
        return [], {"name": name, "status": "missing_pool", "pattern": pattern, **PROVENANCE.get(name, {})}

    pool_docs, pool_chars = measure_pool(files, id_col, text_col, min_score)
    if pool_chars <= 0:
        print(f"  [SKIP] {name}: no eligible text rows at {pattern}", flush=True)
        return [], {"name": name, "status": "empty_pool", "pattern": pattern, "pool_docs": pool_docs, "pool_chars": pool_chars, **PROVENANCE.get(name, {})}

    target_chars = min(char_budget, max(1, int(pool_chars * max_val_fraction)))
    used_relative_fallback = target_chars < char_budget
    out_jsonl = os.path.join(out_dir, f"val_forget_{name}.jsonl")
    chars = kept = 0
    ids = []
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for t, i in iter_rows(files, id_col, text_col, min_score):
            doc_id = str(i) if i is not None else f"{name}_{kept}"
            fout.write(json.dumps({"text": t, "doc_id": doc_id, "source_dataset": name}, ensure_ascii=False) + "\n")
            chars += len(t); kept += 1
            ids.append(doc_id)
            if chars >= target_chars:
                break
    frac = chars / pool_chars
    mode = "relative fallback" if used_relative_fallback else "absolute target"
    print(
        f"  {name}: {kept:,}/{pool_docs:,} docs, {chars/1e9:.2f}B/{pool_chars/1e9:.2f}B chars "
        f"({frac:.1%} held out; {mode}) -> {out_jsonl}",
        flush=True,
    )
    meta = {
        "name": name,
        "status": "built",
        "pattern": pattern,
        "files": files,
        "pool_docs": pool_docs,
        "pool_chars": pool_chars,
        "heldout_docs": kept,
        "heldout_chars": chars,
        "heldout_fraction": frac,
        "char_budget": char_budget,
        "max_val_fraction": max_val_fraction,
        "used_relative_fallback": used_relative_fallback,
        **PROVENANCE.get(name, {}),
    }
    return ids, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--char-budget", type=int, default=2_000_000_000)  # ~0.5B tok
    ap.add_argument("--max-val-fraction", type=float, default=float(os.environ.get("MAX_VAL_FRACTION", "0.25")))
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    # Rebuild the id-drop parquet from fresh validation selections.
    # This deliberately avoids carrying stale en/de/ru/zh ids from an earlier
    # run that used too-small staged pools.
    all_ids = []
    manifest = {
        "policy": "absolute char-budget first; if that would hold out more than max_val_fraction of the eligible pool, cap validation at max_val_fraction",
        "char_budget": a.char_budget,
        "max_val_fraction": a.max_val_fraction,
        "sets": [],
    }
    for name, (pattern, id_col, text_col, min_score) in SOURCES.items():
        ids, meta = build_one(name, pattern, id_col, text_col, min_score, a.output_dir, a.char_budget, a.max_val_fraction)
        all_ids.extend(ids)
        manifest["sets"].append(meta)

    out = os.path.join(a.output_dir, "forget_holdout_ids.parquet")
    all_ids = list(dict.fromkeys(all_ids))
    pq.write_table(pa.table({"doc_id": all_ids}), out, compression="zstd")
    manifest_path = os.path.join(a.output_dir, "forgetting_val_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2, ensure_ascii=False)
    print(f"DONE: forgetting val sets in {a.output_dir}; {len(all_ids):,} unique held-out ids -> {out}", flush=True)
    print(f"DONE: forgetting validation manifest -> {manifest_path}", flush=True)

if __name__ == "__main__":
    main()

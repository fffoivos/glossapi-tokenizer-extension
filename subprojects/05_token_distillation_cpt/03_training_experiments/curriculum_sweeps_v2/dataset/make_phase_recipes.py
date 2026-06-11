#!/usr/bin/env python3
"""Author the THREE single-bucket mix_builder recipes for the curriculum:
  hplt_only     — new-Greek HPLT stream (^HPLT/ell_Grek), drops the new-Greek val holdouts
  glossapi_only — new-Greek OpenArchives stream (^openarchives\\.gr), drops the same holdouts
  replay_only   — the 24 multilingual tiers + code + math + greek_replay, renormalized,
                  with the OLD-data forgetting holdouts dropped from the relevant sources.

mix_builder.py has NO --source flag (argparse: --recipe/--target-tokens/--tokenizer/
--output/--seed/--source-shard-index/--source-shard-count), so per-binary single-bucket
recipes are the only mechanism. The three binaries are then BLENDED at train time via the
Megatron weighted --data-path (replay% = the blend weight), so this runs ONCE.

Usage: python make_phase_recipes.py <bulk_13b.json> <out_recipes_dir>
"""
import json, sys
from pathlib import Path

BULK = Path(sys.argv[1])   # .../dataset_build/bulk_13b.json (deployed pilot recipe)
OUT  = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SC = "/iopsstor/scratch/cscs/fffoivos"
VAL_DROP    = f"{SC}/cpt_corpus/curriculum_v2/val_holdout_ids.parquet"     # new-Greek held-outs
FORGET_DROP = f"{SC}/cpt_corpus/curriculum_v2/forget_holdout_ids.parquet"  # old-data forgetting held-outs

# the replay sources whose forgetting held-outs we must exclude from training, and the
# doc-key field whose VALUE was stored in forget_holdout_ids.parquet. FineWeb
# pools use `id`, StarCoderData uses the staging-added `doc_id`, and greek_replay
# uses `source_doc_id`.
FORGET_DROP_SOURCES = {
    "replay_t1_english_edu":            "id",
    "replay_t1_deu_Latn":               "id",
    "replay_t1_rus_Cyrl":               "id",
    "replay_t1_cmn_Hani":               "id",
    "code_starcoderdata_subset":        "doc_id",
    "greek_replay_apertus_original":    "source_doc_id",
}

# Read the full staged local pool for the finite old-data validation sources.
# This is a glob so a future, provenance-checked staging pass can add more
# Apertus-source shards without changing the recipe generator; with today's
# single-shard staging it still resolves to the existing 000_00000.parquet.
EXPANDED_REPLAY_GLOBS = {
    "replay_t1_english_edu": f"{SC}/cpt_corpus/replay/eng_fineweb_edu/sample/10BT/*.parquet",
    "replay_t1_deu_Latn": f"{SC}/cpt_corpus/replay/deu_Latn_fw2hq/deu_Latn/*.parquet",
    "replay_t1_rus_Cyrl": f"{SC}/cpt_corpus/replay/rus_Cyrl_fw2hq/rus_Cyrl/*.parquet",
    "replay_t1_cmn_Hani": f"{SC}/cpt_corpus/replay/cmn_Hani_fw2hq/cmn_Hani/*.parquet",
}

STARCODER_SOURCE = {
    "name": "code_starcoderdata_subset",
    "bucket": "code",
    "share_within_bucket": 1.0,
    "id": "bigcode/starcoderdata",
    "local_parquet": f"{SC}/cpt_corpus/replay/starcoderdata_v2/*/*.parquet",
    "split": "train",
    "streaming": True,
    "text_column": "content",
    "comment": "Apertus-family code replay source, staged locally from bigcode/starcoderdata with stable doc_id.",
}

b = json.loads(BULK.read_text())
seed = b.get("seed", 20260609)
byname = {s["name"]: s for s in b["sources"]}

def single(name, src_name, regex):
    s = dict(byname[src_name]); s["weight"] = 1.0
    s["filter_field"] = "source_dataset"; s["filter_values_regex"] = regex
    s["doc_key_field"] = "source_doc_id"; s["drop_doc_keys_parquet"] = VAL_DROP
    json.dump({"name": name, "version": "v2_curriculum", "seed": seed,
               "buckets": {"greek": 1.0}, "sources": [s]},
              open(OUT / f"{name}.json", "w"), indent=1, ensure_ascii=False)

single("hplt_only",     "greek_hplt_70",        r"^HPLT/ell_Grek")
single("glossapi_only", "greek_openarchives_30", r"^openarchives\.gr")

# replay_only: every non-Greek-new source, weights renormalized to sum 1.0, forgetting drops wired.
rep = [dict(s) for s in b["sources"] if s.get("bucket") in ("replay", "code", "math", "greek_replay")]
for i, s in enumerate(rep):
    if s.get("bucket") == "code":
        replacement = dict(STARCODER_SOURCE)
        replacement["weight"] = s["weight"]
        rep[i] = replacement
tot = sum(float(s["weight"]) for s in rep)
for s in rep:
    s["weight"] = round(float(s["weight"]) / tot, 8)
    if s["name"] in EXPANDED_REPLAY_GLOBS:
        s["local_parquet"] = EXPANDED_REPLAY_GLOBS[s["name"]]
    if s["name"] in FORGET_DROP_SOURCES:
        s["doc_key_field"] = FORGET_DROP_SOURCES[s["name"]]
        s["drop_doc_keys_parquet"] = FORGET_DROP   # safe even if that source has no held-out ids (drops nothing)
buckets = {}
for s in rep:
    buckets[s["bucket"]] = round(buckets.get(s["bucket"], 0.0) + float(s["weight"]), 8)
json.dump({"name": "replay_only", "version": "v2_curriculum", "seed": seed,
           "buckets": buckets, "sources": rep},
          open(OUT / "replay_only.json", "w"), indent=1, ensure_ascii=False)

print(f"wrote 3 recipes to {OUT}")
print(f"  hplt_only / glossapi_only: drop_doc_keys_parquet={VAL_DROP}")
print(f"  replay_only: {len(rep)} sources; forgetting-drop on {sorted(set(s['name'] for s in rep) & set(FORGET_DROP_SOURCES))}")

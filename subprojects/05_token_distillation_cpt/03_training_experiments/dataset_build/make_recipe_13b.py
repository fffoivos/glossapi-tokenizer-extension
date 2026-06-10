#!/usr/bin/env python3
"""Generate the 13.5B 2-arm recipe (bulk_13b.json).

new Greek (10B) = 70% HPLT + 30% openarchives, EXCLUDING the held-out val docs
(drop_doc_keys_parquet); + replay at 24/4/2/5% of new (multilingual/code/math/
Greek-replay). Keeps the bakeoff replay/code/math sources (rescaled); replaces
the Greek sources with the two new ones.
"""
import json, sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bulk.json")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bulk_13b.json")
SC = "/iopsstor/scratch/cscs/fffoivos"
SELECTED = "${SELECTED}"
GREEK_REPLAY = f"{SC}/cpt_corpus/greek_replay/greek_replay.parquet"
VAL_HOLDOUT = f"{SC}/cpt_corpus/cpt_2arm_13b/val_holdout_ids.parquet"  # doc_id col; excluded from new Greek

P = 135.0  # parts: new 100 + ML 24 + code 4 + math 2 + greek_replay 5
NB = {"greek": round(100/P,6), "replay": round(24/P,6), "code": round(4/P,6),
      "math": round(2/P,6), "greek_replay": round(5/P,6)}

d = json.loads(SRC.read_text())
old = d["buckets"]
# keep + rescale the non-Greek bakeoff sources (replay/code/math)
kept = []
for s in d["sources"]:
    b = s.get("bucket")
    if b in ("replay", "code", "math"):
        s["weight"] = round(float(s["weight"]) * (NB[b]/old[b]), 8)
        kept.append(s)

# two NEW Greek sources (HPLT 70% + openarchives 30% of the greek bucket),
# both reading SELECTED, both excluding the held-out val doc-ids.
g = NB["greek"]
greek_sources = [
    {"name": "greek_hplt_70", "bucket": "greek", "weight": round(0.70*g, 8),
     "local_parquet": SELECTED, "split": "train", "streaming": True, "text_column": "text",
     "filter_field": "source_dataset", "filter_values_regex": "^HPLT/ell_Grek",
     "doc_key_field": "source_doc_id", "drop_doc_keys_parquet": VAL_HOLDOUT,
     "comment": "70% of new Greek = HPLT clean60; held-out val docs excluded."},
    {"name": "greek_openarchives_30", "bucket": "greek", "weight": round(0.30*g, 8),
     "local_parquet": SELECTED, "split": "train", "streaming": True, "text_column": "text",
     "filter_field": "source_dataset", "filter_values_regex": "^openarchives\\.gr",
     "doc_key_field": "source_doc_id", "drop_doc_keys_parquet": VAL_HOLDOUT,
     "comment": "30% of new Greek = openarchives.gr (glossapi); held-out val docs excluded."},
]
greek_replay = {
    "name": "greek_replay_apertus_original", "bucket": "greek_replay", "weight": NB["greek_replay"],
    "local_parquet": GREEK_REPLAY, "split": "train", "streaming": True, "text_column": "text",
    "doc_key_field": "source_doc_id",
    "comment": "5% Apertus-original Greek replay (nanochat ∩ apertus_overlap_drop). Rides the "
               "mix-wide decontaminate + anonymize passes.",
}
d["sources"] = greek_sources + kept + [greek_replay]
d["buckets"] = NB
d["name"] = "bulk_13b"
d["version"] = "v2.0_70-30_holdout"
d["description"] = ("13.5B 2-arm CPT mix. New Greek (10B) = 70% HPLT + 30% openarchives, EXCLUDING "
                    "the 0.5B×3 held-out val docs. Replay = 24% ML / 4% code / 2% math / 5% "
                    f"Greek-replay OF new. Shares-of-total: {NB}.")

OUT.write_text(json.dumps(d, indent=1, ensure_ascii=False))
per = {}
for s in d["sources"]:
    per[s["bucket"]] = per.get(s["bucket"], 0.0) + float(s["weight"])
print(f"wrote {OUT}  buckets={NB} sum={sum(NB.values()):.4f}")
print("per-bucket source-weight sums:", {k: round(v,4) for k,v in per.items()})
print("sources:", [s["name"] for s in d["sources"]])

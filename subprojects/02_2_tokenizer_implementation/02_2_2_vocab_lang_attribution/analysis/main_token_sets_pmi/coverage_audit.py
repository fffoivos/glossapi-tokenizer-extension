"""Coverage audit — which tokens are in some masked set, and which aren't.

Loads every <key>__masked.txt, takes the union, computes uncovered tokens,
and categorises each uncovered token by the reason it didn't promote.

Categorisation priority (each token gets the first reason that applies):

  1. is_special                     — Apertus special tokens
  2. T5 unknown standalone          — partial_utf8 / byte_unmapped / special status
  3. substrate (popcount == N_LANG_BITS) — universal; excluded from masked by rule
  4. bitmask_and == 0               — no in-scope locale admits any char
  5. fired only in unmapped keys    — would need char-tool coverage for that locale
  6. fires but below min_count      — too rare for confident attribution
  7. fires but PMI < δ for every L  — substrate-adjacent or shared across many langs
  8. doesn't fire anywhere          — vocab tokens never seen in 114 B-token corpus
"""
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import polars as pl


HERE = Path(__file__).resolve().parent
VLA  = HERE.parent.parent
CLM  = VLA.parent / "02_2_1_char_language_membership"

mf = json.loads((HERE / "manifest.json").read_text())
keys = mf["marginal_keys"]
ALPHA     = mf["parameters"]["alpha"]
DELTA     = mf["parameters"]["delta"]
MIN_COUNT = mf["parameters"]["min_count"]

# ── load char masks + decoded ──────────────────────────────────────────────
tbm = pl.read_parquet(CLM / "artifacts/token_language_bitmask.parquet")
def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
bm_lang = np.array([to_int(b) for b in tbm["bitmask_and"].to_list()], dtype=object)
status  = tbm["status"].to_list()
decoded = tbm["decoded_text"].to_list()
cl_manifest = json.loads((CLM / "artifacts/manifest.json").read_text())
N_LANG_BITS = cl_manifest["levels"]["language"]["bits_used"]
V = len(bm_lang)
popcount = np.array([bin(int(x)).count("1") for x in bm_lang], dtype=np.int32)

# ── load Apertus token metadata for is_special / is_byte_fragment ──────────
tok_meta = pl.read_parquet(VLA / "outputs/token_metadata.parquet")
is_special = tok_meta["is_special"].to_numpy()
is_byte_fragment = tok_meta["is_byte_fragment"].to_numpy()

# ── load histogram for total-firing per token across cap-hit + non-cap-hit ─
z = np.load(VLA / "outputs/histogram_matrix.npz", allow_pickle=True)
H = z["H"].astype(np.int64)
all_keys = list(z["canonical_keys"])
totals_per_key = H.sum(axis=1)
# Threshold for "cap-hit" key — read from PMI manifest so it tracks the
# build config (instead of hardcoded). Falls back to 1B if missing.
MARGINAL_FLOOR = mf.get("parameters", {}).get("marginal_floor", 1_000_000_000)
caphit_idx = np.where(totals_per_key >= MARGINAL_FLOOR)[0]
caphit_keys = [all_keys[i] for i in caphit_idx]
unmapped_caphit = set(mf["keys_without_lang_code_mapping"])
mapped_caphit_idx = np.array([i for i in caphit_idx if all_keys[i] not in unmapped_caphit])
unmapped_caphit_idx = np.array([i for i in caphit_idx if all_keys[i] in unmapped_caphit])
noncaphit_idx = np.array([i for i in range(len(all_keys)) if i not in set(caphit_idx.tolist())])

# Per-token firing aggregates
fire_in_mapped_caphit   = H[mapped_caphit_idx].sum(axis=0) if mapped_caphit_idx.size else np.zeros(V, dtype=np.int64)
fire_in_unmapped_caphit = H[unmapped_caphit_idx].sum(axis=0) if unmapped_caphit_idx.size else np.zeros(V, dtype=np.int64)
fire_in_noncaphit       = H[noncaphit_idx].sum(axis=0) if noncaphit_idx.size else np.zeros(V, dtype=np.int64)
fire_anywhere           = fire_in_mapped_caphit + fire_in_unmapped_caphit + fire_in_noncaphit
# Max count over MAPPED cap-hit keys only — the unmapped cap-hit keys
# can't promote anyway under Variant A, so they shouldn't influence the
# "below min_count" decision. Using ALL cap-hit (mapped + unmapped) would
# mislabel ~184 tokens whose fire was concentrated in unmapped keys.
fire_in_caphit_max = (H[mapped_caphit_idx].max(axis=0)
                      if mapped_caphit_idx.size else np.zeros(V, dtype=np.int64))

# ── parse all <key>__masked.txt files to get the union of covered IDs ──────
def parse_masked(path):
    out = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\{(\d+):", line)
        if m: out.append(int(m.group(1)))
    return out

covered = set()
per_key_sets = {}
for k in keys:
    ids = parse_masked(HERE / "tables" / f"{k}__masked.txt")
    per_key_sets[k] = set(ids)
    covered.update(ids)

uncovered = set(range(V)) - covered

print(f"Apertus vocab size:              {V:>7,}")
print(f"Covered (in some masked set):    {len(covered):>7,}  ({100*len(covered)/V:.2f}%)")
print(f"Uncovered:                       {len(uncovered):>7,}  ({100*len(uncovered)/V:.2f}%)")

# ── categorise the uncovered tokens ───────────────────────────────────────
UNKNOWN_STATUSES = {"partial_utf8", "byte_unmapped", "special"}

def classify(i):
    if is_special[i]:                            return "1_is_special"
    if status[i] in UNKNOWN_STATUSES:            return "2_T5_unknown"
    if popcount[i] == N_LANG_BITS:               return "3_substrate"
    if popcount[i] == 0:                         return "4_no_locale_admits_chars"
    # Token fires somewhere — figure out where
    fa = int(fire_anywhere[i])
    if fa == 0:                                  return "8_never_fires"
    fm = int(fire_in_mapped_caphit[i])
    fu = int(fire_in_unmapped_caphit[i])
    if fm == 0 and fu > 0:                       return "5_fires_only_in_unmapped_caphit"
    if fm == 0:                                  return "9_fires_only_in_noncaphit"
    if int(fire_in_caphit_max[i]) < MIN_COUNT:   return "6_below_min_count"
    # Otherwise: fires ≥ min_count in some mapped cap-hit, but PMI < δ everywhere
    return "7_PMI_below_delta_for_every_lang"

classes = {i: classify(i) for i in uncovered}
counter = Counter(classes.values())

print("\n=== Uncovered tokens by category ===")
print(f"  {'category':<40} {'count':>7} {'pct of uncovered':>17}")
for cat, n in sorted(counter.items()):
    print(f"  {cat:<40} {n:>7,} {100*n/len(uncovered):>16.2f}%")

# ── sample examples per category ──────────────────────────────────────────
print("\n=== Sample uncovered tokens per category (5 by firing count desc) ===")
by_cat = {}
for i, c in classes.items():
    by_cat.setdefault(c, []).append(i)
for cat in sorted(by_cat.keys()):
    ids = sorted(by_cat[cat], key=lambda i: -int(fire_anywhere[i]))[:5]
    print(f"\n  {cat}:")
    for i in ids:
        print(f"    id={i:>6}  fire_anywhere={int(fire_anywhere[i]):>11,}  "
              f"pc={popcount[i]:>2}  decoded={decoded[i]!r}")

# ── save full classification ──────────────────────────────────────────────
rows = [{
    "token_id":           int(i),
    "decoded":            "" if decoded[i] is None else decoded[i],
    "category":           classes[i],
    "popcount":           int(popcount[i]),
    "fire_anywhere":      int(fire_anywhere[i]),
    "fire_mapped_caphit": int(fire_in_mapped_caphit[i]),
    "fire_unmapped_caphit": int(fire_in_unmapped_caphit[i]),
    "fire_noncaphit":     int(fire_in_noncaphit[i]),
} for i in uncovered]
df = pl.DataFrame(rows, schema={"token_id": pl.Int64, "decoded": pl.String,
                                 "category": pl.String, "popcount": pl.Int64,
                                 "fire_anywhere": pl.Int64,
                                 "fire_mapped_caphit": pl.Int64,
                                 "fire_unmapped_caphit": pl.Int64,
                                 "fire_noncaphit": pl.Int64})
df.write_csv(HERE / "uncovered_tokens.tsv", separator="\t")
print(f"\nwrote uncovered_tokens.tsv ({len(rows):,} rows)")

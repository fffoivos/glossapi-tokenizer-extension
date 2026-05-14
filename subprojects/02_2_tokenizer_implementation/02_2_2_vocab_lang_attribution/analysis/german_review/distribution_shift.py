"""Quantify the English↔German distribution shift inside the Latin family.

Set overlap (Jaccard 0.94) is misleading at 1 B samples because a single
quotation across languages already triggers "fired in both". This script
slices the Latin family by the *rate ratio* count_de / count_en and shows
mass per log-ratio bin, plus rank-intersection at top-K.

Outputs:
  tables/log_ratio_histogram.tsv          — per-bin token-types and mass
                                             (German mass + English mass).
  tables/topk_rank_intersection.tsv       — |top-K(de) ∩ top-K(en)| / K.
  tables/shared_mass_partition.tsv        — German mass split into
                                             "shared-comparable" (|log r| < 1),
                                             "shared-skewed", "near-singleton".
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
VLA  = HERE.parent.parent
CLM  = VLA.parent / "02_2_1_char_language_membership"
TABLES = HERE / "tables"; TABLES.mkdir(exist_ok=True)

z = np.load(VLA / "outputs/histogram_matrix.npz", allow_pickle=True)
H = z["H"].astype(np.int64); ck = list(z["canonical_keys"])
DEU = ck.index("deu_Latn"); ENG = ck.index("eng_Latn")
de = H[DEU]; en = H[ENG]
total_de = int(de.sum()); total_en = int(en.sum())
V = H.shape[1]

tok = pl.read_parquet(VLA / "outputs/token_metadata.parquet")
decoded = tok["decoded_string"].to_list()

clm = pl.read_parquet(CLM / "artifacts/token_language_bitmask.parquet")
def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
bm_and = [to_int(b) for b in clm["bitmask_and"].to_list()]
status = clm["status"].to_list()
manifest = json.load(open(CLM / "artifacts/manifest.json"))

HAN_SCRIPTS = {"Hans", "Hant", "Jpan"}
script_mask = {}
han_mask = 0
for L in manifest["languages"]:
    bit = 1 << L["bit"]
    if L["script"] in HAN_SCRIPTS:
        han_mask |= bit
    else:
        script_mask.setdefault(L["script"], 0)
        script_mask[L["script"]] |= bit
script_mask["Han"] = han_mask
N_BITS = 55
popcount = np.array([bin(bm_and[i]).count("1") for i in range(V)], dtype=np.int32)

unknown_statuses = {"partial_utf8", "byte_unmapped", "special"}
def family(i):
    s = status[i]
    if s in unknown_statuses: return "unknown_standalone"
    m = bm_and[i]
    if popcount[i] == N_BITS: return "substrate"
    if m == 0: return "no_script_admits"
    scripts = [name for name, mask in script_mask.items() if (m & mask)]
    return f"family:{scripts[0]}" if len(scripts) == 1 else "mixed_scripts"

fam = np.array([family(i) for i in range(V)])
latin_mask = (fam == "family:Latn")
substrate_mask = (fam == "substrate")

# ── Log-ratio per token (Latin family, fired in either) ────────────────────
# We use Laplace smoothing so that count==0 maps to log finite
ALPHA = 0.5
log_ratio = np.full(V, np.nan, dtype=np.float64)
for i in np.where(latin_mask)[0]:
    if de[i] == 0 and en[i] == 0: continue
    log_ratio[i] = np.log10((de[i] + ALPHA) / (en[i] + ALPHA))

# Bin by log-ratio
edges = [-np.inf, -3, -2, -1, -0.5, 0.5, 1, 2, 3, np.inf]
labels = [
    "<= -3        (≥1000× English)",
    "-3 .. -2     (100–1000× English)",
    "-2 .. -1     (10–100× English)",
    "-1 .. -0.5   (3–10× English)",
    "-0.5 .. 0.5  (within 3× — comparable)",
    "0.5 .. 1     (3–10× German)",
    "1 .. 2       (10–100× German)",
    "2 .. 3       (100–1000× German)",
    "> 3          (≥1000× German)",
]
in_pool = latin_mask & ~np.isnan(log_ratio)
print("=== Log-ratio histogram (Latin family only, both-Laplace-smoothed) ===")
print(f"{'bin (log10 de/en)':<40} {'tokens':>8} {'de_mass':>14} {'en_mass':>14} {'de_pct':>7} {'en_pct':>7}")
rows = []
total_latin_de = int(de[in_pool].sum())
total_latin_en = int(en[in_pool].sum())
for i in range(len(labels)):
    lo, hi = edges[i], edges[i+1]
    mask = in_pool & (log_ratio > lo) & (log_ratio <= hi)
    n = int(mask.sum())
    dm = int(de[mask].sum()); em = int(en[mask].sum())
    print(f"{labels[i]:<40} {n:>8,} {dm:>14,} {em:>14,} "
          f"{100*dm/total_latin_de:>6.2f}% {100*em/total_latin_en:>6.2f}%")
    rows.append({"bin": labels[i], "edge_lo": lo, "edge_hi": hi,
                 "tokens": n, "de_mass": dm, "en_mass": em,
                 "de_mass_pct_of_latin": 100*dm/total_latin_de,
                 "en_mass_pct_of_latin": 100*em/total_latin_en})
with open(TABLES / "log_ratio_histogram.tsv", "w") as f:
    f.write("bin\tedge_lo\tedge_hi\ttokens\tde_mass\ten_mass\tde_pct_of_latin\ten_pct_of_latin\n")
    for r in rows:
        f.write(f"{r['bin']}\t{r['edge_lo']}\t{r['edge_hi']}\t{r['tokens']}\t"
                f"{r['de_mass']}\t{r['en_mass']}\t"
                f"{r['de_mass_pct_of_latin']:.4f}\t{r['en_mass_pct_of_latin']:.4f}\n")

# ── Rank intersection at top-K (Latin family) ──────────────────────────────
de_lat_order = np.argsort(-np.where(latin_mask, de, -1))
en_lat_order = np.argsort(-np.where(latin_mask, en, -1))
print("\n=== Top-K rank intersection in the Latin family ===")
print(f"  {'K':>8} {'|top-K(de) ∩ top-K(en)|':>26} {'pct of K':>10}")
rows_k = []
for K in [10, 50, 100, 500, 1_000, 5_000, 10_000, 20_000, 50_000]:
    de_top = set(de_lat_order[:K].tolist())
    en_top = set(en_lat_order[:K].tolist())
    inter = len(de_top & en_top)
    print(f"  {K:>8,} {inter:>26,} {100*inter/K:>9.2f}%")
    rows_k.append({"K": K, "intersection": inter, "pct": 100*inter/K})
with open(TABLES / "topk_rank_intersection.tsv", "w") as f:
    f.write("K\tintersection\tpct_of_K\n")
    for r in rows_k:
        f.write(f"{r['K']}\t{r['intersection']}\t{r['pct']:.4f}\n")

# ── Shared mass partition by ratio class ───────────────────────────────────
print("\n=== German Latin mass split by ratio class (compared to English) ===")
classes = [
    ("near-singleton-english",     "fired only in English (de=0)",  (de == 0) & latin_mask & (en > 0)),
    ("near-singleton-german",      "fired only in German (en=0)",   (en == 0) & latin_mask & (de > 0)),
    ("english-dominant >=10×",     "log_ratio <= -1, shared",      latin_mask & (de > 0) & (en > 0) & (log_ratio <= -1)),
    ("english-leaning 3–10×",      "-1 < log_ratio <= -0.5",       latin_mask & (de > 0) & (en > 0) & (log_ratio > -1) & (log_ratio <= -0.5)),
    ("comparable (within 3×)",     "|log_ratio| < 0.5",            latin_mask & (de > 0) & (en > 0) & (log_ratio > -0.5) & (log_ratio < 0.5)),
    ("german-leaning 3–10×",       "0.5 <= log_ratio < 1",         latin_mask & (de > 0) & (en > 0) & (log_ratio >= 0.5) & (log_ratio < 1)),
    ("german-dominant >=10×",      "log_ratio >= 1, shared",       latin_mask & (de > 0) & (en > 0) & (log_ratio >= 1)),
]
print(f"  {'class':<28} {'tokens':>8} {'de_mass':>14} {'en_mass':>14} {'de_pct':>7} {'en_pct':>7}")
rows_c = []
for name, _desc, mask in classes:
    n = int(mask.sum())
    dm = int(de[mask].sum()); em = int(en[mask].sum())
    print(f"  {name:<28} {n:>8,} {dm:>14,} {em:>14,} "
          f"{100*dm/total_latin_de:>6.2f}% {100*em/total_latin_en:>6.2f}%")
    rows_c.append({"class": name, "tokens": n, "de_mass": dm, "en_mass": em,
                   "de_pct_of_latin": 100*dm/total_latin_de,
                   "en_pct_of_latin": 100*em/total_latin_en})
with open(TABLES / "shared_mass_partition.tsv", "w") as f:
    f.write("class\ttokens\tde_mass\ten_mass\tde_pct_of_latin\ten_pct_of_latin\n")
    for r in rows_c:
        f.write(f"{r['class']}\t{r['tokens']}\t{r['de_mass']}\t{r['en_mass']}\t"
                f"{r['de_pct_of_latin']:.4f}\t{r['en_pct_of_latin']:.4f}\n")

# ── Substrate contrast ─────────────────────────────────────────────────────
print("\n=== Substrate (for contrast) ===")
sub_n = int(substrate_mask.sum())
sub_de = int(de[substrate_mask].sum()); sub_en = int(en[substrate_mask].sum())
print(f"  substrate tokens: {sub_n:,}, de_mass {sub_de:,}, en_mass {sub_en:,}")
print(f"  ratio de/en for top 10 substrate by joint count:")
sub_idx = np.where(substrate_mask)[0]
sub_idx = sub_idx[np.argsort(-(de[sub_idx] + en[sub_idx]))][:10]
for i in sub_idx:
    r = (de[i] + 0.5) / (en[i] + 0.5)
    print(f"    {decoded[i]!r:<10} de={int(de[i]):>10,}  en={int(en[i]):>10,}  ratio {r:>6.2f}")

print("\n[done]")

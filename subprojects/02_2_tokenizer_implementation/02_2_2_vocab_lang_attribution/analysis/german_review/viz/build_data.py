"""Build data.json for the en-de distribution viz.

For every fired token we record per-language tier under three samples:
  - de:        deu_Latn (FineWeb2-HQ)
  - en_fwhq:   eng_Latn_fineweb_hq (FineWeb-HQ)
  - en_wiki:   eng_Latn (clean-wikipedia)

Tier values per (token, language): T0 | T2 | T3 | T4 | T5
(Skipping T1 for now — empty for en and de under v4; documented in
`tiered_attribution_report.md`.)

This lets the frontend pick:
  • a "scope" (strict T0 / premise T0+T2 / premise+substrate T0+T2+T3)
  • a target language (en / de)
  • for en, a source variant (FineWeb-HQ / wiki)

The plots filter to the selected scope and pair the chosen English
variant against German.
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
VLA  = HERE.parent.parent.parent
CLM  = VLA.parent / "02_2_1_char_language_membership"

# ── load firing counts ─────────────────────────────────────────────────────
z = np.load(VLA / "outputs/histogram_matrix.npz", allow_pickle=True)
H = z["H"].astype(np.int64); ck = list(z["canonical_keys"])
DEU      = ck.index("deu_Latn")
ENG_FWHQ = ck.index("eng_Latn_fineweb_hq")
ENG_WIKI = ck.index("eng_Latn")
de       = H[DEU]
en_fwhq  = H[ENG_FWHQ]
en_wiki  = H[ENG_WIKI]
total_de       = int(de.sum())
total_en_fwhq  = int(en_fwhq.sum())
total_en_wiki  = int(en_wiki.sum())
V = H.shape[1]

tok = pl.read_parquet(VLA / "outputs/token_metadata.parquet")
decoded = tok["decoded_string"].to_list()

# ── load char masks (v4) ───────────────────────────────────────────────────
clm = pl.read_parquet(CLM / "artifacts/token_language_bitmask.parquet")
def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
bm_lang  = [to_int(b) for b in clm["bitmask_and"].to_list()]
status   = clm["status"].to_list()

manifest = json.load(open(CLM / "artifacts/manifest.json"))
lang_bit = {L["code"]: L["bit"] for L in manifest["languages"]}
DE_BIT = lang_bit["de"]
EN_BIT = lang_bit["en"]
N_LANG_BITS  = manifest["levels"]["language"]["bits_used"]
UNKNOWN = {"partial_utf8", "byte_unmapped", "special"}

popcount = np.array([bin(bm_lang[i]).count("1") for i in range(V)], dtype=np.int32)

def tier(i, L_bit):
    if status[i] in UNKNOWN: return "T5"
    bm  = bm_lang[i]
    pc  = popcount[i]
    has_L = ((bm >> L_bit) & 1) == 1
    if pc == 1 and has_L:        return "T0"
    if pc == N_LANG_BITS:        return "T3"
    if not has_L:                return "T4"
    return "T2"

de_tier      = [tier(i, DE_BIT) for i in range(V)]
en_tier      = [tier(i, EN_BIT) for i in range(V)]

fired_de       = de       > 0
fired_en_fwhq  = en_fwhq  > 0
fired_en_wiki  = en_wiki  > 0
fired_any = fired_de | fired_en_fwhq | fired_en_wiki

records = []
for i in range(V):
    if not fired_any[i]: continue
    records.append({
        "i":   int(i),
        "d":   decoded[i],
        "de":  int(de[i]),
        "ef":  int(en_fwhq[i]),
        "ew":  int(en_wiki[i]),
        "td":  de_tier[i],
        "te":  en_tier[i],    # tier is the same for fwhq and wiki — depends on chars, not counts
        "pc":  int(popcount[i]),
    })

# ── log-ratio histogram bins, mass-weighted, per (English variant × tier) ──
ALPHA = 0.5
EDGES = [-float("inf"), -4, -3, -2.5, -2, -1.5, -1, -0.5, 0,
         0.5, 1, 1.5, 2, 2.5, 3, 4, float("inf")]

def label_edge(v):
    if v == float("inf"):  return "+inf"
    if v == -float("inf"): return "-inf"
    return f"{v:+.1f}"

def build_histogram(en_counts, total_en):
    """Bin tokens by log10((de + α) / (en + α)); stack mass by de-tier."""
    bins = []
    for k in range(len(EDGES) - 1):
        bins.append({
            "lo": EDGES[k], "hi": EDGES[k+1],
            "label": f"{label_edge(EDGES[k])}…{label_edge(EDGES[k+1])}",
            "tokens": 0,
            "de_mass_T0": 0, "de_mass_T2": 0, "de_mass_T3": 0,
            "en_mass_T0": 0, "en_mass_T2": 0, "en_mass_T3": 0,
        })
    for r in records:
        if r["td"] not in ("T0", "T2", "T3"): continue
        de_c = r["de"]; en_c = en_counts[r["i"]]
        rat = np.log10((de_c + ALPHA) / (en_c + ALPHA))
        for b in bins:
            if b["lo"] < rat <= b["hi"]:
                b["tokens"] += 1
                b[f"de_mass_{r['td']}"] += de_c
                b[f"en_mass_{r['td']}"] += int(en_c)
                break
    return bins

hist_fwhq = build_histogram(en_fwhq, total_en_fwhq)
hist_wiki = build_histogram(en_wiki, total_en_wiki)

# ── rank-rank pairs (premise pool — T0 + T2, fired in both) per en variant ─
def rankrank_pairs(en_counts):
    pool = [r for r in records
            if r["td"] in ("T0", "T2") and r["de"] > 0 and en_counts[r["i"]] > 0]
    de_sorted = sorted(pool, key=lambda r: -r["de"])
    rank_de = {r["i"]: k+1 for k, r in enumerate(de_sorted)}
    en_sorted = sorted(pool, key=lambda r: -en_counts[r["i"]])
    rank_en = {r["i"]: k+1 for k, r in enumerate(en_sorted)}
    return [{"i": r["i"], "d": r["d"], "td": r["td"],
             "rde": rank_de[r["i"]], "ren": rank_en[r["i"]],
             "de": r["de"], "en": int(en_counts[r["i"]])}
            for r in pool]

rr_fwhq = rankrank_pairs(en_fwhq)
rr_wiki = rankrank_pairs(en_wiki)

# ── tier totals per sample for the stats banner ────────────────────────────
TIERS = ["T0", "T2", "T3", "T4", "T5"]
def tier_totals(tier_list, counts):
    out = {}
    for t in TIERS:
        ids = [i for i in range(V) if tier_list[i] == t and counts[i] > 0]
        out[t] = {"tokens": len(ids), "mass": int(counts[ids].sum()) if ids else 0}
    return out

de_tier_totals       = tier_totals(de_tier, de)
en_tier_totals_fwhq  = tier_totals(en_tier, en_fwhq)
en_tier_totals_wiki  = tier_totals(en_tier, en_wiki)

out = {
    "summary": {
        "total_de":      total_de,
        "total_en_fwhq": total_en_fwhq,
        "total_en_wiki": total_en_wiki,
        "fired_either":  len(records),
        "de_tier_totals":      de_tier_totals,
        "en_tier_totals_fwhq": en_tier_totals_fwhq,
        "en_tier_totals_wiki": en_tier_totals_wiki,
        "premise_text":        ("In each language's dataset, tokens whose chars are "
                                "all L-admissible default to L under T2 (premise). "
                                "T0 is char-evidenced. T3 is substrate. T4 is "
                                "char-excluded. T5 is unknown standalone."),
        "modes": {
            "strict":       "T0 only — no assumption",
            "premise":      "T0 + T2 — with dataset-premise",
            "premise_sub":  "T0 + T2 + T3 — with premise plus substrate infra",
            "all":          "all fired tokens (no scope filter)",
        },
    },
    "tokens":          records,
    "log_ratio_bins":  {"fwhq": hist_fwhq, "wiki": hist_wiki},
    "rank_rank":       {"fwhq": rr_fwhq,   "wiki": rr_wiki},
}
out_path = HERE / "data.json"
out_path.write_text(json.dumps(out, ensure_ascii=False))
print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
print(f"  fired in any of three:  {len(records):,}")
print(f"  de   tier totals: {de_tier_totals}")
print(f"  fwhq tier totals: {en_tier_totals_fwhq}")
print(f"  wiki tier totals: {en_tier_totals_wiki}")
print(f"  rank-rank pool — fwhq: {len(rr_fwhq):,}, wiki: {len(rr_wiki):,}")

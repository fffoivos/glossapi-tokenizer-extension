"""Tiered attribution of the German firing histogram.

Implements the tier policy from
`02_2_tokenizer_implementation/02_2_3_token_classification/PLAN.md` inline.
Once that sub-subproject is built out, this script becomes a consumer
of its artifact; for now it reads 02_2_1_char_language_membership directly.

Tiers (German, L = de, family = Germanic-Latn, script = Latn):
  T0  definitely-de              bitmask_and == {de} (popcount 1)
  T1  definitely-germanic-latn   family_and  == {Germanic-Latn} ∧ bitmask_and has de-bit
                                 (excludes T0)
  T2  could-be-de (premise)      bitmask_and has de-bit ∧ popcount < 55
                                 (excludes T0, T1)
  T3  substrate                  popcount(bitmask_and) == 55
  T4  excluded (non-de char)     bitmask_and has NO de-bit ∧ status evaluable
  T5  unknown standalone         status ∈ {partial_utf8, byte_unmapped, special}

Outputs:
  tables/tier_summary.tsv        — per-tier token-types and mass.
  tables/tier_top200_<tier>.tsv  — top 200 fired tokens per tier.
  tier_summary.json              — single-glance numbers + provenance.
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
VLA  = HERE.parent.parent
CLM  = VLA.parent / "02_2_1_char_language_membership"
TABLES = HERE / "tables"; TABLES.mkdir(exist_ok=True)

# ── load char masks (v4) ───────────────────────────────────────────────────
clm = pl.read_parquet(CLM / "artifacts/token_language_bitmask.parquet")
def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
bm_lang   = [to_int(b) for b in clm["bitmask_and"].to_list()]
bm_family = [to_int(b) for b in clm["family_and"].to_list()]
bm_script = [to_int(b) for b in clm["script_and"].to_list()]
status    = clm["status"].to_list()
decoded   = clm["decoded_text"].to_list()
V = len(bm_lang)

manifest = json.load(open(CLM / "artifacts/manifest.json"))
lang_bit  = {L["code"]: L["bit"] for L in manifest["languages"]}
fam_bit   = {f["code"]: f["bit"] for f in manifest["families"]}
script_bit = {s["code"]: s["bit"] for s in manifest["scripts"]}

DE_BIT          = lang_bit["de"]                 # = 3
GERMANIC_BIT    = fam_bit["Germanic-Latn"]       # = 1
LATN_BIT        = script_bit["Latn"]             # = 0
UNKNOWN_STATUSES = {"partial_utf8", "byte_unmapped", "special"}
N_LANG_BITS = 55
LANG_MASK_ALL = (1 << N_LANG_BITS) - 1            # popcount 55

# ── load German firing ─────────────────────────────────────────────────────
z = np.load(VLA / "outputs/histogram_matrix.npz", allow_pickle=True)
H = z["H"].astype(np.int64); ck = list(z["canonical_keys"])
DEU = ck.index("deu_Latn")
de  = H[DEU]
total_de = int(de.sum())

# ── tier assignment per fired token ────────────────────────────────────────
def tier_of(i):
    if status[i] in UNKNOWN_STATUSES:
        return ("T5_unknown",           "status_" + status[i])
    bm  = bm_lang[i]
    fam = bm_family[i]
    pc  = bin(bm).count("1")
    has_de = ((bm >> DE_BIT) & 1) == 1
    if pc == 1 and has_de:
        return ("T0_definitely_de",     "bitmask_and_only_de")
    if pc == N_LANG_BITS:
        return ("T3_substrate",         "all_55_locales_admit")
    if not has_de:
        # token has decoded text but de-bit not set → at least one non-de char
        # Subdivide T4 basis based on script_and
        scr = bm_script[i]
        if ((scr >> LATN_BIT) & 1) == 1 and pc > 0:
            return ("T4_excluded",      "non_de_latin_char")
        if pc == 0:
            return ("T4_excluded",      "no_in_scope_locale_admits")
        return ("T4_excluded",          "foreign_script")
    # has de-bit, not popcount-1, not substrate → T1 or T2
    fam_pc = bin(fam).count("1")
    has_germanic_only = (fam == (1 << GERMANIC_BIT))
    if has_germanic_only:
        return ("T1_definitely_germanic_latn",
                "family_and_only_germanic_latn_premise_de")
    return ("T2_premise_de",            "bitmask_and_has_de_bit")

fired_idx = np.where(de > 0)[0]
rows = []
for i in fired_idx:
    t, basis = tier_of(int(i))
    rows.append({
        "token_id":  int(i),
        "decoded":   decoded[i],
        "count":     int(de[i]),
        "tier":      t,
        "basis":     basis,
        "popcount_language": bin(bm_lang[i]).count("1"),
    })

# ── per-tier summary ───────────────────────────────────────────────────────
TIERS = ["T0_definitely_de", "T1_definitely_germanic_latn",
         "T2_premise_de", "T3_substrate", "T4_excluded", "T5_unknown"]
TIER_DESC = {
    "T0_definitely_de":         "char-evidenced (bitmask_and == {de} only)",
    "T1_definitely_germanic_latn": "char-evidenced family (family_and == {Germanic-Latn} only) — premise picks de over en/nl/da/nb/sv/is",
    "T2_premise_de":            "premise (bitmask_and has de-bit; chars also admit other locales)",
    "T3_substrate":             "substrate (every in-scope locale admits all chars)",
    "T4_excluded":              "char evidence rules de out (token has a non-de-admissible char)",
    "T5_unknown":               "char tool cannot evaluate (partial_utf8 / byte_unmapped / special)",
}

print(f"{'tier':<33} {'tokens':>8} {'mass':>14} {'mass%':>7}  meaning")
summary_rows = []
for t in TIERS:
    sub = [r for r in rows if r["tier"] == t]
    n = len(sub); m = sum(r["count"] for r in sub)
    print(f"{t:<33} {n:>8,} {m:>14,} {100*m/total_de:>6.2f}%  {TIER_DESC[t]}")
    summary_rows.append({"tier": t, "tokens": n, "mass": m,
                          "mass_pct": 100*m/total_de,
                          "description": TIER_DESC[t]})

pl.DataFrame(summary_rows).write_csv(TABLES / "tier_summary.tsv", separator="\t")

# ── per-tier top-200 ───────────────────────────────────────────────────────
for t in TIERS:
    sub = [r for r in rows if r["tier"] == t]
    if not sub: continue
    sub.sort(key=lambda r: -r["count"])
    with open(TABLES / f"tier_top200_{t}.tsv", "w") as f:
        f.write("rank\ttoken_id\tdecoded\tcount\tmass_pct\tbasis\tpopcount\n")
        for rk, r in enumerate(sub[:200], 1):
            f.write(f"{rk}\t{r['token_id']}\t{r['decoded']!r}\t{r['count']}\t"
                    f"{100*r['count']/total_de:.4f}\t{r['basis']}\t{r['popcount_language']}\n")

# ── per-tier basis breakdown ───────────────────────────────────────────────
print("\n=== basis breakdown per tier ===")
for t in TIERS:
    sub = [r for r in rows if r["tier"] == t]
    if not sub: continue
    basis_counts = {}
    for r in sub:
        b = r["basis"]
        basis_counts.setdefault(b, [0, 0])
        basis_counts[b][0] += 1
        basis_counts[b][1] += r["count"]
    print(f"  {t}")
    for b, (n, m) in sorted(basis_counts.items(), key=lambda x: -x[1][1]):
        print(f"    {b:<45} {n:>8,} {m:>14,} {100*m/total_de:>6.2f}%")

# ── single-glance JSON ─────────────────────────────────────────────────────
out = {
    "lang_code":    "de",
    "dataset":      "deu_Latn",
    "total_sample": total_de,
    "fired":        len(fired_idx),
    "schema_premise": "In the German dataset, a token whose chars are de-admissible defaults to de under tier T2 (premise). T0/T1 are char-evidenced; T2 is defeasible.",
    "char_membership_schema_version": manifest.get("schema_version", "unknown"),
    "tiers":        summary_rows,
}
(HERE / "tier_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nwrote tier_summary.json")
print("[done]")

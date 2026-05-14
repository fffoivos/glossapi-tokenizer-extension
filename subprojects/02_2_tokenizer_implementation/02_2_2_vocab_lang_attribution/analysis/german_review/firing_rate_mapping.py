"""German (deu_Latn) firing-rate mapping + overlap with English (eng_Latn).

Same framing as english_review/firing_rate_mapping.py: bitmask is for
script-family exclusion, firing rate maps a token to a language. This
script additionally compares the German Latin-family firings to the
English Latin-family firings.

Outputs:
  tables/family_firing_summary.tsv
  tables/latin_rank_frequency.tsv
  tables/latin_cumulative_mass.tsv
  tables/family_top20_*.tsv
  tables/en_de_overlap_summary.tsv      — set + mass overlap in Latin family.
  tables/en_de_overlap_by_count.tsv     — overlap mass at count >= threshold.
  tables/german_distinctive_top200.tsv  — top de-fired tokens with de bit set
                                          but en bit NOT set (ä/ö/ü/ß-bearing).
  tables/german_only_top200.tsv         — tokens fired in German but NOT
                                          in English (Latin family).
  tables/english_only_top200.tsv        — Latin tokens fired in English but
                                          NOT in German.
  tables/joint_top50_scatter.tsv        — top 50 tokens in either lang with
                                          counts in both (for visual review).
  firing_rate_summary.json
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
VLA  = HERE.parent.parent
CLM  = VLA.parent / "02_2_1_char_language_membership"
TABLES = HERE / "tables"; TABLES.mkdir(exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────
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

# ── script-family masks (Han = Hans+Hant+Jpan) ────────────────────────────
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
EN_BIT = 0; DE_BIT = 3
unknown_statuses = {"partial_utf8", "byte_unmapped", "special"}

def family(i):
    s = status[i]
    if s in unknown_statuses: return "unknown_standalone"
    m = bm_and[i]
    if popcount[i] == N_BITS: return "substrate"
    if m == 0: return "no_script_admits"
    scripts = [name for name, mask in script_mask.items() if (m & mask)]
    if len(scripts) == 1: return f"family:{scripts[0]}"
    return "mixed_scripts"

fam = np.array([family(i) for i in range(V)])
fired_de = de > 0
fired_en = en > 0

# ── per-family summary for German ──────────────────────────────────────────
rows = []
for f in sorted(set(fam.tolist())):
    in_fam = (fam == f)
    in_fam_fired = in_fam & fired_de
    rows.append({
        "family":       f,
        "vocab_total":  int(in_fam.sum()),
        "fired":        int(in_fam_fired.sum()),
        "mass":         int(de[in_fam_fired].sum()),
        "mass_pct":     100 * int(de[in_fam_fired].sum()) / total_de,
    })
rows.sort(key=lambda r: -r["mass"])
df = pl.DataFrame(rows)
df.write_csv(TABLES / "family_firing_summary.tsv", separator="\t")
print("=== Per-family firing summary (deu_Latn) ===")
with pl.Config(tbl_rows=30):
    print(df)

# ── top tokens per family ──────────────────────────────────────────────────
for f in sorted(set(fam.tolist())):
    in_fam_fired_idx = np.where((fam == f) & fired_de)[0]
    if in_fam_fired_idx.size == 0: continue
    top = in_fam_fired_idx[np.argsort(-de[in_fam_fired_idx])][:20]
    safe = f.replace(":", "_").replace(" ", "_")
    with open(TABLES / f"family_top20_{safe}.tsv", "w") as out:
        out.write("rank\ttoken_id\tdecoded\tcount\tmass_pct\tpopcount\n")
        for r, i in enumerate(top, 1):
            out.write(f"{r}\t{i}\t{decoded[i]!r}\t{int(de[i])}\t"
                      f"{100*float(de[i])/total_de:.4f}\t{popcount[i]}\n")

# ── Latin-family rank-frequency ────────────────────────────────────────────
latin_fired_idx = np.where((fam == "family:Latn") & fired_de)[0]
latin_counts = de[latin_fired_idx]
sort_order = np.argsort(-latin_counts)
latin_ranked_idx = latin_fired_idx[sort_order]
latin_ranked_counts = latin_counts[sort_order]
N_lat = latin_ranked_counts.size
total_latin_de = int(latin_ranked_counts.sum())
cum = np.cumsum(latin_ranked_counts)

with open(TABLES / "latin_rank_frequency.tsv", "w") as f:
    f.write("rank\ttoken_id\tdecoded\tcount\tcum_count\tcum_pct_of_latin\tcum_pct_of_total\n")
    for r in range(N_lat):
        i = latin_ranked_idx[r]
        f.write(f"{r+1}\t{i}\t{decoded[i]!r}\t{int(latin_ranked_counts[r])}\t"
                f"{int(cum[r])}\t{100*cum[r]/total_latin_de:.4f}\t"
                f"{100*cum[r]/total_de:.4f}\n")

print("\n=== Latin family — cumulative mass at rank cutoffs (German) ===")
cutoffs_rank = [10, 100, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 50_000, N_lat]
rows_cut = []
print(f"{'top-N':>10} {'last count':>12} {'cum_mass':>14} {'pct_of_latin':>12} {'pct_of_total':>12}")
for k in cutoffs_rank:
    if k > N_lat: break
    cm = int(cum[k-1]); lc = int(latin_ranked_counts[k-1])
    print(f"{k:>10,} {lc:>12,} {cm:>14,} {100*cm/total_latin_de:>11.3f}% {100*cm/total_de:>11.3f}%")
    rows_cut.append({"top_n_rank": k, "last_count": lc, "cum_mass": cm,
                     "pct_of_latin_mass": 100*cm/total_latin_de,
                     "pct_of_total": 100*cm/total_de})

print("\n=== Latin family — token-types and mass at count >= threshold (German) ===")
thresholds = [1, 5, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
print(f"{'count >=':>10} {'tokens':>10} {'cum_mass':>14} {'pct_of_latin':>12} {'pct_of_total':>12}")
rows_cnt = []
for t in thresholds:
    mask = latin_counts >= t
    n = int(mask.sum()); m = int(latin_counts[mask].sum())
    print(f"{t:>10,} {n:>10,} {m:>14,} {100*m/total_latin_de:>11.3f}% {100*m/total_de:>11.3f}%")
    rows_cnt.append({"count_threshold": t, "tokens_at_or_above": n,
                     "cum_mass": m, "pct_of_latin_mass": 100*m/total_latin_de,
                     "pct_of_total": 100*m/total_de})
with open(TABLES / "latin_cumulative_mass.tsv", "w") as f:
    f.write("cutoff_kind\tparam\ttokens\tcum_mass\tpct_of_latin\tpct_of_total\n")
    for r in rows_cut:
        f.write(f"rank\t{r['top_n_rank']}\t{r['top_n_rank']}\t{r['cum_mass']}\t"
                f"{r['pct_of_latin_mass']:.4f}\t{r['pct_of_total']:.4f}\n")
    for r in rows_cnt:
        f.write(f"count\t{r['count_threshold']}\t{r['tokens_at_or_above']}\t"
                f"{r['cum_mass']}\t{r['pct_of_latin_mass']:.4f}\t"
                f"{r['pct_of_total']:.4f}\n")

# ── Mass quantiles (Latin family, German) ─────────────────────────────────
print("\n=== Mass quantiles (Latin family, German) ===")
for q in [0.50, 0.80, 0.90, 0.95, 0.99, 0.999]:
    k = int(np.searchsorted(cum, q * total_latin_de)) + 1
    print(f"  {100*q:>5.1f}% reached at rank {k:>7,}  "
          f"(last count {int(latin_ranked_counts[k-1]):,})")

# ════════════════════════════════════════════════════════════════════════
# OVERLAP WITH ENGLISH
# ════════════════════════════════════════════════════════════════════════

latin_mask = (fam == "family:Latn")
latin_fired_de_set = set(np.where(latin_mask & fired_de)[0].tolist())
latin_fired_en_set = set(np.where(latin_mask & fired_en)[0].tolist())
both = latin_fired_de_set & latin_fired_en_set
de_only = latin_fired_de_set - latin_fired_en_set
en_only = latin_fired_en_set - latin_fired_de_set
union = latin_fired_de_set | latin_fired_en_set

print("\n=== Latin-family token-set overlap (English × German) ===")
print(f"  fired in English (Latin)     : {len(latin_fired_en_set):>7,}")
print(f"  fired in German  (Latin)     : {len(latin_fired_de_set):>7,}")
print(f"  fired in BOTH                : {len(both):>7,}   ({100*len(both)/len(union):.2f}% of union)")
print(f"  fired in English ONLY        : {len(en_only):>7,}")
print(f"  fired in German  ONLY        : {len(de_only):>7,}")
print(f"  Jaccard                      : {len(both)/len(union):.4f}")

# Mass overlap — how much of each language's mass falls on shared tokens
both_idx = np.array(sorted(both))
de_only_idx = np.array(sorted(de_only))
en_only_idx = np.array(sorted(en_only))
de_mass_both = int(de[both_idx].sum()) if both_idx.size else 0
en_mass_both = int(en[both_idx].sum()) if both_idx.size else 0
de_mass_de_only = int(de[de_only_idx].sum()) if de_only_idx.size else 0
en_mass_en_only = int(en[en_only_idx].sum()) if en_only_idx.size else 0
print(f"\n  German Latin mass on shared  : {de_mass_both:>14,}  ({100*de_mass_both/total_latin_de:.3f}% of German Latin)")
print(f"  German Latin mass on de-only : {de_mass_de_only:>14,}  ({100*de_mass_de_only/total_latin_de:.3f}% of German Latin)")
total_latin_en = int(en[latin_mask & fired_en].sum())
print(f"  English Latin mass on shared : {en_mass_both:>14,}  ({100*en_mass_both/total_latin_en:.3f}% of English Latin)")
print(f"  English Latin mass on en-only: {en_mass_en_only:>14,}  ({100*en_mass_en_only/total_latin_en:.3f}% of English Latin)")

with open(TABLES / "en_de_overlap_summary.tsv", "w") as f:
    f.write("metric\tvalue\n")
    f.write(f"latin_fired_en\t{len(latin_fired_en_set)}\n")
    f.write(f"latin_fired_de\t{len(latin_fired_de_set)}\n")
    f.write(f"latin_fired_both\t{len(both)}\n")
    f.write(f"latin_fired_de_only\t{len(de_only)}\n")
    f.write(f"latin_fired_en_only\t{len(en_only)}\n")
    f.write(f"jaccard\t{len(both)/len(union):.6f}\n")
    f.write(f"de_mass_both\t{de_mass_both}\n")
    f.write(f"de_mass_de_only\t{de_mass_de_only}\n")
    f.write(f"de_mass_total_latin\t{total_latin_de}\n")
    f.write(f"de_mass_on_shared_pct\t{100*de_mass_both/total_latin_de:.4f}\n")
    f.write(f"en_mass_both\t{en_mass_both}\n")
    f.write(f"en_mass_en_only\t{en_mass_en_only}\n")
    f.write(f"en_mass_total_latin\t{total_latin_en}\n")
    f.write(f"en_mass_on_shared_pct\t{100*en_mass_both/total_latin_en:.4f}\n")

# ── overlap as a function of count threshold (German side) ────────────────
print("\n=== Mass on shared tokens at German count >= threshold ===")
print(f"  {'thresh':>10} {'de tokens':>10} {'de mass':>14} {'shared pct':>12} {'en mass':>14}")
overlap_rows = []
for t in [1, 5, 10, 100, 1_000, 10_000, 100_000, 1_000_000]:
    de_mask = (fam == "family:Latn") & (de >= t)
    de_idx_t = np.where(de_mask)[0]
    de_n = de_idx_t.size
    de_m = int(de[de_mask].sum())
    en_m_on_those = int(en[de_idx_t].sum())
    shared = sum(1 for i in de_idx_t if fired_en[i])
    print(f"  {t:>10,} {de_n:>10,} {de_m:>14,} {100*shared/max(de_n,1):>11.2f}% {en_m_on_those:>14,}")
    overlap_rows.append({"de_count_threshold": t, "de_tokens": de_n,
                         "de_mass": de_m, "shared_with_en_pct": 100*shared/max(de_n,1),
                         "en_mass_on_these": en_m_on_those})
with open(TABLES / "en_de_overlap_by_count.tsv", "w") as f:
    f.write("de_count_threshold\tde_tokens\tde_mass\tshared_with_en_pct\ten_mass_on_these\n")
    for r in overlap_rows:
        f.write(f"{r['de_count_threshold']}\t{r['de_tokens']}\t{r['de_mass']}\t"
                f"{r['shared_with_en_pct']:.4f}\t{r['en_mass_on_these']}\n")

# ── German-distinctive: de bit set AND en bit not set (ä/ö/ü/ß-bearing) ───
de_cap = np.array([(bm_and[i] >> DE_BIT) & 1 for i in range(V)], dtype=bool)
en_cap = np.array([(bm_and[i] >> EN_BIT) & 1 for i in range(V)], dtype=bool)
de_dist = de_cap & ~en_cap   # contains a letter not in English ASCII
de_dist_fired_idx = np.where(de_dist & fired_de)[0]
de_dist_fired_idx = de_dist_fired_idx[np.argsort(-de[de_dist_fired_idx])]
total_de_dist_mass = int(de[de_dist & fired_de].sum())
print(f"\nGerman-distinctive (de-cap AND NOT en-cap): "
      f"{int((de_dist & fired_de).sum()):,} tokens, mass {total_de_dist_mass:,} "
      f"({100*total_de_dist_mass/total_de:.3f}% of German, "
      f"{100*total_de_dist_mass/total_latin_de:.3f}% of German Latin)")
with open(TABLES / "german_distinctive_top200.tsv", "w") as f:
    f.write("rank\ttoken_id\tdecoded\tcount_de\tcount_en\tpopcount\n")
    for r, i in enumerate(de_dist_fired_idx[:200], 1):
        f.write(f"{r}\t{i}\t{decoded[i]!r}\t{int(de[i])}\t{int(en[i])}\t{popcount[i]}\n")

# ── tokens that fired in German but NOT in English (Latin family) ─────────
de_only_idx_sorted = sorted(de_only, key=lambda i: -de[i])[:200]
with open(TABLES / "german_only_top200.tsv", "w") as f:
    f.write("rank\ttoken_id\tdecoded\tcount_de\tpopcount\n")
    for r, i in enumerate(de_only_idx_sorted, 1):
        f.write(f"{r}\t{i}\t{decoded[i]!r}\t{int(de[i])}\t{popcount[i]}\n")

# ── tokens fired in English but NOT in German (Latin family) ──────────────
en_only_idx_sorted = sorted(en_only, key=lambda i: -en[i])[:200]
with open(TABLES / "english_only_top200.tsv", "w") as f:
    f.write("rank\ttoken_id\tdecoded\tcount_en\tpopcount\n")
    for r, i in enumerate(en_only_idx_sorted, 1):
        f.write(f"{r}\t{i}\t{decoded[i]!r}\t{int(en[i])}\t{popcount[i]}\n")

# ── joint top-50 scatter for inspection ───────────────────────────────────
union_top = list(set(np.argsort(-de)[:50].tolist()) | set(np.argsort(-en)[:50].tolist()))
union_top.sort(key=lambda i: -(de[i] + en[i]))
with open(TABLES / "joint_top50_scatter.tsv", "w") as f:
    f.write("token_id\tdecoded\tcount_de\tcount_en\tratio_de_over_en\tpopcount\tfamily\n")
    for i in union_top:
        rat = de[i] / max(en[i], 1)
        f.write(f"{i}\t{decoded[i]!r}\t{int(de[i])}\t{int(en[i])}\t{rat:.3f}\t"
                f"{popcount[i]}\t{fam[i]}\n")

# ── summary.json ───────────────────────────────────────────────────────────
summary = {
    "lang": "deu_Latn",
    "total_de_sample": total_de,
    "total_en_sample": total_en,
    "fired_total_de": int(fired_de.sum()),
    "fired_total_en": int(fired_en.sum()),
    "families_de": rows,
    "latin_family_de": {
        "fired": int(N_lat),
        "mass": int(total_latin_de),
        "mass_pct": 100 * total_latin_de / total_de,
    },
    "overlap_latin": {
        "fired_in_en": len(latin_fired_en_set),
        "fired_in_de": len(latin_fired_de_set),
        "fired_in_both": len(both),
        "fired_in_de_only": len(de_only),
        "fired_in_en_only": len(en_only),
        "jaccard": len(both) / len(union),
        "de_mass_on_shared": de_mass_both,
        "de_mass_on_shared_pct_of_de_latin": 100 * de_mass_both / total_latin_de,
        "en_mass_on_shared": en_mass_both,
        "en_mass_on_shared_pct_of_en_latin": 100 * en_mass_both / total_latin_en,
    },
    "german_distinctive": {
        "tokens_fired": int((de_dist & fired_de).sum()),
        "mass": total_de_dist_mass,
        "mass_pct_of_de_total": 100 * total_de_dist_mass / total_de,
        "mass_pct_of_de_latin": 100 * total_de_dist_mass / total_latin_de,
    },
}
(HERE / "firing_rate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nwrote firing_rate_summary.json")
print("[done]")

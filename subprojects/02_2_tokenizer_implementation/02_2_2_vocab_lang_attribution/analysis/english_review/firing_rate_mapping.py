"""Firing-rate mapping for English (eng_Latn) — corrected framing.

The bitmask is used here only to *exclude* tokens whose chars cannot be
English (different script family). The MAPPING ("this token is an
English token") comes from the firing-rate distribution within the
Latin family in the English dataset.

Per-token script family is derived from 02_2_1_char_language_membership bits:
  - script_family = the unique script in {Latn, Cyrl, Grek, Arab,
    Han (Hans+Hant+Jpan), Hang, Deva, Hebr, Thai, Armn, Geor, Beng,
    Taml, Telu, Knda, Mlym, Gujr, Guru, Mymr} that owns the set bits.
  - "substrate" = popcount==55 (every locale admits).
  - "mixed_scripts" = bits from more than one script family.
  - "unknown_standalone" = partial_utf8 / byte_unmapped / special.
  - "no_script_admits" = decoded text, popcount==0.

Outputs:
  tables/family_firing_summary.tsv     — per-family token types, mass,
                                          mass% of eng_Latn sample.
  tables/latin_rank_frequency.tsv      — rank-frequency curve of the
                                          Latin family fired in English.
  tables/latin_cumulative_mass.tsv     — cumulative mass at quantile and
                                          natural-break cutoffs.
  tables/family_top20_*.tsv            — top 20 tokens per family for
                                          inspection.
  firing_rate_summary.json             — single-glance numbers.
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
ENG = ck.index("eng_Latn")
eng = H[ENG]
total = int(eng.sum())
V = H.shape[1]

tok = pl.read_parquet(VLA / "outputs/token_metadata.parquet")
decoded = tok["decoded_string"].to_list()

clm = pl.read_parquet(CLM / "artifacts/token_language_bitmask.parquet")
def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
bm_and = [to_int(b) for b in clm["bitmask_and"].to_list()]
status = clm["status"].to_list()
manifest = json.load(open(CLM / "artifacts/manifest.json"))

# ── script-family masks ────────────────────────────────────────────────────
# Hans + Hant + Jpan share Han characters; collapse to "Han"
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

ALL_SCRIPT_NAMES = sorted(script_mask.keys())
N_BITS = 55

popcount = np.array([bin(bm_and[i]).count("1") for i in range(V)], dtype=np.int32)
unknown_statuses = {"partial_utf8", "byte_unmapped", "special"}

def family(i):
    s = status[i]
    if s in unknown_statuses:
        return "unknown_standalone"
    m = bm_and[i]
    if popcount[i] == N_BITS:
        return "substrate"
    if m == 0:
        return "no_script_admits"
    scripts = [name for name, mask in script_mask.items() if (m & mask)]
    if len(scripts) == 1:
        return f"family:{scripts[0]}"
    # multi-script: e.g. Latin + Greek (mixed-script tokens)
    return "mixed_scripts"

fam = np.array([family(i) for i in range(V)])
fired = eng > 0

# ── per-family summary ─────────────────────────────────────────────────────
rows = []
for f in sorted(set(fam.tolist())):
    in_fam = (fam == f)
    in_fam_fired = in_fam & fired
    rows.append({
        "family":       f,
        "vocab_total":  int(in_fam.sum()),
        "fired":        int(in_fam_fired.sum()),
        "mass":         int(eng[in_fam_fired].sum()),
        "mass_pct":     100 * int(eng[in_fam_fired].sum()) / total,
    })
rows.sort(key=lambda r: -r["mass"])
df = pl.DataFrame(rows)
df.write_csv(TABLES / "family_firing_summary.tsv", separator="\t")
print("=== Per-family firing summary (eng_Latn) ===")
print(df)

# ── top tokens per family ──────────────────────────────────────────────────
for f in sorted(set(fam.tolist())):
    in_fam_fired_idx = np.where((fam == f) & fired)[0]
    if in_fam_fired_idx.size == 0:
        continue
    top = in_fam_fired_idx[np.argsort(-eng[in_fam_fired_idx])][:20]
    safe = f.replace(":", "_").replace(" ", "_")
    with open(TABLES / f"family_top20_{safe}.tsv", "w") as out:
        out.write("rank\ttoken_id\tdecoded\tcount\tmass_pct\tpopcount\n")
        for r, i in enumerate(top, 1):
            out.write(f"{r}\t{i}\t{decoded[i]!r}\t{int(eng[i])}\t"
                      f"{100*float(eng[i])/total:.4f}\t{popcount[i]}\n")

# ── Latin-family: rank-frequency curve + cumulative mass ──────────────────
latin_fired_idx = np.where((fam == "family:Latn") & fired)[0]
latin_counts = eng[latin_fired_idx]
sort_order = np.argsort(-latin_counts)
latin_ranked_idx = latin_fired_idx[sort_order]
latin_ranked_counts = latin_counts[sort_order]
N_lat = latin_ranked_counts.size
total_latin = int(latin_ranked_counts.sum())

# cumulative mass at each rank
cum = np.cumsum(latin_ranked_counts)
with open(TABLES / "latin_rank_frequency.tsv", "w") as f:
    f.write("rank\ttoken_id\tdecoded\tcount\tcum_count\tcum_pct_of_latin\tcum_pct_of_total\n")
    for r in range(N_lat):
        i = latin_ranked_idx[r]
        f.write(f"{r+1}\t{i}\t{decoded[i]!r}\t{int(latin_ranked_counts[r])}\t"
                f"{int(cum[r])}\t{100*cum[r]/total_latin:.4f}\t"
                f"{100*cum[r]/total:.4f}\n")

# cumulative mass at quantile / natural-break cutoffs
print("\n=== Latin family — cumulative mass at rank cutoffs ===")
cutoffs_rank = [10, 100, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 50_000, N_lat]
rows_cut = []
print(f"{'top-N':>10} {'last count':>12} {'cum_mass':>14} {'pct_of_latin':>12} {'pct_of_total':>12}")
for k in cutoffs_rank:
    if k > N_lat: break
    cm = int(cum[k-1])
    last_count = int(latin_ranked_counts[k-1])
    print(f"{k:>10,} {last_count:>12,} {cm:>14,} {100*cm/total_latin:>11.3f}% {100*cm/total:>11.3f}%")
    rows_cut.append({"top_n_rank": k, "last_count": last_count, "cum_mass": cm,
                     "pct_of_latin_mass": 100*cm/total_latin,
                     "pct_of_total_eng": 100*cm/total})

# cumulative mass at count cutoffs (firing-count threshold rather than rank)
print("\n=== Latin family — token-types and mass at count >= threshold ===")
thresholds = [1, 5, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
print(f"{'count >=':>10} {'tokens':>10} {'cum_mass':>14} {'pct_of_latin':>12} {'pct_of_total':>12}")
rows_cnt = []
for t in thresholds:
    mask = latin_counts >= t
    n = int(mask.sum())
    m = int(latin_counts[mask].sum())
    print(f"{t:>10,} {n:>10,} {m:>14,} {100*m/total_latin:>11.3f}% {100*m/total:>11.3f}%")
    rows_cnt.append({"count_threshold": t, "tokens_at_or_above": n,
                     "cum_mass": m, "pct_of_latin_mass": 100*m/total_latin,
                     "pct_of_total_eng": 100*m/total})

with open(TABLES / "latin_cumulative_mass.tsv", "w") as f:
    f.write("cutoff_kind\tparam\ttokens\tcum_mass\tpct_of_latin\tpct_of_total\n")
    for r in rows_cut:
        f.write(f"rank\t{r['top_n_rank']}\t{r['top_n_rank']}\t{r['cum_mass']}\t"
                f"{r['pct_of_latin_mass']:.4f}\t{r['pct_of_total_eng']:.4f}\n")
    for r in rows_cnt:
        f.write(f"count\t{r['count_threshold']}\t{r['tokens_at_or_above']}\t"
                f"{r['cum_mass']}\t{r['pct_of_latin_mass']:.4f}\t"
                f"{r['pct_of_total_eng']:.4f}\n")

# ── natural break: log-linear knee of rank-frequency ───────────────────────
# Identify the rank where the slope of log-count vs log-rank changes
# meaningfully. We use the simple max-distance-from-chord heuristic on
# the log-log curve.
log_r = np.log10(np.arange(1, N_lat + 1))
log_c = np.log10(np.maximum(latin_ranked_counts.astype(np.float64), 1.0))
chord = log_c[0] + (log_c[-1] - log_c[0]) * (log_r - log_r[0]) / (log_r[-1] - log_r[0])
dist = chord - log_c
knee_rank = int(np.argmax(dist)) + 1
print(f"\nLog-log chord knee — rank {knee_rank:,}, count {int(latin_ranked_counts[knee_rank-1]):,}, "
      f"cum mass {int(cum[knee_rank-1]):,} ({100*cum[knee_rank-1]/total_latin:.2f}% of Latin, "
      f"{100*cum[knee_rank-1]/total:.2f}% of English)")

# Where does mass reach 95% / 99% / 99.9% of Latin?
print("\n=== Mass quantiles (Latin family) ===")
for q in [0.50, 0.80, 0.90, 0.95, 0.99, 0.999]:
    k = int(np.searchsorted(cum, q * total_latin)) + 1
    print(f"  {100*q:>5.1f}% of Latin mass reached at rank {k:>7,}  "
          f"(last count {int(latin_ranked_counts[k-1]):,})")

# ── summary.json ───────────────────────────────────────────────────────────
fam_sum = {row["family"]: row for row in rows}
summary = {
    "lang": "eng_Latn",
    "total_sample": total,
    "fired_total": int(fired.sum()),
    "families": rows,
    "latin_family": {
        "fired": int(N_lat),
        "mass": int(total_latin),
        "mass_pct": 100 * total_latin / total,
        "log_log_knee_rank": knee_rank,
        "log_log_knee_count": int(latin_ranked_counts[knee_rank-1]),
        "log_log_knee_cum_mass_pct_of_latin": 100 * cum[knee_rank-1] / total_latin,
        "log_log_knee_cum_mass_pct_of_total": 100 * cum[knee_rank-1] / total,
    },
}
(HERE / "firing_rate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nwrote firing_rate_summary.json")
print("[done]")

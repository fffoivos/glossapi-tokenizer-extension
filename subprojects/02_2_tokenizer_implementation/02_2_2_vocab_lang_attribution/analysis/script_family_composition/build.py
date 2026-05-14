"""Script-family composition per dataset.

For each canonical language L (1,933 of them), classify every token in L's
firing histogram into a script-family bucket using the
02_2_1_char_language_membership bitmask. Then sum mass per bucket per L.

Buckets are mutually exclusive, chosen via:

  status == partial_utf8  → partial_utf8             (standalone token is not decodable)
  status == special/...   → non_text                 (not a decoded text token)
  popcount(bm_and) == 55  → substrate                (compatible with all locales)
  popcount(bm_and) == 0   → no_modeled_script        (decoded chars not admitted by any modeled locale)
  bits limited to script X → that script's name      (e.g. "Latn", "Cyrl", "Grek", ...)
  bits span >1 script (rare) → "mixed_scripts"

Outputs:
  script_composition_mass.tsv      — wide table, rows = canonical_key, cols = buckets, values = token mass
  script_composition_pct.tsv       — same with row-normalized percentages
  per_token_script_bucket.tsv      — debugging: every vocab token's script bucket
  plots/heatmap_top50.png          — top-50 langs (by sample size) × buckets heatmap
  plots/heatmap_all_grouped.png    — all 1,933 langs grouped by their canonical script
  summary.json                     — quick lookup, plus methodology notes
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent             # 02_2_2_vocab_lang_attribution/
CLM  = ROOT.parent / "02_2_1_char_language_membership"
PLOTS = HERE / "plots"; PLOTS.mkdir(exist_ok=True)

# ── Bit → script mapping ───────────────────────────────────────────────────
manifest = json.load(open(CLM / "artifacts/manifest.json"))
bit_to_script: dict[int, str] = {L["bit"]: L["script"] for L in manifest["languages"]}
bit_to_code:   dict[int, str] = {L["bit"]: L["code"]   for L in manifest["languages"]}
N_BITS = len(bit_to_script)
# Script → list of bits
script_to_bits = defaultdict(list)
for b, s in bit_to_script.items():
    script_to_bits[s].append(b)
# Bucket order — keep visualizations stable
BUCKETS = [
    "Latn", "Cyrl", "Grek", "Arab",
    "CJK_Han",          # Hans + Hant + Jpan share Han chars; this bucket captures pure-Han
    "Jpan_kana",        # Jpan-only bits (hiragana/katakana)
    "Hang",
    "Deva", "Hebr", "Thai", "Armn", "Geor",
    "Beng", "Taml", "Telu", "Knda", "Mlym", "Gujr", "Guru", "Mymr",
    "substrate", "partial_utf8", "non_text", "mixed_scripts", "no_modeled_script",
]
CJK_HAN_SET = {"Hans", "Hant", "Jpan"}
B2I = {b: i for i, b in enumerate(BUCKETS)}

# Precompute bitmask of each script
script_mask: dict[str, int] = {}
for s, bits in script_to_bits.items():
    m = 0
    for b in bits: m |= (1 << b)
    script_mask[s] = m
ALL_MASK = (1 << N_BITS) - 1

print(f"Loaded {N_BITS} bits → {len(script_mask)} scripts: {sorted(script_mask.keys())}")


# ── Load histogram + bitmask ───────────────────────────────────────────────
print("loading histogram_matrix + bitmask…")
z = np.load(ROOT / "outputs/histogram_matrix.npz", allow_pickle=True)
H = z["H"].astype(np.int64)
canonical_keys = list(z["canonical_keys"])
N_LANG, V = H.shape
print(f"H: {H.shape}")

clm = pl.read_parquet(CLM / "artifacts/token_language_bitmask.parquet")
def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
bm_and = np.array([to_int(b) for b in clm["bitmask_and"].to_list()], dtype=np.int64)
status = clm["status"].to_list()
# Use python int for safety because Python ints are arbitrary precision,
# but for popcount on int64 we're fine since we only use 55 bits.

# ── Categorize each token into one bucket ──────────────────────────────────
print("categorizing tokens…")
token_bucket = np.empty(V, dtype="<U24")
popcount = np.array([bin(int(x)).count("1") for x in bm_and], dtype=np.int32)
for t in range(V):
    st = status[t]
    if st == "partial_utf8":
        token_bucket[t] = "partial_utf8"
        continue
    if st in {"special", "byte_unmapped"}:
        token_bucket[t] = "non_text"
        continue

    pc = popcount[t]
    if pc == N_BITS:
        token_bucket[t] = "substrate"
        continue
    if pc == 0:
        token_bucket[t] = "no_modeled_script"
        continue
    m = int(bm_and[t])
    # check which scripts have any bit set
    scripts_present = {s for s, sm in script_mask.items() if (m & sm) != 0}
    if len(scripts_present) == 1:
        s = next(iter(scripts_present))
        if s == "Jpan":
            token_bucket[t] = "Jpan_kana"   # Jpan-only means kana (no Han)
        else:
            token_bucket[t] = s
        continue
    # Han-family: subset of {Hans, Hant, Jpan} — common case for shared Han chars
    if scripts_present <= CJK_HAN_SET:
        token_bucket[t] = "CJK_Han"
        continue
    token_bucket[t] = "mixed_scripts"

# Sanity check distribution
from collections import Counter
buckets_count = Counter(token_bucket.tolist())
print("Per-vocab bucket counts:")
for b in BUCKETS:
    print(f"  {b:<22}: {buckets_count.get(b, 0):>7,}")

# ── Compute mass per (lang, bucket) ────────────────────────────────────────
print("computing mass per (lang, bucket)…")
mass = np.zeros((N_LANG, len(BUCKETS)), dtype=np.int64)
for i, b in enumerate(BUCKETS):
    cat_mask = token_bucket == b
    if cat_mask.any():
        mass[:, i] = H[:, cat_mask].sum(axis=1)

# Sample sizes per lang
sample_sizes = mass.sum(axis=1)
print(f"langs with non-zero sample: {(sample_sizes > 0).sum():,}")

# ── Write tables ───────────────────────────────────────────────────────────
print("writing TSVs…")
# Wide table (mass)
with open(HERE / "script_composition_mass.tsv", "w") as f:
    f.write("canonical_key\tsample_total\t" + "\t".join(BUCKETS) + "\n")
    for j, k in enumerate(canonical_keys):
        f.write(f"{k}\t{int(sample_sizes[j])}\t" +
                "\t".join(str(int(mass[j, i])) for i in range(len(BUCKETS))) + "\n")

# Wide table (percentages)
with open(HERE / "script_composition_pct.tsv", "w") as f:
    f.write("canonical_key\tsample_total\t" + "\t".join(BUCKETS) + "\n")
    for j, k in enumerate(canonical_keys):
        s = sample_sizes[j]
        if s == 0:
            row = ["0"] * len(BUCKETS)
        else:
            row = [f"{100*mass[j, i]/s:.4f}" for i in range(len(BUCKETS))]
        f.write(f"{k}\t{int(s)}\t" + "\t".join(row) + "\n")

# Per-token bucket assignment (debugging)
tok = pl.read_parquet(ROOT / "outputs/token_metadata.parquet")
decoded = tok["decoded_string"].to_list()
with open(HERE / "per_token_script_bucket.tsv", "w") as f:
    f.write("token_id\tdecoded\tpopcount_and\tscript_bucket\tbitmask_and_int\n")
    for t in range(V):
        f.write(f"{t}\t{decoded[t]!r}\t{popcount[t]}\t{token_bucket[t]}\t{int(bm_and[t])}\n")

# ── Plots ─────────────────────────────────────────────────────────────────
print("plots…")
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.family": "DejaVu Sans"})

# Heatmap top-50 langs by sample size
order = np.argsort(-sample_sizes)
top50 = order[:50]
M50 = mass[top50] / np.maximum(sample_sizes[top50, None], 1)  # row-normalized
fig, ax = plt.subplots(figsize=(13, 13))
im = ax.imshow(M50, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_yticks(range(len(top50)))
ax.set_yticklabels([canonical_keys[j] for j in top50], fontsize=10)
ax.set_xticks(range(len(BUCKETS)))
ax.set_xticklabels(BUCKETS, rotation=45, ha="right", fontsize=10)
ax.set_title("Script-family composition · top 50 canonical languages by sample size (row-normalized)")
for i in range(M50.shape[0]):
    for j in range(M50.shape[1]):
        v = M50[i, j]
        if v >= 0.01:
            color = "white" if v > 0.4 else "black"
            ax.text(j, i, f"{100*v:.0f}", ha="center", va="center",
                    fontsize=8, color=color)
fig.colorbar(im, ax=ax, label="fraction of dataset mass", fraction=0.025, pad=0.01)
fig.tight_layout()
fig.savefig(PLOTS / "heatmap_top50.png", dpi=130)
plt.close(fig)

# Heatmap: all langs grouped by canonical script
lang_meta = json.load(open(ROOT / "outputs/lang_metadata.json"))
canon_script = np.array([lang_meta[k].get("script_iso15924", "?") for k in canonical_keys])
# Sort by (canonical script, then sample size desc)
sort_keys = list(zip(canon_script, -sample_sizes, np.arange(N_LANG)))
sort_idx = [t[2] for t in sorted(sort_keys, key=lambda x: (x[0], x[1]))]
# Filter to langs with non-zero samples
sort_idx = [j for j in sort_idx if sample_sizes[j] > 0]

# To make this readable, we'll only plot one representative bar per
# canonical script, plus the rest aggregated as a script-family-mean bar.
# Actually, summary by script family:
script_summary = defaultdict(lambda: np.zeros(len(BUCKETS), dtype=np.int64))
script_lang_count = defaultdict(int)
for j in range(N_LANG):
    s = canon_script[j]
    if sample_sizes[j] > 0:
        script_summary[s] += mass[j]
        script_lang_count[s] += 1

# Build a summary heatmap: rows = canonical script (sorted by total mass), cols = buckets
script_order = sorted(script_summary.keys(), key=lambda s: -script_summary[s].sum())
script_mass_array = np.stack([script_summary[s] for s in script_order])
script_pct = script_mass_array / np.maximum(script_mass_array.sum(axis=1, keepdims=True), 1)

fig, ax = plt.subplots(figsize=(14, max(6, 0.32 * len(script_order))))
im = ax.imshow(script_pct, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_yticks(range(len(script_order)))
ax.set_yticklabels([f"{s}  ({script_lang_count[s]} langs · {script_mass_array[i].sum()/1e9:.2f} B)"
                    for i, s in enumerate(script_order)], fontsize=10)
ax.set_xticks(range(len(BUCKETS)))
ax.set_xticklabels(BUCKETS, rotation=45, ha="right", fontsize=10)
ax.set_title("Script-family composition aggregated by canonical-key script\n"
             "(each row = sum of all canonical keys with that script ISO 15924 tag)")
for i in range(script_pct.shape[0]):
    for j in range(script_pct.shape[1]):
        v = script_pct[i, j]
        if v >= 0.01:
            color = "white" if v > 0.4 else "black"
            ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=8, color=color)
fig.colorbar(im, ax=ax, label="fraction of group mass", fraction=0.02, pad=0.01)
fig.tight_layout()
fig.savefig(PLOTS / "heatmap_by_canonical_script.png", dpi=130)
plt.close(fig)

# ── Summary ───────────────────────────────────────────────────────────────
summary = {
    "n_langs": int(N_LANG),
    "n_langs_with_data": int((sample_sizes > 0).sum()),
    "n_buckets": len(BUCKETS),
    "buckets": BUCKETS,
    "per_token_bucket_counts": {b: int(buckets_count.get(b, 0)) for b in BUCKETS},
    "by_canonical_script": {
        s: {
            "n_langs": script_lang_count[s],
            "total_mass": int(script_summary[s].sum()),
            "pct_per_bucket": {
                BUCKETS[i]: float(script_pct[script_order.index(s)][i])
                for i in range(len(BUCKETS))
            },
        } for s in script_order
    },
    "outputs": {
        "wide_mass":  "script_composition_mass.tsv",
        "wide_pct":   "script_composition_pct.tsv",
        "per_token":  "per_token_script_bucket.tsv",
        "heatmap_top50":              "plots/heatmap_top50.png",
        "heatmap_by_canonical_script":"plots/heatmap_by_canonical_script.png",
    },
    "methodology": (
        "Bucket assignment per token: popcount(bitmask_and)=55 → substrate; "
        "partial_utf8 and special/byte_unmapped tokens are separated before "
        "bitmask interpretation because their zero masks mean unknown/non-text, "
        "not language impossibility; popcount=0 for decoded text → no_modeled_script; "
        "bits limited to a single script → that script's ISO 15924 name; "
        "bits spanning multiple scripts → mixed_scripts. "
        "Per-language mass aggregated by token-bucket via the (N_LANG×V) histogram_matrix."
    ),
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print(f"\nWrote outputs under {HERE}")
for k, v in summary["outputs"].items():
    print(f"  {k}: {v}")
print(f"\nQuick sanity — top 5 buckets per canonical script:")
for s in script_order[:8]:
    pct = summary["by_canonical_script"][s]["pct_per_bucket"]
    top5 = sorted(pct.items(), key=lambda x: -x[1])[:5]
    breakdown = " · ".join(f"{b}={100*v:.1f}%" for b, v in top5)
    print(f"  {s:<5} (n={script_lang_count[s]:>4}): {breakdown}")

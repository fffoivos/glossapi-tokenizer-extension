"""Methodology review — using Greek as the validation case for a general
"which tokens belong to language L" classifier.

What we want to know:
  1. Frequency plot of Greek tokens (Zipf with annotations).
  2. What ELSE fires in the Greek dataset, and where do those tokens "belong"
     by rate (i.e., where are they overrepresented).
  3. Cross-script overlap: when a token has script S, how is its count
     distributed across languages of all scripts?

Methodological candidate: per-token "specificity" — the fraction of a token's
total per-language-rate that goes to a single language.

  rate[L,t]        = count[L,t] / sample_size[L]
  specificity[L,t] = rate[L,t] / Σ_L' rate[L',t]

Validation:
  Ground truth for Greek = tokens containing ≥1 Greek codepoint (1,507).
  Sweep specificity threshold; report precision / recall vs ground truth.

Outputs:
  plots/06_greek_zipf_annotated.png
  plots/07_what_fires_in_greek.png
  plots/08_script_overlap_matrix.png
  plots/09_specificity_dist.png
  plots/10_specificity_pr_curve.png
  plots/11_method_per_language.png
  tables/non_greek_script_in_greek.tsv
  tables/script_overlap.tsv
  tables/specificity_pr.tsv
  tables/method_per_language.tsv
  methodology_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # 02_2_2_vocab_lang_attribution/
OUT = HERE
(OUT / "plots").mkdir(exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)


# --- Theme ---
plt.rcParams.update({
    "figure.facecolor": "#0f0e0b",
    "axes.facecolor":   "#0f0e0b",
    "axes.edgecolor":   "#5a5346",
    "axes.labelcolor":  "#ece2cd",
    "axes.titlecolor":  "#ece2cd",
    "xtick.color":      "#a9a294",
    "ytick.color":      "#a9a294",
    "text.color":       "#ece2cd",
    "grid.color":       "#2a2620",
    "grid.alpha":       0.6,
    "font.family":      "serif",
    "font.serif":       ["EB Garamond", "Georgia", "DejaVu Serif"],
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "normal",
})
CRIMSON = "#c44536"
GOLD    = "#b89968"
PATINA  = "#486f87"
CREAM   = "#ece2cd"


def main():
    print("[methodology] loading …")
    z = np.load(ROOT / "outputs/histogram_matrix.npz", allow_pickle=True)
    H = z["H"].astype(np.int64)            # (N_lang, V)
    ck = list(z["canonical_keys"])
    N_LANG, V = H.shape
    print(f"  H: {H.shape}")

    lang_meta = json.load(open(ROOT / "outputs/lang_metadata.json"))
    tok       = pl.read_parquet(ROOT / "outputs/token_metadata.parquet")

    sample_sizes = np.array([lang_meta[k]["sample_tokens_total"] for k in ck], dtype=np.float64)
    valid_lang = sample_sizes > 1_000_000          # ≥ 1 M tokens to compute meaningful rates
    print(f"  langs with sample ≥ 1 M tokens: {int(valid_lang.sum())} / {N_LANG}")

    scripts = np.array([lang_meta[k]["script_iso15924"] or "?" for k in ck])
    iso639  = np.array([lang_meta[k]["iso_639_3"] or "?" for k in ck])
    name    = np.array([lang_meta[k]["name"] or "?" for k in ck])

    # Per-language rates (count per token of language sample)
    safe_ss = np.where(sample_sizes > 0, sample_sizes, 1.0)
    rates = H / safe_ss[:, None]                     # (N_lang, V) float
    rates[~valid_lang] = 0.0                          # mask noisy small langs

    # Per-token specificity
    col_sum = rates.sum(axis=0)                      # (V,)
    safe_col = np.where(col_sum > 0, col_sum, 1.0)
    spec = rates / safe_col[None, :]                 # (N_lang, V)

    # Greek-related
    ELL = ck.index("ell_Grek")
    greek_counts = H[ELL]
    greek_total = int(greek_counts.sum())
    greek_spec = spec[ELL]

    tok = tok.with_columns(
        (pl.col("has_greek_mono") | pl.col("has_greek_poly")).alias("is_greek_script")
    )
    is_gs = tok["is_greek_script"].to_numpy()
    decoded = tok["decoded_string"].to_list()
    has_latin   = (tok["has_latin_basic"] | tok["has_latin_extended"]).to_numpy()
    has_cyr     = tok["has_cyrillic"].to_numpy()
    has_han     = tok["has_han"].to_numpy()
    has_arab    = tok["has_arabic"].to_numpy()
    has_hebr    = tok["has_hebrew"].to_numpy()
    has_dev     = tok["has_devanagari"].to_numpy()
    has_hangul  = tok["has_hangul"].to_numpy()
    has_thai    = tok["has_thai"].to_numpy()
    is_struct   = tok["is_structural_only"].to_numpy()
    is_digit    = tok["is_pure_digits"].to_numpy()
    is_special  = tok["is_special"].to_numpy()
    is_byte     = tok["is_byte_fragment"].to_numpy()

    # Primary language by specificity (argmax)
    primary_lang_idx = spec.argmax(axis=0)
    primary_lang_key = np.array(ck)[primary_lang_idx]

    # ====================================================================
    # PLOT 6 — Greek Zipf with script annotations
    # ====================================================================
    print("[methodology] plot 06 — annotated Greek Zipf")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    order = np.argsort(-greek_counts)
    counts_sorted = greek_counts[order]
    gs_sorted = is_gs[order]
    lat_sorted = has_latin[order] & ~gs_sorted
    struct_sorted = is_struct[order] & ~gs_sorted & ~lat_sorted
    other_sorted = ~(gs_sorted | lat_sorted | struct_sorted)

    nz = counts_sorted > 0
    rank = np.arange(1, V + 1)
    # plot each script category as a separate scatter for legibility
    ax.scatter(rank[gs_sorted & nz], counts_sorted[gs_sorted & nz], s=2, c=CRIMSON,
               label=f"Greek-script ({int((is_gs & (greek_counts>0)).sum()):,})", alpha=0.85)
    ax.scatter(rank[struct_sorted & nz], counts_sorted[struct_sorted & nz], s=2, c=PATINA,
               label=f"Structural (P/S/Z) ({int((is_struct & ~is_gs & (greek_counts>0)).sum()):,})", alpha=0.7)
    ax.scatter(rank[lat_sorted & nz], counts_sorted[lat_sorted & nz], s=2, c=GOLD,
               label=f"Latin letters ({int((has_latin & ~is_gs & (greek_counts>0)).sum()):,})", alpha=0.5)
    ax.scatter(rank[other_sorted & nz], counts_sorted[other_sorted & nz], s=2, c="#7a7466",
               label=f"Other ({int((other_sorted & nz).sum()):,})", alpha=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rank by Greek count"); ax.set_ylabel("count in 1 B Greek sample")
    ax.set_title(f"Greek-token frequency · {int((greek_counts>0).sum()):,} of 131,072 entries fired · sample = {greek_total/1e9:.2f} B")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "plots/06_greek_zipf_annotated.png", dpi=130)
    plt.close(fig)

    # ====================================================================
    # PLOT 7 — What else fires in Greek — by primary language (by rate)
    # ====================================================================
    print("[methodology] plot 07 — what else fires in Greek")
    # For non-Greek-script tokens with greek_count >= 100, what's their primary lang?
    sel = (~is_gs) & (greek_counts >= 100)
    pkeys = primary_lang_key[sel]
    pcounts = greek_counts[sel]
    # Aggregate by primary lang
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])  # primary_key -> [n_tokens, sum_greek_count]
    for pk, c in zip(pkeys, pcounts):
        agg[pk][0] += 1
        agg[pk][1] += int(c)
    rows = sorted(agg.items(), key=lambda x: -x[1][1])[:20]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    labels = [r[0] for r in rows]
    masses = [r[1][1] for r in rows]
    n_toks = [r[1][0] for r in rows]
    bars = ax.barh(range(len(labels)), [m / 1e6 for m in masses], color=GOLD, edgecolor="#5a5346")
    for i, (b, n) in enumerate(zip(bars, n_toks)):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2,
                f"{int(b.get_width()):,} M  · {n:,} tokens",
                va="center", fontsize=9, color=CREAM)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{l}" for l in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Greek-sample mass carried by these tokens (millions)")
    ax.set_title("Where do non-Greek-script tokens that fire in Greek 'belong'?\n"
                 "(primary lang = argmax over per-language rates; only tokens with Greek count ≥ 100)")
    fig.tight_layout()
    fig.savefig(OUT / "plots/07_what_fires_in_greek.png", dpi=130)
    plt.close(fig)

    # Table for the same
    with open(OUT / "tables/non_greek_script_in_greek.tsv", "w") as f:
        f.write("primary_lang_by_rate\tn_tokens\tsum_greek_count\texample_decoded_top5\n")
        for k, (n, m) in sorted(agg.items(), key=lambda x: -x[1][1]):
            # find top 5 example tokens for this primary lang
            mask = sel & (primary_lang_key == k)
            order_local = np.argsort(-greek_counts * mask)[:5]
            ex = " | ".join(repr(decoded[i])[:30] for i in order_local if mask[i])
            f.write(f"{k}\t{n}\t{m}\t{ex}\n")

    # ====================================================================
    # PLOT 8 — Script overlap matrix (token script × language script)
    # ====================================================================
    print("[methodology] plot 08 — script overlap matrix")
    # Define token-script (mutually exclusive priority)
    token_script_label = np.full(V, "Other", dtype=object)
    # priority: Greek > Cyrillic > Han > Arabic > Hebrew > Devanagari > Hangul > Thai > Latin > Structural > Digit > Byte > Special > Other
    token_script_label[has_latin] = "Latin"
    token_script_label[has_thai] = "Thai"
    token_script_label[has_hangul] = "Hangul"
    token_script_label[has_dev] = "Devanagari"
    token_script_label[has_hebr] = "Hebrew"
    token_script_label[has_arab] = "Arabic"
    token_script_label[has_han] = "Han"
    token_script_label[has_cyr] = "Cyrillic"
    token_script_label[is_gs] = "Greek"
    token_script_label[is_struct & ~has_latin & ~is_gs & ~has_cyr & ~has_han] = "Structural"
    token_script_label[is_digit] = "Digit"
    token_script_label[is_byte] = "Byte-frag"
    token_script_label[is_special] = "Special"

    # Language-script families: just use the 15924 code as-is for major ones, lump tiny ones
    major_scripts = ["Latn", "Cyrl", "Hani", "Arab", "Grek", "Hebr", "Deva",
                     "Hang", "Thai", "Jpan", "Kana", "Ethi", "Mymr", "Sinh", "Mlym"]
    lang_script_label = np.array([s if s in major_scripts else "Other" for s in scripts])

    token_scripts = ["Greek", "Cyrillic", "Han", "Arabic", "Hebrew", "Devanagari",
                     "Hangul", "Thai", "Latin", "Structural", "Digit", "Byte-frag", "Special", "Other"]
    lang_scripts = ["Grek", "Cyrl", "Hani", "Arab", "Hebr", "Deva",
                    "Hang", "Thai", "Latn", "Jpan", "Kana", "Ethi", "Mymr", "Sinh", "Mlym", "Other"]

    M = np.zeros((len(token_scripts), len(lang_scripts)), dtype=np.float64)
    for i, ts in enumerate(token_scripts):
        tok_mask = (token_script_label == ts)
        if not tok_mask.any():
            continue
        # sum across languages, grouped by lang_script_label
        for j, ls in enumerate(lang_scripts):
            lang_mask = (lang_script_label == ls)
            M[i, j] = float(H[lang_mask][:, tok_mask].sum())
    # Normalize each row to sum 1 to see "given token script S, what's the language-script distribution of its mass?"
    M_row = M / np.where(M.sum(axis=1, keepdims=True) > 0, M.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    norm = mcolors.LogNorm(vmin=max(M_row[M_row > 0].min(), 1e-5), vmax=1.0)
    im = ax.imshow(M_row, cmap="YlOrRd", norm=norm, aspect="auto")
    ax.set_xticks(range(len(lang_scripts))); ax.set_xticklabels(lang_scripts, rotation=45, ha="right")
    ax.set_yticks(range(len(token_scripts))); ax.set_yticklabels(token_scripts)
    ax.set_xlabel("Language script (ISO 15924)")
    ax.set_ylabel("Token script category")
    ax.set_title("Row-normalized: given a token of script S, where does its mass land across language scripts?\n"
                 "(log scale; values are the fraction of that token-script bucket's total count)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.ax.set_ylabel("fraction of row total (log)", color=CREAM)
    # Annotate non-trivial cells
    for i in range(M_row.shape[0]):
        for j in range(M_row.shape[1]):
            v = M_row[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                        color="white" if v > 0.1 else "#ece2cd", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plots/08_script_overlap_matrix.png", dpi=130)
    plt.close(fig)

    # Also save the raw matrix
    with open(OUT / "tables/script_overlap.tsv", "w") as f:
        f.write("token_script\t" + "\t".join(lang_scripts) + "\trow_total\n")
        for i, ts in enumerate(token_scripts):
            f.write(ts + "\t" + "\t".join(f"{int(v)}" for v in M[i]) + f"\t{int(M[i].sum())}\n")

    # ====================================================================
    # PLOT 9 — Greek-specificity histogram (for tokens that fire in Greek)
    # ====================================================================
    print("[methodology] plot 09 — specificity distribution")
    fired = greek_counts > 0
    fired_gs = fired & is_gs
    fired_other = fired & ~is_gs
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(greek_spec[fired_gs], bins=bins, color=CRIMSON, alpha=0.85,
            label=f"Greek-script tokens that fired ({int(fired_gs.sum()):,})", edgecolor="black", linewidth=0.5)
    ax.hist(greek_spec[fired_other], bins=bins, color=GOLD, alpha=0.6,
            label=f"Non-Greek-script tokens that fired ({int(fired_other.sum()):,})", edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Greek specificity = rate_in_Greek / Σ rate_in_any_lang")
    ax.set_ylabel("number of tokens (log)")
    ax.set_title("Greek-specificity distribution\n"
                 "(rate-normalized: removes sample-size bias; > 0.5 = mostly fires in Greek)")
    ax.axvline(0.5, color=PATINA, linestyle="--", linewidth=1)
    ax.text(0.51, ax.get_ylim()[1] * 0.5, "0.5 — argmax threshold", color=PATINA, fontsize=10)
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "plots/09_specificity_dist.png", dpi=130)
    plt.close(fig)

    # ====================================================================
    # PLOT 10 — Precision/recall curve: specificity threshold vs Greek-script GT
    # ====================================================================
    print("[methodology] plot 10 — specificity PR curve")
    thresholds = np.concatenate([np.array([0]), np.logspace(-3, 0, 60)])
    P, R, F1 = [], [], []
    rows = []
    for tau in thresholds:
        pred = greek_spec >= tau                  # predicted-Greek
        tp = int((pred & is_gs).sum())
        fp = int((pred & ~is_gs).sum())
        fn = int((~pred & is_gs).sum())
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        P.append(precision); R.append(recall); F1.append(f1)
        rows.append({"tau": float(tau), "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1,
                     "tokens_predicted_greek": tp + fp})
    pl.DataFrame(rows).write_csv(OUT / "tables/specificity_pr.tsv", separator="\t")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, P, label="precision", color=CRIMSON, lw=2)
    ax.plot(thresholds, R, label="recall",    color=PATINA, lw=2)
    ax.plot(thresholds, F1, label="F1",       color=GOLD,   lw=2, linestyle="--")
    # Pick best F1
    best_i = int(np.argmax(F1))
    ax.axvline(thresholds[best_i], color="white", alpha=0.3, lw=1)
    ax.scatter([thresholds[best_i]], [F1[best_i]], color="white", zorder=5)
    ax.annotate(f"  best F1={F1[best_i]:.3f}\n  τ={thresholds[best_i]:.3f}\n  P={P[best_i]:.3f}, R={R[best_i]:.3f}",
                xy=(thresholds[best_i], F1[best_i]), xytext=(15, -30),
                textcoords="offset points", color="white", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Greek-specificity threshold τ")
    ax.set_ylabel("score vs Greek-script ground truth")
    ax.set_title("Validation: specificity-threshold classifier recovers Greek-script ground truth")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "plots/10_specificity_pr_curve.png", dpi=130)
    plt.close(fig)

    # ====================================================================
    # PLOT 11 — Generalization: does specificity recover the script-ground-truth
    # for OTHER languages too?
    # ====================================================================
    print("[methodology] plot 11 — generalization across languages")
    # For each test language, define its script-ground-truth = tokens with at-least-one codepoint in that language's script.
    # Need a script→token-mask map.
    SCRIPT_FLAGS = {
        "Grek": is_gs,
        "Cyrl": has_cyr,
        "Hani": has_han,
        "Arab": has_arab,
        "Hebr": has_hebr,
        "Deva": has_dev,
        "Hang": has_hangul,
        "Thai": has_thai,
    }
    test_langs = [
        ("ell_Grek", "Greek"),
        ("rus_Cyrl", "Russian"),
        ("ukr_Cyrl", "Ukrainian"),
        ("cmn_Hani", "Mandarin"),
        ("arb_Arab", "Arabic"),
        ("fas_Arab", "Persian"),
        ("heb_Hebr", "Hebrew"),
        ("hin_Deva", "Hindi"),
        ("mar_Deva", "Marathi"),
        ("kor_Hang", "Korean"),
        ("tha_Thai", "Thai"),
    ]
    method_rows = []
    fig, ax = plt.subplots(figsize=(11, 6.5))
    method_taus = np.logspace(-3, 0, 40)
    for ckey, label in test_langs:
        if ckey not in ck: continue
        L = ck.index(ckey)
        script = scripts[L]
        gt = SCRIPT_FLAGS.get(script)
        if gt is None: continue
        lspec = spec[L]
        Ps, Rs, F1s = [], [], []
        for tau in method_taus:
            pred = lspec >= tau
            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            Ps.append(p); Rs.append(r); F1s.append(2 * p * r / max(p + r, 1e-12))
        best = int(np.argmax(F1s))
        method_rows.append({
            "language": ckey, "name": label, "script": script,
            "ground_truth_tokens": int(gt.sum()),
            "best_tau": float(method_taus[best]),
            "best_F1": F1s[best],
            "best_P": Ps[best],
            "best_R": Rs[best],
        })
        ax.plot(method_taus, F1s, lw=1.8, label=f"{label} ({script}) F1={F1s[best]:.2f}")
    pl.DataFrame(method_rows).write_csv(OUT / "tables/method_per_language.tsv", separator="\t")
    ax.set_xscale("log")
    ax.set_xlabel("specificity threshold τ")
    ax.set_ylabel("F1 vs script-of-language ground truth")
    ax.set_title("Does the specificity-threshold classifier generalize?\n"
                 "(F1 of recovering tokens containing the language's script codepoints)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", ncol=2, framealpha=0.5, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "plots/11_method_per_language.png", dpi=130)
    plt.close(fig)

    # ====================================================================
    # Summary JSON
    # ====================================================================
    best_pr = max(rows, key=lambda r: r["f1"])
    summary = {
        "method": "per-language rate-normalized specificity",
        "definition": "specificity[L,t] = (count[L,t] / sample_size[L]) / Σ_L' (count[L',t] / sample_size[L'])",
        "validation_target": "tokens with ≥1 Greek codepoint (1507)",
        "greek_validation": {
            "best_threshold":  best_pr["tau"],
            "F1":              best_pr["f1"],
            "precision":       best_pr["precision"],
            "recall":          best_pr["recall"],
            "tokens_classified_greek_at_best": best_pr["tokens_predicted_greek"],
        },
        "generalization": method_rows,
        "what_else_fires_in_greek_top20_by_primary_lang": [
            {"primary_lang_by_rate": k, "n_tokens": v[0], "greek_mass": v[1]}
            for k, v in sorted(agg.items(), key=lambda x: -x[1][1])[:20]
        ],
        "n_langs_with_valid_sample": int(valid_lang.sum()),
        "n_langs_total": N_LANG,
    }
    (OUT / "methodology_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[methodology] DONE")
    print(f"  best Greek F1 = {best_pr['f1']:.4f} at τ = {best_pr['tau']:.4f} (P={best_pr['precision']:.3f}, R={best_pr['recall']:.3f})")


if __name__ == "__main__":
    main()

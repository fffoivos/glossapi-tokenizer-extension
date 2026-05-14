"""Greek review analysis.

Loads the aggregated artifacts and produces:
  plots/01_greek_zipf.png            — rank vs count log-log for all 131k tokens by their Greek count
  plots/02_greek_cdf.png             — CDF of Greek-token mass, all tokens vs Greek-script-only subset
  plots/03_script_mass_breakdown.png — bar chart of Greek-token mass by script-bucket
  plots/04_greek_script_zipf.png     — same Zipf restricted to Greek-codepoint tokens
  plots/05_greek_script_threshold.png — for thresholds 0/1/10/100/1000/...  | tokens kept, Greek-mass captured, precision (Greek-script ∩ kept) / kept
  tables/top200_overall.tsv          — top 200 tokens by ell_Grek count: id, decoded, count, share, script flags
  tables/top200_greek_script.tsv     — top 200 Greek-script tokens by ell_Grek count
  tables/top200_NOT_greek_script.tsv — top 200 NON-Greek-script tokens by ell_Grek count (the "shared infrastructure" — likely punctuation/whitespace/numerics)
  tables/greek_script_zero_count.tsv — Greek-script tokens that fired zero times in Greek (suspect noise / specialized symbols)
  summary.json                       — all the topline numbers
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)


def main():
    # ---- Load ----
    z = np.load(ROOT / "outputs/histogram_matrix.npz", allow_pickle=True)
    H = z["H"]
    canonical_keys = list(z["canonical_keys"])
    tok = pl.read_parquet(ROOT / "outputs/token_metadata.parquet")
    print(f"H: {H.shape}, vocab: {tok.height}")

    ell_idx = canonical_keys.index("ell_Grek")
    greek = H[ell_idx]                # (131072,) Greek per-token counts
    total_greek = int(greek.sum())
    print(f"ell_Grek total tokens in sample: {total_greek:,}")

    # ---- Script flags ----
    is_greek_script = (tok["has_greek_mono"] | tok["has_greek_poly"]).to_numpy()
    is_greek_mono   = tok["has_greek_mono"].to_numpy()
    is_greek_poly   = tok["has_greek_poly"].to_numpy()
    is_latin        = (tok["has_latin_basic"] | tok["has_latin_extended"]).to_numpy()
    is_cyrillic     = tok["has_cyrillic"].to_numpy()
    is_han          = tok["has_han"].to_numpy()
    is_struct       = tok["is_structural_only"].to_numpy()
    is_digit        = tok["is_pure_digits"].to_numpy()
    is_ws           = tok["is_pure_whitespace"].to_numpy()
    is_special      = tok["is_special"].to_numpy()
    is_byte_frag    = tok["is_byte_fragment"].to_numpy()

    decoded = tok["decoded_string"].to_list()

    fired_mask = greek > 0
    n_fired = int(fired_mask.sum())
    n_geq_10 = int((greek >= 10).sum())
    n_geq_100 = int((greek >= 100).sum())
    n_geq_1000 = int((greek >= 1000).sum())
    print(f"tokens fired in Greek: {n_fired:,} of 131,072 ({100*n_fired/131072:.1f}%)")

    # ---- PLOT 1: rank-count log-log ----
    sorted_counts = np.sort(greek)[::-1]
    nonzero = sorted_counts[sorted_counts > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(np.arange(1, len(nonzero) + 1), nonzero, color="#1f77b4", lw=1.2)
    ax.set_xlabel("Rank (1 = highest count)")
    ax.set_ylabel("Count in Greek sample")
    ax.set_title(f"Greek token-count distribution (Zipf)\n{n_fired:,} of 131,072 vocab entries fire in Greek (1B-token sample)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "plots/01_greek_zipf.png", dpi=130)
    plt.close(fig)

    # ---- PLOT 2: CDF of Greek mass, all vs Greek-script ----
    sort_idx = np.argsort(-greek)
    cum_all = np.cumsum(greek[sort_idx]) / total_greek
    cum_in_gs = np.cumsum(is_greek_script[sort_idx].astype(np.int64) * greek[sort_idx]) / total_greek
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(cum_all) + 1)
    ax.semilogx(x, cum_all, label="All tokens", lw=1.5)
    ax.semilogx(x, cum_in_gs, label="Greek-script tokens only", lw=1.5, color="#d62728")
    ax.set_xlabel("Top-N tokens by Greek count")
    ax.set_ylabel("Cumulative share of total Greek-mass")
    ax.set_title("Cumulative Greek-token mass: all tokens vs Greek-script subset")
    ax.axhline(1.0, color="black", lw=0.5, alpha=0.4)
    final_gs_share = cum_in_gs[-1]
    ax.axhline(final_gs_share, color="#d62728", lw=0.5, ls="--", alpha=0.6)
    ax.text(2, final_gs_share - 0.04, f"Greek-script captures {100*final_gs_share:.2f}% of all Greek mass", color="#d62728", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "plots/02_greek_cdf.png", dpi=130)
    plt.close(fig)

    # ---- PLOT 3: Bar chart of Greek mass by script-bucket ----
    # Mutually-exclusive buckets, top-down precedence
    buckets = []
    used = np.zeros_like(is_greek_script)
    def add_bucket(label, mask, color):
        m = mask & ~used
        buckets.append((label, int(greek[m].sum()), int(m.sum()), color))
        used[:] = used | m

    add_bucket("Greek-script (poly+mono)", is_greek_script, "#d62728")
    add_bucket("Cyrillic", is_cyrillic, "#ff7f0e")
    add_bucket("Han / CJK", is_han, "#9467bd")
    add_bucket("Latin letters", is_latin, "#2ca02c")
    add_bucket("Digits-only", is_digit, "#7f7f7f")
    add_bucket("Whitespace-only", is_ws, "#bcbd22")
    add_bucket("Structural (P/S/Z only)", is_struct, "#17becf")
    add_bucket("Special tokens", is_special, "#e377c2")
    add_bucket("Byte fragments", is_byte_frag, "#8c564b")
    rest = ~used
    buckets.append(("Other", int(greek[rest].sum()), int(rest.sum()), "#cccccc"))

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [b[0] for b in buckets]
    shares = [100 * b[1] / total_greek for b in buckets]
    colors = [b[3] for b in buckets]
    ypos = np.arange(len(labels))[::-1]
    bars = ax.barh(ypos, shares, color=colors, edgecolor="black", lw=0.5)
    for bar, b in zip(bars, buckets):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f}%  ({b[2]:,} tokens, {b[1]:,} mass)",
                va="center", fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Share of total Greek-token mass (%)")
    ax.set_title(f"Where Greek-sample tokens end up by script-bucket (total {total_greek:,} tokens)")
    ax.set_xlim(0, max(shares) * 1.4)
    fig.tight_layout()
    fig.savefig(OUT / "plots/03_script_mass_breakdown.png", dpi=130)
    plt.close(fig)

    # ---- PLOT 4: Zipf for Greek-script-only ----
    gs_counts = greek[is_greek_script]
    gs_nonzero = np.sort(gs_counts[gs_counts > 0])[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(np.arange(1, len(gs_nonzero) + 1), gs_nonzero, color="#d62728", lw=1.2)
    ax.set_xlabel("Rank within Greek-script tokens")
    ax.set_ylabel("Count in Greek sample")
    ax.set_title(f"Greek-script tokens only: {len(gs_nonzero):,} of {is_greek_script.sum():,} fire in Greek sample\n"
                 f"Cumulative mass: {100 * gs_counts.sum() / total_greek:.2f}% of total Greek mass")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "plots/04_greek_script_zipf.png", dpi=130)
    plt.close(fig)

    # ---- PLOT 5: Threshold sweep — Greek-script as classifier ----
    thresholds = [0, 1, 5, 10, 50, 100, 500, 1000, 10000, 100000, 1000000]
    rows = []
    for T in thresholds:
        kept = greek >= T  # tokens that pass count threshold
        n_kept = int(kept.sum())
        mass_kept = int(greek[kept].sum())
        gs_in_kept = int((kept & is_greek_script).sum())
        mass_gs_in_kept = int(greek[kept & is_greek_script].sum())
        prec_token = gs_in_kept / n_kept if n_kept else 0
        prec_mass = mass_gs_in_kept / mass_kept if mass_kept else 0
        rows.append({
            "threshold_T": T,
            "tokens_kept": n_kept,
            "greek_mass_kept": mass_kept,
            "share_of_total_greek_mass": mass_kept / total_greek if total_greek else 0,
            "greek_script_in_kept": gs_in_kept,
            "greek_mass_from_greek_script_in_kept": mass_gs_in_kept,
            "precision_token_level": prec_token,
            "precision_mass_level": prec_mass,
        })
    th_df = pl.DataFrame(rows)
    th_df.write_csv(OUT / "tables/threshold_sweep.tsv", separator="\t")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = [r["threshold_T"] for r in rows]
    ax1.semilogx([max(t, 0.5) for t in x], [r["tokens_kept"] for r in rows], marker="o", label="kept (any script)")
    ax1.semilogx([max(t, 0.5) for t in x], [r["greek_script_in_kept"] for r in rows], marker="s", color="#d62728", label="kept ∩ Greek-script")
    ax1.set_xlabel("Greek count threshold T")
    ax1.set_ylabel("Tokens kept (count ≥ T)")
    ax1.set_title("Tokens passing each Greek-count threshold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogx([max(t, 0.5) for t in x], [100 * r["precision_token_level"] for r in rows], marker="o", label="token-level precision")
    ax2.semilogx([max(t, 0.5) for t in x], [100 * r["precision_mass_level"] for r in rows], marker="s", color="#d62728", label="mass-level precision")
    ax2.set_xlabel("Greek count threshold T")
    ax2.set_ylabel("% of kept tokens that are Greek-script")
    ax2.set_title("Greek-script precision as count threshold rises")
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "plots/05_greek_script_threshold.png", dpi=130)
    plt.close(fig)

    # ---- TABLE: top 200 overall ----
    top_idx = np.argsort(-greek)[:200]
    rows = []
    for i in top_idx:
        rows.append({
            "rank": len(rows) + 1,
            "token_id": int(i),
            "decoded": decoded[i],
            "greek_count": int(greek[i]),
            "share_of_greek_mass_pct": 100 * float(greek[i]) / total_greek,
            "is_greek_script": bool(is_greek_script[i]),
            "is_latin": bool(is_latin[i]),
            "is_structural": bool(is_struct[i]),
            "is_digit": bool(is_digit[i]),
            "is_ws": bool(is_ws[i]),
            "is_special": bool(is_special[i]),
            "is_byte_fragment": bool(is_byte_frag[i]),
        })
    pl.DataFrame(rows).write_csv(OUT / "tables/top200_overall.tsv", separator="\t")

    # ---- TABLE: top 200 Greek-script ----
    gs_idx = np.where(is_greek_script)[0]
    gs_sorted = gs_idx[np.argsort(-greek[gs_idx])][:200]
    rows = []
    for i in gs_sorted:
        rows.append({
            "rank": len(rows) + 1,
            "token_id": int(i),
            "decoded": decoded[i],
            "greek_count": int(greek[i]),
            "share_of_greek_mass_pct": 100 * float(greek[i]) / total_greek,
            "has_greek_mono": bool(is_greek_mono[i]),
            "has_greek_poly": bool(is_greek_poly[i]),
            "has_latin": bool(is_latin[i]),
        })
    pl.DataFrame(rows).write_csv(OUT / "tables/top200_greek_script.tsv", separator="\t")

    # ---- TABLE: top 200 NON-Greek-script (the "shared infrastructure") ----
    non_gs_idx = np.where(~is_greek_script)[0]
    non_gs_sorted = non_gs_idx[np.argsort(-greek[non_gs_idx])][:200]
    rows = []
    for i in non_gs_sorted:
        rows.append({
            "rank": len(rows) + 1,
            "token_id": int(i),
            "decoded": decoded[i],
            "greek_count": int(greek[i]),
            "share_of_greek_mass_pct": 100 * float(greek[i]) / total_greek,
            "is_latin": bool(is_latin[i]),
            "is_cyrillic": bool(is_cyrillic[i]),
            "is_han": bool(is_han[i]),
            "is_structural": bool(is_struct[i]),
            "is_digit": bool(is_digit[i]),
            "is_ws": bool(is_ws[i]),
            "is_special": bool(is_special[i]),
            "is_byte_fragment": bool(is_byte_frag[i]),
        })
    pl.DataFrame(rows).write_csv(OUT / "tables/top200_NOT_greek_script.tsv", separator="\t")

    # ---- TABLE: Greek-script tokens with zero Greek count ----
    gs_zero = gs_idx[greek[gs_idx] == 0]
    rows = []
    for i in gs_zero:
        rows.append({
            "token_id": int(i),
            "decoded": decoded[i],
            "has_greek_mono": bool(is_greek_mono[i]),
            "has_greek_poly": bool(is_greek_poly[i]),
            "has_latin": bool(is_latin[i]),
        })
    pl.DataFrame(rows).write_csv(OUT / "tables/greek_script_zero_count.tsv", separator="\t")

    # ---- summary.json ----
    summary = {
        "ell_Grek_sample_total_tokens": total_greek,
        "vocab_total": 131072,
        "vocab_entries_with_greek_codepoint": int(is_greek_script.sum()),
        "greek_script_mono_only": int((is_greek_mono & ~is_greek_poly).sum()),
        "greek_script_poly_any":  int(is_greek_poly.sum()),
        "vocab_entries_fired_in_greek_sample": n_fired,
        "vocab_entries_fired_geq_10":   n_geq_10,
        "vocab_entries_fired_geq_100":  n_geq_100,
        "vocab_entries_fired_geq_1000": n_geq_1000,
        "greek_mass_from_greek_script_tokens": int(greek[is_greek_script].sum()),
        "greek_mass_from_greek_script_pct":   100 * float(greek[is_greek_script].sum()) / total_greek,
        "greek_script_tokens_that_fired":       int((greek[is_greek_script] > 0).sum()),
        "greek_script_tokens_zero_count":       int((greek[is_greek_script] == 0).sum()),
        "non_greek_script_tokens_that_fired":   int((greek[~is_greek_script] > 0).sum()),
        "script_buckets": {
            b[0]: {"tokens": b[2], "mass": b[1], "share_pct": 100 * b[1] / total_greek}
            for b in buckets
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Print top-line summary
    print("=" * 60)
    print(f"Total Greek-sample tokens         : {total_greek:,}")
    print(f"Vocab entries with Greek codepoint: {is_greek_script.sum():,}")
    print(f"Vocab entries fired in Greek      : {n_fired:,}  (≥10: {n_geq_10:,}; ≥100: {n_geq_100:,}; ≥1000: {n_geq_1000:,})")
    print(f"Greek mass in Greek-script tokens : {summary['greek_mass_from_greek_script_pct']:.2f}%")
    print(f"Greek-script tokens that fired    : {summary['greek_script_tokens_that_fired']:,} / {is_greek_script.sum():,}")
    print(f"Greek-script tokens with 0 count  : {summary['greek_script_tokens_zero_count']:,}")
    print(f"Non-Greek-script tokens that fired: {summary['non_greek_script_tokens_that_fired']:,}  (= the 'shared infrastructure')")
    print("=" * 60)
    print("Wrote plots/, tables/, summary.json under", OUT)


if __name__ == "__main__":
    main()

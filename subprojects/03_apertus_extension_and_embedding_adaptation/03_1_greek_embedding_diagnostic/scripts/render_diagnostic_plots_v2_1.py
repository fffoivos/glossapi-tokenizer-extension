"""v2.1 plots — filtered infiltrators + quantile-based hull occupancy.

Updates Fig 2b, Fig 6, Fig 11, Fig 12 from v2 with the floor-filtered
data and the quantile-based bands.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
V21 = ROOT / "geometry" / "v2_1"
FIG = ROOT / "figures" / "v2_1"
FIG.mkdir(parents=True, exist_ok=True)

COL_GREEK = "#1f77b4"
COL_NOT = "#ff7f0e"
COL_INFIL = "#d62728"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{png,pdf}}", flush=True)


def fig21_filtered_kde():
    """Fig 2b v2.1: filtered ¬Greek vs Greek's distribution + Greek-quantile lines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mat in zip(axes, ("E", "U")):
        d = np.load(V21 / f"distance_filtered_{mat}.npz")
        meta = json.load(open(V21 / f"infiltrators_filtered_{mat}.json"))
        gq = meta["greek_mahalanobis_quantiles"]
        ax.hist(d["greek_mahalanobis"], bins=80, density=True, alpha=0.6,
                 color=COL_GREEK, label=f"Greek (n={d['greek_mahalanobis'].size})")
        ax.hist(d["filtered_negreek_mahalanobis"], bins=120, density=True, alpha=0.5,
                 color=COL_NOT,
                 label=f"¬Greek filtered (n={d['filtered_negreek_mahalanobis'].size:,})")
        for q, ls in zip(("q10", "q25", "q50", "q75"), (":", "--", "-.", "-")):
            ax.axvline(gq[q], color="black", linestyle=ls, linewidth=0.7,
                        alpha=0.6, label=f"Greek {q} = {gq[q]:.1f}" if mat == "E" else None)
        ax.set_xlabel(f"Mahalanobis to μ_Greek (Greek's subspace) — {mat}")
        ax.set_ylabel("density" if mat == "E" else "")
        ax.set_title(f"{mat} matrix — filter: ‖row‖ > p1")
        if mat == "E":
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Figure 2b — Filtered ¬Greek vs Greek's own Mahalanobis distribution")
    save(fig, "fig02b_filtered_kde")


def fig61_quantile_hull():
    """Fig 6 v2.1: quantile-based hull occupancy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, mat in zip(axes, ("E", "U")):
        hull = json.load(open(V21 / f"hull_quantiles_{mat}.json"))
        qs = ["q10", "q25", "q50", "q75", "q90", "q99"]
        greek_fracs = [hull["fraction_of_greek_below_each_greek_quantile"][q] for q in qs]
        ng_fracs = [hull["fraction_of_filtered_negreek_below_each_greek_quantile"][q] for q in qs]
        x = np.arange(len(qs))
        width = 0.35
        ax.bar(x - width/2, greek_fracs, width, color=COL_GREEK, label="Greek (in-group)")
        ax.bar(x + width/2, ng_fracs, width, color=COL_NOT, label="¬Greek filtered")
        ax.set_xticks(x)
        ax.set_xticklabels(qs)
        ax.set_xlabel(f"Greek's own m-quantile threshold ({mat})")
        ax.set_ylabel("fraction of tokens with m ≤ threshold" if mat == "E" else "")
        ax.set_title(f"{mat} matrix")
        ax.set_ylim(0, 1.1)
        if mat == "E":
            ax.legend()
        # Annotate counts on ¬Greek bars
        ng_counts = [hull["count_of_filtered_negreek_below_each_greek_quantile"][q] for q in qs]
        for xi, ng, c in zip(x, ng_fracs, ng_counts):
            ax.annotate(f"{c:,}", xy=(xi + width/2, ng), ha="center", va="bottom", fontsize=7)
    fig.suptitle("Figure 6 — Quantile-based hull occupancy")
    save(fig, "fig06_hull_quantile")


def fig111_filtered_top20():
    """Fig 11 v2.1: filtered top-20."""
    for mat in ("E", "U"):
        d = json.load(open(V21 / f"infiltrators_filtered_{mat}.json"))
        rows = d["top_20_filtered_infiltrators"]
        cell = [[r["rank"], r["id"], r["decoded_text"], r["source_group"],
                  f"{r['row_norm']:.2f}", f"{r['mahalanobis_to_greek']:.3f}",
                  f"{r['percentile_in_greek_distribution']:.2f}"]
                 for r in rows]
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.axis("off")
        tbl = ax.table(
            cellText=cell,
            colLabels=["rank", "id", "decoded_text", "source", f"‖{mat}‖", "m_Greek", "%ile in Greek"],
            loc="center", cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        ax.set_title(f"Figure 11 ({mat}) — Top-20 filtered infiltrators (‖row‖>p1)")
        save(fig, f"fig11_filtered_top20_{mat}")


def fig121_percentile():
    """Fig 12 v2.1: top-1000 filtered percentile in Greek's distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mat in zip(axes, ("E", "U")):
        d = np.load(V21 / f"distance_filtered_{mat}.npz")
        pcts = d["top1000_percentile_in_greek"]
        ax.hist(pcts, bins=20, color=COL_INFIL, alpha=0.7, edgecolor="black")
        ax.set_xlabel("percentile of Greek's distance distribution")
        ax.set_ylabel("count of top-1000 filtered ¬Greek")
        ax.set_title(f"{mat} matrix")
        for q, label, color in [(10, "Greek q10", "blue"),
                                  (25, "Greek q25", "green"),
                                  (50, "Greek q50", "purple")]:
            ax.axvline(q, color=color, linestyle="--", linewidth=0.8, label=label)
        ax.legend(fontsize=7)
    fig.suptitle("Figure 12 — Top-1000 filtered ¬Greek on Greek's distance percentile")
    save(fig, "fig12_filtered_percentile")


def main():
    fig21_filtered_kde()
    fig61_quantile_hull()
    fig111_filtered_top20()
    fig121_percentile()
    print("[done] v2.1 figures under figures/v2_1/", flush=True)


if __name__ == "__main__":
    main()

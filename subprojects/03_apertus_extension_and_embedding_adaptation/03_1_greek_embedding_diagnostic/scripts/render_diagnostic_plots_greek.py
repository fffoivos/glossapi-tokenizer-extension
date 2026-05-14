"""v2 §4 plots — Greek vs ¬Greek diagnostic visualisations.

Reads artefacts from geometry/v2/. Produces all figures under figures/v2/.

Palette: Greek = #1f77b4 (blue); ¬Greek = #ff7f0e (orange).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
V2 = ROOT / "geometry" / "v2"
FIG = ROOT / "figures" / "v2"
FIG.mkdir(parents=True, exist_ok=True)

COL_GREEK = "#1f77b4"
COL_NOT = "#ff7f0e"
COL_INFIL = "#d62728"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{png,pdf}}", flush=True)


def fig1_distance_kde():
    """Fig 1: Euclidean distance to own centroid, faceted by matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mat in zip(axes, ("E", "U")):
        g = np.load(V2 / f"distance_greek_{mat}.npz")
        n = np.load(V2 / f"distance_not_greek_{mat}.npz")
        for label, data, color in [("Greek (n=1,494)", g["euclid"], COL_GREEK),
                                     ("¬Greek (n=126,990)", n["euclid"], COL_NOT)]:
            ax.hist(data, bins=80, density=True, alpha=0.5, color=color, label=label)
            ax.axvline(np.median(data), color=color, linestyle=":", linewidth=1.0)
        ax.set_xlabel(f"‖row − μ_own‖₂ ({mat})")
        ax.set_ylabel("density" if mat == "E" else "")
        ax.set_title(f"{mat} matrix")
        ax.legend()
    fig.suptitle("Figure 1 — Euclidean distance to own centroid (Greek vs ¬Greek)")
    save(fig, "fig01_distance_euclid")


def fig2_mahalanobis_kde():
    """Fig 2: Mahalanobis to own centroid (own subspace)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mat in zip(axes, ("E", "U")):
        g = np.load(V2 / f"distance_greek_{mat}.npz")
        n = np.load(V2 / f"distance_not_greek_{mat}.npz")
        sigma_g = float(np.std(g["mahalanobis"]))
        for label, data, color in [("Greek", g["mahalanobis"], COL_GREEK),
                                     ("¬Greek", n["mahalanobis"], COL_NOT)]:
            ax.hist(data, bins=80, density=True, alpha=0.5, color=color, label=label)
        for k, ls in zip((0.5, 1, 2, 3), (":", "--", "-.", "-")):
            ax.axvline(k * sigma_g, color="black", linestyle=ls, linewidth=0.7,
                        alpha=0.5, label=f"{k}σ_Greek" if mat == "E" else None)
        ax.set_xlabel(f"Mahalanobis to own μ ({mat})")
        ax.set_ylabel("density" if mat == "E" else "")
        ax.set_title(f"{mat} matrix")
        if mat == "E":
            ax.legend(fontsize=8)
    fig.suptitle("Figure 2 — Mahalanobis distance to own centroid (own subspace)")
    save(fig, "fig02_mahalanobis_own")


def fig2b_infiltrators_kde():
    """Fig 2b: ¬Greek tokens' Mahalanobis to μ_Greek using Greek's hull.

    Overlaid on Greek's own in-group Mahalanobis. Same sigma_Greek axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mat in zip(axes, ("E", "U")):
        g = np.load(V2 / f"distance_greek_{mat}.npz")
        # Load all ¬Greek Mahalanobis-to-Greek (computed by geometry script)
        inf_top1000 = np.load(V2 / f"infiltrators_top1000_{mat}_distance.npz")
        inf_summary = json.loads((V2 / f"infiltrators_{mat}.json").read_text())
        sigma_g = inf_summary["mahalanobis_std_greek_in_group"]
        # KDE of Greek (in-group) vs top-1000 ¬Greek (close to Greek hull)
        ax.hist(g["mahalanobis"], bins=80, density=True, alpha=0.6,
                 color=COL_GREEK, label="Greek (in-group)")
        ax.hist(inf_top1000["mahalanobis_to_greek"], bins=80, density=True, alpha=0.6,
                 color=COL_INFIL, label="top-1000 ¬Greek infiltrators")
        for k, ls in zip((0.5, 1, 2, 3), (":", "--", "-.", "-")):
            ax.axvline(k * sigma_g, color="black", linestyle=ls, linewidth=0.7,
                        alpha=0.5, label=f"{k}σ_Greek" if mat == "E" else None)
        ax.set_xlabel(f"Mahalanobis to μ_Greek (Greek's subspace) — {mat}")
        ax.set_ylabel("density" if mat == "E" else "")
        ax.set_title(f"{mat} matrix")
        if mat == "E":
            ax.legend(fontsize=8)
    fig.suptitle("Figure 2b — Top-1000 ¬Greek infiltrators vs Greek's own distance distribution")
    save(fig, "fig02b_infiltrators_top1000")


def fig3_top2_scatter():
    """Fig 3: top-2-PC scatter density per (group, matrix)."""
    groups = json.loads((ROOT / "geometry" / "groups_greek_vs_not.json").read_text())
    g_ids = np.asarray(groups["Greek"], dtype=np.int64)
    n_ids = np.asarray(groups["not_Greek"], dtype=np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col, mat in enumerate(("E", "U")):
        M = np.load(ROOT / "arrays" / f"{mat}_fp32.npy", mmap_mode="r")
        mu_g = np.load(V2 / f"mu_greek_{mat}.npy")
        mu_n = np.load(V2 / f"mu_not_greek_{mat}.npy")
        pc_g = np.load(V2 / f"pc_basis_greek_{mat}.npy")[:2]
        pc_n = np.load(V2 / f"pc_basis_not_greek_{mat}.npy")[:2]
        eig_g = np.load(V2 / f"pc_eigvals_greek_{mat}.npy")[:2]
        eig_n = np.load(V2 / f"pc_eigvals_not_greek_{mat}.npy")[:2]

        # Greek panel
        zg = ((np.asarray(M[g_ids]) - mu_g) @ pc_g.T) / np.sqrt(np.maximum(eig_g, 1e-12))
        ax = axes[0, col]
        ax.hexbin(zg[:, 0], zg[:, 1], gridsize=40, cmap="Blues", bins="log")
        # overlay top-20 infiltrators
        inf_summary = json.loads((V2 / f"infiltrators_{mat}.json").read_text())
        for r in inf_summary["top_20_infiltrators"]:
            tid = r["id"]
            zi = ((np.asarray(M[tid]) - mu_g) @ pc_g.T) / np.sqrt(np.maximum(eig_g, 1e-12))
            ax.plot(zi[0], zi[1], "o", color=COL_INFIL, markersize=5, alpha=0.8)
        ax.set_title(f"Greek on Greek-PCs ({mat})  — infiltrators in red")
        ax.set_xlabel("PC1 (z)")
        ax.set_ylabel("PC2 (z)")

        # ¬Greek panel (subsample to 20k for plot tractability)
        rng = np.random.default_rng(20260513)
        sub = rng.choice(n_ids.size, size=min(20000, n_ids.size), replace=False)
        zn = ((np.asarray(M[n_ids[sub]]) - mu_n) @ pc_n.T) / np.sqrt(np.maximum(eig_n, 1e-12))
        ax = axes[1, col]
        ax.hexbin(zn[:, 0], zn[:, 1], gridsize=40, cmap="Oranges", bins="log")
        ax.set_title(f"¬Greek on ¬Greek-PCs ({mat}) — 20k subsample")
        ax.set_xlabel("PC1 (z)")
        ax.set_ylabel("PC2 (z)")
    fig.suptitle("Figure 3 — Top-2-PC z-scored scatter density (own subspace per panel)")
    save(fig, "fig03_top2_pc_scatter")


def fig6_hull_stacked_bars():
    """Fig 6: hull occupancy bars + infiltrator share."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    band_labels = ["≤0.5σ", "0.5–1σ", "1–2σ", "2–3σ", ">3σ"]
    band_colors = ["#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"]
    for ax, mat in zip(axes, ("E", "U")):
        g_hull = json.loads((V2 / f"hull_greek_{mat}.json").read_text())
        inf = json.loads((V2 / f"infiltrators_{mat}.json").read_text())
        g_fracs = [
            g_hull["frac_within_0_5_sigma"],
            g_hull["frac_within_1_sigma"] - g_hull["frac_within_0_5_sigma"],
            g_hull["frac_within_2_sigma"] - g_hull["frac_within_1_sigma"],
            g_hull["frac_within_3_sigma"] - g_hull["frac_within_2_sigma"],
            1.0 - g_hull["frac_within_3_sigma"],
        ]
        i_fracs = [
            inf["frac_negreek_within_0_5_sigma"],
            inf["frac_negreek_within_1_sigma"] - inf["frac_negreek_within_0_5_sigma"],
            inf["frac_negreek_within_2_sigma"] - inf["frac_negreek_within_1_sigma"],
            inf["frac_negreek_within_3_sigma"] - inf["frac_negreek_within_2_sigma"],
            1.0 - inf["frac_negreek_within_3_sigma"],
        ]
        x = ["Greek\n(own hull)", "¬Greek\n(Greek's hull)"]
        bottom_g = 0
        bottom_i = 0
        for fg, fi, color, label in zip(g_fracs, i_fracs, band_colors, band_labels):
            ax.bar(x[0], fg, bottom=bottom_g, color=color,
                    label=label if mat == "E" else None)
            ax.bar(x[1], fi, bottom=bottom_i, color=color)
            bottom_g += fg
            bottom_i += fi
        # Annotate count within 1σ
        ax.annotate(f"{int(g_hull['frac_within_1_sigma'] * 1494)}/1494",
                     xy=(x[0], g_hull["frac_within_1_sigma"]), ha="center", va="bottom")
        ax.annotate(f"{inf['count_negreek_within_1_sigma']:,}/126,990",
                     xy=(x[1], inf["frac_negreek_within_1_sigma"]), ha="center", va="bottom")
        ax.set_title(f"{mat} matrix")
        ax.set_ylabel("fraction of tokens" if mat == "E" else "")
        ax.set_ylim(0, 1.1)
        if mat == "E":
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Figure 6 — Hull occupancy (in-group Greek + infiltrating ¬Greek), Greek's σ axis")
    save(fig, "fig06_hull_occupancy")


def fig6b_infiltrator_source_breakdown():
    """Fig 6b: source-group composition of within-1σ infiltrators."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat in zip(axes, ("E", "U")):
        inf = json.loads((V2 / f"infiltrators_{mat}.json").read_text())
        breakdown = inf.get("count_within_1_sigma_by_source_group", {})
        if not breakdown:
            ax.text(0.5, 0.5, "no infiltrators within 1σ", ha="center", va="center")
            continue
        labels, counts = zip(*sorted(breakdown.items(), key=lambda kv: -kv[1]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        ax.barh(labels, counts, color=colors)
        ax.set_xlabel(f"count within 1σ of μ_Greek ({mat})")
        ax.set_title(f"{mat} matrix — total {sum(counts):,}")
        for i, c in enumerate(counts):
            ax.annotate(f"{c:,}", xy=(c, i), xytext=(3, 0),
                         textcoords="offset points", va="center", fontsize=9)
    fig.suptitle("Figure 6b — Source-group composition of ¬Greek tokens inside Greek's 1σ hull")
    save(fig, "fig06b_infiltrator_source_breakdown")


def fig7_scree_with_mp():
    """Fig 7: scree per (group, matrix) with MP edge."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (group, label) in enumerate([("greek", "Greek"), ("not_greek", "¬Greek")]):
        for col, mat in enumerate(("E", "U")):
            spec = np.load(V2 / f"spectrum_{group}_{mat}.npz")["eigvals"]
            meta = json.loads((V2 / f"spectrum_{group}_{mat}.json").read_text())
            ax = axes[row, col]
            ax.plot(np.arange(1, len(spec) + 1), spec, color=COL_GREEK if group == "greek" else COL_NOT)
            ax.axhline(meta["mp_upper_edge"], color="black", linestyle="--",
                        linewidth=0.8, label=f"MP edge = {meta['mp_upper_edge']:.4f}")
            ax.axvline(meta["k_significant"], color="red", linestyle=":",
                        linewidth=0.8, label=f"K_sig = {meta['k_significant']}")
            ax.set_yscale("log")
            ax.set_xlabel(f"PC index ({mat})")
            ax.set_ylabel("eigenvalue (log)" if col == 0 else "")
            ax.set_title(f"{label} — {mat}")
            ax.legend(fontsize=8)
    fig.suptitle("Figure 7 — Per-group scree spectrum with Marchenko-Pastur edge")
    save(fig, "fig07_scree_mp")


def fig8_cumulative_variance():
    """Fig 8: cumulative variance per group."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mat in zip(axes, ("E", "U")):
        for group, color, label in [("greek", COL_GREEK, "Greek"),
                                       ("not_greek", COL_NOT, "¬Greek")]:
            spec = np.load(V2 / f"spectrum_{group}_{mat}.npz")["eigvals"]
            meta = json.loads((V2 / f"spectrum_{group}_{mat}.json").read_text())
            cum = np.cumsum(spec) / spec.sum()
            ax.plot(np.arange(1, len(cum) + 1), cum, color=color, label=label)
            k_sig = meta["k_significant"]
            ax.plot([k_sig], [cum[min(k_sig - 1, len(cum) - 1)]],
                     "o", color=color)
        for y, ls in zip((0.5, 0.9, 0.95), (":", "--", "-.")):
            ax.axhline(y, color="gray", linestyle=ls, linewidth=0.5)
        ax.set_xlabel(f"PC index ({mat})")
        ax.set_ylabel("cumulative variance fraction" if mat == "E" else "")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title(f"{mat} matrix")
    fig.suptitle("Figure 8 — Cumulative variance vs PC index")
    save(fig, "fig08_cumvar")


def fig9_pr_kappa():
    """Fig 9: participation ratio + κ comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars_pr = {}
    bars_k = {}
    for mat in ("E", "U"):
        for group, label in [("greek", f"Greek/{mat}"), ("not_greek", f"¬Greek/{mat}")]:
            meta = json.loads((V2 / f"spectrum_{group}_{mat}.json").read_text())
            bars_pr[label] = meta["participation_ratio"]
            bars_k[label] = meta["shape_anisotropy_kappa"]
    for ax, vals, title in [(axes[0], bars_pr, "Participation ratio"),
                              (axes[1], bars_k, "Shape anisotropy κ")]:
        labels = list(vals.keys())
        values = [vals[k] for k in labels]
        colors = [COL_GREEK if "Greek" in k and "¬" not in k else COL_NOT for k in labels]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(values):
            ax.annotate(f"{v:.3f}", xy=(i, v), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Figure 9 — Shape statistics")
    save(fig, "fig09_pr_kappa")


def fig11_infiltrator_gallery():
    """Fig 11: top-20 infiltrator tables-as-figure."""
    for mat in ("E", "U"):
        inf = json.loads((V2 / f"infiltrators_{mat}.json").read_text())
        rows = inf["top_20_infiltrators"]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        cell = [[r["rank"], r["id"], r["raw_token"], r["decoded_text"],
                  r["source_group"], f"{r['mahalanobis_to_greek']:.3f}"]
                 for r in rows]
        tbl = ax.table(
            cellText=cell,
            colLabels=["rank", "id", "raw_token", "decoded_text", "source_group", "m_Greek"],
            loc="center", cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        ax.set_title(f"Figure 11 ({mat}) — Top-20 ¬Greek tokens nearest the Greek centroid")
        save(fig, f"fig11_top20_infiltrators_{mat}")


def fig12_percentile_histogram():
    """Fig 12 (new for §3.10): where do the top-1000 ¬Greek tokens
    fall in Greek's own distance distribution?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mat in zip(axes, ("E", "U")):
        d = np.load(V2 / f"infiltrators_top1000_{mat}_distance.npz")
        ax.hist(d["percentile_in_greek_distribution"], bins=20, color=COL_INFIL,
                 alpha=0.7, edgecolor="black")
        ax.set_xlabel("percentile of Greek's own distance distribution")
        ax.set_ylabel("count of top-1000 ¬Greek tokens")
        ax.set_title(f"{mat} matrix")
        ax.axvline(50, color="gray", linestyle="--", linewidth=0.8,
                    label="Greek median")
        ax.legend(fontsize=8)
    fig.suptitle("Figure 12 — Top-1000 ¬Greek infiltrators on Greek's distance-distribution percentile axis")
    save(fig, "fig12_top1000_percentile")


def main():
    print("=== plotting ===", flush=True)
    fig1_distance_kde()
    fig2_mahalanobis_kde()
    fig2b_infiltrators_kde()
    fig3_top2_scatter()
    fig6_hull_stacked_bars()
    fig6b_infiltrator_source_breakdown()
    fig7_scree_with_mp()
    fig8_cumulative_variance()
    fig9_pr_kappa()
    fig11_infiltrator_gallery()
    fig12_percentile_histogram()
    print("[done] all figures under figures/v2/", flush=True)


if __name__ == "__main__":
    main()

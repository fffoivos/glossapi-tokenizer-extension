"""v3 — render per-language + pairwise plots from the v3_perlang artefacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic"
)
V3 = SP / "artifacts" / "geometry" / "v3_perlang"
FIG = SP / "artifacts" / "figures" / "v3_perlang"
FIG.mkdir(parents=True, exist_ok=True)

GREEK = "ell_Grek"

# Consistent palette
PALETTE = {
    "ell_Grek":             "#d62728",  # red — anchor
    "hin_Deva":             "#1f77b4",
    "hye_Armn":             "#9467bd",
    "heb_Hebr":             "#17becf",
    "kat_Geor":             "#bcbd22",
    "tha_Thai":             "#e377c2",
    "kor_Hang":             "#8c564b",
    "fas_Arab":             "#7f7f7f",
    "eng_Latn_fineweb_hq":  "#ff7f0e",
    "fra_Latn":             "#2ca02c",
    "deu_Latn":             "#aec7e8",
}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{png,pdf}}", flush=True)


def load_summary():
    return json.loads((V3 / "per_lang_summary.json").read_text())


def fig01_scree_per_lang():
    summary = load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat in zip(axes, ("E", "U")):
        for lang, color in PALETTE.items():
            spec = np.load(V3 / f"spectrum_{lang}_{mat}.npz")["eigvals"]
            ax.plot(np.arange(1, len(spec) + 1), spec, color=color, label=lang, alpha=0.9, linewidth=1.4)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(f"PC index ({mat})")
        ax.set_ylabel("eigenvalue (log)" if mat == "E" else "")
        ax.set_title(f"{mat} matrix")
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("Figure 1 — Scree per language (overlay)")
    save(fig, "fig01_scree_per_lang")


def fig02_cumvar_per_lang():
    summary = load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat in zip(axes, ("E", "U")):
        for lang, color in PALETTE.items():
            spec = np.load(V3 / f"spectrum_{lang}_{mat}.npz")["eigvals"]
            cum = np.cumsum(spec) / spec.sum()
            s = summary["languages"][f"{lang}_{mat}"]["spectrum"]
            label = f"{lang} (K_sig={s['k_significant']})"
            ax.plot(np.arange(1, len(cum) + 1), cum, color=color, label=label, alpha=0.9, linewidth=1.4)
            ksig = s["k_significant"]
            ax.plot([ksig], [cum[min(ksig - 1, len(cum) - 1)]], "o", color=color, markersize=4)
        for y, ls in zip((0.5, 0.9, 0.95), (":", "--", "-.")):
            ax.axhline(y, color="gray", linestyle=ls, linewidth=0.5)
        ax.set_xlabel(f"PC index ({mat})")
        ax.set_ylabel("cumulative variance" if mat == "E" else "")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{mat} matrix")
    axes[0].legend(fontsize=6, loc="lower right", ncol=2)
    fig.suptitle("Figure 2 — Cumulative variance per language")
    save(fig, "fig02_cumvar_per_lang")


def fig03_pr_kappa():
    summary = load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    langs = list(PALETTE.keys())
    x = np.arange(len(langs))
    width = 0.4
    for i, mat in enumerate(("E", "U")):
        pr_vals = [summary["languages"][f"{l}_{mat}"]["spectrum"]["participation_ratio"] for l in langs]
        k_vals = [summary["languages"][f"{l}_{mat}"]["spectrum"]["shape_anisotropy_kappa"] for l in langs]
        axes[0].bar(x + (i - 0.5) * width, pr_vals, width,
                     color=[PALETTE[l] for l in langs], alpha=0.5 + 0.5 * i,
                     edgecolor="black", linewidth=0.5,
                     label=f"{mat} matrix")
        axes[1].bar(x + (i - 0.5) * width, k_vals, width,
                     color=[PALETTE[l] for l in langs], alpha=0.5 + 0.5 * i,
                     edgecolor="black", linewidth=0.5,
                     label=f"{mat} matrix")
    for ax, title, ylabel in [(axes[0], "Participation ratio", "PR"),
                                (axes[1], "Shape anisotropy κ", "κ")]:
        ax.set_xticks(x)
        ax.set_xticklabels(langs, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle("Figure 3 — Shape statistics across 11 languages")
    save(fig, "fig03_pr_kappa")


def fig04_hull_quantiles():
    summary = load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    langs = list(PALETTE.keys())
    x = np.arange(len(langs))
    for ax, mat in zip(axes, ("E", "U")):
        for q in (10, 25, 50, 75, 90):
            vals = [summary["languages"][f"{l}_{mat}"]["hull_quantiles"][f"q{q}"] for l in langs]
            ax.plot(x, vals, "-o", label=f"q{q}", markersize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(langs, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{mat} matrix")
        ax.set_ylabel("Mahalanobis distance" if mat == "E" else "")
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 4 — Within-language Mahalanobis hull quantiles")
    save(fig, "fig04_hull_quantiles")


def fig05_pairwise_centroid_cosine():
    summary = load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, mat in zip(axes, ("E", "U")):
        labels = summary["pairwise"][mat]["labels"]
        cos_mat = np.asarray(summary["pairwise"][mat]["centroid_cosine"])
        im = ax.imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{mat} matrix")
        plt.colorbar(im, ax=ax, label="cosine")
        # annotate
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cos_mat[i, j]:+.2f}", ha="center", va="center",
                         fontsize=6, color="black" if abs(cos_mat[i, j]) < 0.5 else "white")
    fig.suptitle("Figure 5 — Pairwise centroid cosines (11×11 per matrix)")
    save(fig, "fig05_pairwise_centroid_cosine")


def fig06_hull_overlap():
    summary = load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, mat in zip(axes, ("E", "U")):
        labels = summary["pairwise"][mat]["labels"]
        ov = np.asarray(summary["pairwise"][mat]["hull_overlap_frac_j_within_i_q25"])
        # Mask diagonal for visualization (always 0.25 by construction)
        ov_disp = ov.copy()
        np.fill_diagonal(ov_disp, np.nan)
        im = ax.imshow(ov_disp, cmap="viridis", vmin=0, vmax=ov_disp[~np.isnan(ov_disp)].max())
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{mat} matrix")
        plt.colorbar(im, ax=ax, label="frac of j inside i's q25 hull")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j: continue
                ax.text(j, i, f"{ov[i, j]:.2f}", ha="center", va="center",
                         fontsize=6, color="white" if ov[i, j] < 0.5 else "black")
    fig.suptitle("Figure 6 — Hull overlap: fraction of L_j tokens inside L_i's q25 hull")
    save(fig, "fig06_pairwise_hull_overlap")


def fig07_mahalanobis_kde():
    """Per-language Mahalanobis KDE, overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat in zip(axes, ("E", "U")):
        for lang, color in PALETTE.items():
            d = np.load(V3 / f"distance_{lang}_{mat}.npz")
            ax.hist(d["mahalanobis"], bins=60, density=True, alpha=0.4,
                     color=color, label=lang)
        ax.set_xlabel(f"in-group Mahalanobis ({mat})")
        ax.set_ylabel("density" if mat == "E" else "")
        ax.set_title(f"{mat} matrix")
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("Figure 7 — In-group Mahalanobis distribution per language")
    save(fig, "fig07_mahalanobis_kde")


def main():
    print("=== rendering v3 per-language plots ===", flush=True)
    fig01_scree_per_lang()
    fig02_cumvar_per_lang()
    fig03_pr_kappa()
    fig04_hull_quantiles()
    fig05_pairwise_centroid_cosine()
    fig06_hull_overlap()
    fig07_mahalanobis_kde()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

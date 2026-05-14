"""v2.3 — render the figures promised in the plan but missing from v2:
  Fig 4   cos(v_t, top-1 PC) KDE per group, per matrix
  Fig 5   top-K PC angular heatmap (mean |cos(v_t, e_g,k)| for k = 1..K_HEATMAP)
  Fig 10  binary classifier summary (F1 + weight-direction alignment)

Plus refresh Figs 7/8/9 with the corrected full ¬Greek spectrum.
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
FIG = ROOT / "figures" / "v2_3"
FIG.mkdir(parents=True, exist_ok=True)

COL_GREEK = "#1f77b4"
COL_NOT = "#ff7f0e"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{png,pdf}}", flush=True)


def fig4_top1_pc_cosines():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mat in zip(axes, ("E", "U")):
        g = np.load(V2 / f"direction_cosines_greek_{mat}.npz")
        n = np.load(V2 / f"direction_cosines_not_greek_{mat}.npz")
        ax.hist(g["cos_top_k"][:, 0], bins=80, density=True, alpha=0.6,
                 color=COL_GREEK, label=f"Greek (n={g['cos_top_k'].shape[0]})")
        ax.hist(n["cos_top_k"][:, 0], bins=80, density=True, alpha=0.5,
                 color=COL_NOT, label=f"¬Greek (n={n['cos_top_k'].shape[0]:,})")
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(f"cos(v_t, top-1 PC of own group) — {mat}")
        ax.set_ylabel("density" if mat == "E" else "")
        ax.set_title(f"{mat} matrix")
        ax.legend(fontsize=8)
    fig.suptitle("Figure 4 — Distribution of cosine to own-group's top-1 PC axis")
    save(fig, "fig04_top1_pc_cosine")


def fig5_top_k_pc_angular_heatmap():
    summary = json.load(open(V2 / "direction_cosines_summary.json"))
    # 4 rows (Greek/E, Greek/U, ¬Greek/E, ¬Greek/U) × K_HEATMAP cols
    K = max(len(summary["by_matrix_group"][k]["mean_abs_cos_per_pc_top32"])
             for k in summary["by_matrix_group"])
    rows_data, row_labels = [], []
    for mat in ("E", "U"):
        for grp, gname in (("greek", "Greek"), ("not_greek", "¬Greek")):
            key = f"{grp}_{mat}"
            v = summary["by_matrix_group"][key]["mean_abs_cos_per_pc_top32"]
            padded = [x if x is not None else float("nan") for x in v] + [float("nan")] * (K - len(v))
            rows_data.append(padded)
            row_labels.append(f"{gname}/{mat}")
    arr = np.array(rows_data)
    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(f"PC index (top {K})")
    ax.set_title("Figure 5 — Mean |cos(v_t, e_g,k)| across top-K PCs per (group, matrix)")
    plt.colorbar(im, ax=ax, label="mean |cosine|")
    save(fig, "fig05_top_k_pc_angular_heatmap")


def fig10_binary_classifier_summary():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-class metrics (precision, recall, F1) per matrix
    ax = axes[0]
    metric_names = ["precision", "recall", "f1"]
    x = np.arange(len(metric_names))
    width = 0.35
    for i, mat in enumerate(("E", "U")):
        d = json.load(open(V2 / f"linear_classifier_binary_{mat}.json"))
        vals = [d[f"greek_{m}"] for m in metric_names]
        ax.bar(x + (i - 0.5) * width, vals,
                width, label=f"{mat} matrix", color=[COL_GREEK, COL_NOT][i])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Greek-class metrics (binary classifier)")
    ax.legend()

    # Right: weight-direction alignment
    ax = axes[1]
    align_keys = [
        ("vs μ_Greek − μ_¬Greek",      "cos_weight_vs_mu_greek_minus_mu_negreek"),
        ("vs μ_Greek − μ_global",      "cos_weight_vs_mu_greek_minus_mu_global"),
        ("vs top-1 PC of Greek",       "cos_weight_vs_top1_pc_greek"),
    ]
    x = np.arange(len(align_keys))
    for i, mat in enumerate(("E", "U")):
        d = json.load(open(V2 / f"linear_classifier_binary_{mat}.json"))
        vals = [d[k] for _, k in align_keys]
        ax.bar(x + (i - 0.5) * width, vals,
                width, color=[COL_GREEK, COL_NOT][i], label=f"{mat} matrix")
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in align_keys], rotation=15, ha="right")
    ax.set_ylabel("cosine")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Greek-direction (binary-classifier weight) alignment")
    ax.legend()
    fig.suptitle("Figure 10 — Binary Greek-vs-¬Greek classifier summary")
    save(fig, "fig10_binary_classifier_summary")


def fig07_refresh_scree():
    """Refresh Fig 7 (per-group scree) with the corrected full ¬Greek spectrum."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (group, label) in enumerate([("greek", "Greek"), ("not_greek", "¬Greek")]):
        for col, mat in enumerate(("E", "U")):
            spec = np.load(V2 / f"spectrum_{group}_{mat}.npz")["eigvals"]
            meta = json.load(open(V2 / f"spectrum_{group}_{mat}.json"))
            ax = axes[row, col]
            ax.plot(np.arange(1, len(spec) + 1), spec,
                     color=COL_GREEK if group == "greek" else COL_NOT)
            ax.axhline(meta["mp_upper_edge"], color="black", linestyle="--",
                        linewidth=0.8, label=f"MP edge = {meta['mp_upper_edge']:.4f}")
            ax.axvline(meta["k_significant"], color="red", linestyle=":",
                        linewidth=0.8, label=f"K_sig = {meta['k_significant']}")
            ax.set_yscale("log")
            ax.set_xlabel(f"PC index ({mat})")
            ax.set_ylabel("eigenvalue (log)" if col == 0 else "")
            ax.set_title(f"{label} — {mat}  (method={meta['method']})")
            ax.legend(fontsize=8)
    fig.suptitle("Figure 7 — Per-group scree with MP edge (full spectrum, ¬Greek corrected)")
    save(fig, "fig07_scree_mp_full")


def fig08_refresh_cumvar():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mat in zip(axes, ("E", "U")):
        for group, color, label in [("greek", COL_GREEK, "Greek"),
                                       ("not_greek", COL_NOT, "¬Greek")]:
            spec = np.load(V2 / f"spectrum_{group}_{mat}.npz")["eigvals"]
            meta = json.load(open(V2 / f"spectrum_{group}_{mat}.json"))
            s_sorted = np.sort(spec)[::-1]
            cum = np.cumsum(s_sorted) / s_sorted.sum()
            ax.plot(np.arange(1, len(cum) + 1), cum, color=color,
                     label=f"{label} (K_sig={meta['k_significant']})")
            ksig = meta["k_significant"]
            ax.plot([ksig], [cum[min(ksig - 1, len(cum) - 1)]],
                     "o", color=color)
        for y, ls in zip((0.5, 0.9, 0.95), (":", "--", "-.")):
            ax.axhline(y, color="gray", linestyle=ls, linewidth=0.5)
        ax.set_xlabel(f"PC index ({mat})")
        ax.set_ylabel("cumulative variance fraction" if mat == "E" else "")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title(f"{mat} matrix")
    fig.suptitle("Figure 8 — Cumulative variance (full spectrum, ¬Greek corrected)")
    save(fig, "fig08_cumvar_full")


def fig09_refresh_pr_kappa():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars_pr, bars_k = {}, {}
    for mat in ("E", "U"):
        for group, label in [("greek", f"Greek/{mat}"), ("not_greek", f"¬Greek/{mat}")]:
            meta = json.load(open(V2 / f"spectrum_{group}_{mat}.json"))
            bars_pr[label] = meta["participation_ratio"]
            bars_k[label] = meta["shape_anisotropy_kappa"]
    for ax, vals, title in [(axes[0], bars_pr, "Participation ratio"),
                              (axes[1], bars_k, "Shape anisotropy κ")]:
        labels = list(vals.keys())
        values = [vals[k] for k in labels]
        colors = [COL_GREEK if "¬" not in k else COL_NOT for k in labels]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        for i, v in enumerate(values):
            ax.annotate(f"{v:.3f}", xy=(i, v), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Figure 9 — Shape statistics (full spectrum, ¬Greek corrected)")
    save(fig, "fig09_pr_kappa_full")


def main():
    fig4_top1_pc_cosines()
    fig5_top_k_pc_angular_heatmap()
    fig10_binary_classifier_summary()
    fig07_refresh_scree()
    fig08_refresh_cumvar()
    fig09_refresh_pr_kappa()
    print("[done] v2.3 figures under figures/v2_3/", flush=True)


if __name__ == "__main__":
    main()

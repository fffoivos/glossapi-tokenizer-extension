"""v3 — pairwise subspace overlap between per-language PC bases.

For each pair (L_i, L_j) and each matrix:
  - top-1 PC cosine: |cos(e_i,1, e_j,1)|  (orientation of the dominant direction)
  - subspace overlap: mean(cos²(θ_k)) where {θ_k} are the principal angles between
    L_i's K_sig-dim subspace and L_j's K_sig-dim subspace
      cos(θ_k) = singular values of B_i B_j^T  (B = orthonormal basis as rows)
      "subspace overlap" = mean(cos²) = trace(B_i B_j^T B_j B_i^T) / min(K_i, K_j)
      = fraction of L_i's variance that lies in L_j's subspace (per shared dim)
  - principal angles for a handful of interesting Greek pairs (full distribution)

Output:
  artifacts/geometry/v3_perlang/subspace_overlap_{E,U}.json
  artifacts/geometry/v3_perlang/subspace_overlap_{E,U}.npy (11×11 mean cos² matrix)
  artifacts/geometry/v3_perlang/top1_pc_cosine_{E,U}.npy (11×11 |cos(top-1 PCs)|)
  artifacts/figures/v3_perlang/fig08_subspace_overlap_{E,U}.png
  artifacts/figures/v3_perlang/fig09_top1_pc_cosine.png
  artifacts/figures/v3_perlang/fig10_principal_angles_greek_pairs.png
"""
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


def principal_angle_cosines(B_i: np.ndarray, B_j: np.ndarray) -> np.ndarray:
    """B_i: (k_i, d), B_j: (k_j, d), both with orthonormal rows.
    Returns the cosines of principal angles, descending — singular values of B_i B_j^T."""
    M = B_i @ B_j.T          # (k_i, k_j)
    return np.linalg.svd(M, compute_uv=False)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    groups = json.loads((V3 / "groups.json").read_text())
    LANGS = list(groups["languages"].keys())

    summary = {}
    for mat in ("E", "U"):
        print(f"\n=== {mat} ===", flush=True)
        bases = {L: np.load(V3 / f"pc_basis_{L}_{mat}.npy") for L in LANGS}
        # The bases produced by phase0_perlang_geometry_v3 from np.linalg.eigh ARE orthonormal
        # (eigenvectors of a real symmetric matrix). Verify for sanity on Greek:
        Bg = bases["ell_Grek"]
        prod = Bg[:5] @ Bg[:5].T
        if not np.allclose(prod, np.eye(5), atol=1e-4):
            print(f"  warning: {mat} bases may not be perfectly orthonormal; "
                  f"deviation max={np.abs(prod - np.eye(5)).max():.2e}")

        N = len(LANGS)
        overlap = np.zeros((N, N), dtype=np.float32)
        top1_cos = np.zeros((N, N), dtype=np.float32)
        for i, Li in enumerate(LANGS):
            Bi = bases[Li]
            for j, Lj in enumerate(LANGS):
                Bj = bases[Lj]
                # top-1 PC cosine (absolute, since sign is arbitrary)
                top1_cos[i, j] = abs(float(Bi[0] @ Bj[0]))
                # Subspace overlap: mean cos²(θ) across all principal angles
                cos_angles = principal_angle_cosines(Bi, Bj)
                k_min = min(Bi.shape[0], Bj.shape[0])
                overlap[i, j] = float((cos_angles[:k_min] ** 2).mean())
        np.save(V3 / f"subspace_overlap_{mat}.npy", overlap)
        np.save(V3 / f"top1_pc_cosine_{mat}.npy", top1_cos)

        summary[mat] = {
            "labels": LANGS,
            "subspace_overlap_mean_cos2": overlap.tolist(),
            "top1_pc_cosine_abs": top1_cos.tolist(),
        }

        # ── plot: subspace overlap heatmap ─────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 7))
        disp = overlap.copy()
        np.fill_diagonal(disp, np.nan)
        im = ax.imshow(disp, cmap="viridis", vmin=0,
                        vmax=np.nanmax(disp))
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(LANGS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(LANGS, fontsize=8)
        plt.colorbar(im, ax=ax, label="mean cos²(principal angle)  =  shared-variance fraction")
        for i in range(N):
            for j in range(N):
                if i == j: continue
                v = overlap[i, j]
                col = "white" if v < disp[~np.isnan(disp)].mean() else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=col)
        ax.set_title(f"Subspace overlap (mean cos² of principal angles) — {mat}")
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"fig08_subspace_overlap_{mat}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved fig08_subspace_overlap_{mat}.{{png,pdf}}", flush=True)

    # ── single combined figure for top-1 PC cosine across E and U ──────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, mat in zip(axes, ("E", "U")):
        t1 = np.load(V3 / f"top1_pc_cosine_{mat}.npy")
        disp = t1.copy()
        np.fill_diagonal(disp, np.nan)
        im = ax.imshow(disp, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(LANGS)))
        ax.set_yticks(range(len(LANGS)))
        ax.set_xticklabels(LANGS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(LANGS, fontsize=8)
        plt.colorbar(im, ax=ax, label="|cos(top-1 PCs)|")
        ax.set_title(f"{mat} matrix")
        for i in range(len(LANGS)):
            for j in range(len(LANGS)):
                if i == j: continue
                v = t1[i, j]
                col = "white" if v < 0.4 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=col)
    fig.suptitle("Figure 9 — Top-1 PC alignment (absolute cosine) between per-language dominant directions")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig09_top1_pc_cosine.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig09_top1_pc_cosine.{png,pdf}", flush=True)

    # ── principal angles distribution for Greek vs select partners (E only) ──
    fig, ax = plt.subplots(figsize=(11, 5))
    Bg = np.load(V3 / "pc_basis_ell_Grek_E.npy")
    PARTNERS = ["hye_Armn", "heb_Hebr", "hin_Deva", "kat_Geor", "tha_Thai",
                 "kor_Hang", "fas_Arab", "deu_Latn", "fra_Latn", "eng_Latn_fineweb_hq"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(PARTNERS)))
    for L, c in zip(PARTNERS, colors):
        B = np.load(V3 / f"pc_basis_{L}_E.npy")
        angles_cos = principal_angle_cosines(Bg, B)
        # plot cos(θ) sorted descending; closer to 1 = more aligned subspace direction
        k = min(len(angles_cos), 200)
        ax.plot(np.arange(1, k + 1), angles_cos[:k], color=c, label=L, linewidth=1.4)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, label="parallel directions")
    ax.set_xlabel("principal angle index (top-K shared dims)")
    ax.set_ylabel("cos(θ_k)")
    ax.set_title("Figure 10 — Principal-angle cosines: Greek's PC subspace vs each partner (E, top-200)")
    ax.legend(fontsize=8, ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig10_principal_angles_greek_pairs_E.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig10_principal_angles_greek_pairs_E.{png,pdf}", flush=True)

    # ── persistence ───────────────────────────────────────────────────
    (V3 / "subspace_overlap_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] subspace overlap artefacts written")


if __name__ == "__main__":
    main()

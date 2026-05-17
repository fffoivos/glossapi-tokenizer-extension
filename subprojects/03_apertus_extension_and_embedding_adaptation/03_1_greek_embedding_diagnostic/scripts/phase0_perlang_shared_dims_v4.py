"""v3 — quantify how much of each language-pair's PC subspace is genuinely shared.

For two K-dim subspaces with orthonormal bases B_i (K_i × d) and B_j (K_j × d):
  cos(θ_k) = k-th singular value of B_i B_j^T,  k = 1..min(K_i, K_j)

  Σ cos²(θ_k) = trace(B_i B_j^T B_j B_i^T)
              = expected number of dimensions of B_i contained in B_j's span,
                in the sense that projecting B_i onto B_j preserves Σ cos² worth of "directions".

  Random baseline: E[Σ cos²(θ_k)] = K_i * K_j / d  (for random orthonormal frames in R^d).

  Excess shared dims = observed Σ cos² − K_i K_j / d.
  Normalised excess = excess / min(K_i, K_j)  ∈ [-K_max/d, 1].

Variance-weighted alternative (asymmetric):
  For L_i's K_sig top eigendirections {e_i,k} with eigenvalues {λ_i,k}:
    var_capture(L_j ⊃ L_i) = (Σ_k λ_i,k · ‖B_j e_i,k‖²) / (Σ_k λ_i,k)
  = the fraction of L_i's K-truncated total variance that lies in L_j's subspace.

Outputs:
  geometry/v4_perlang/shared_dims_{E,U}.json     full pairwise tables
  geometry/v4_perlang/shared_dims_{E,U}.npy      11×11 matrix of excess Σ cos²
  geometry/v4_perlang/var_capture_{E,U}.npy      11×11 matrix of var_capture(L_j ⊃ L_i)
  figures/v4_perlang/fig11_shared_dims_excess_{E,U}.{png,pdf}
  figures/v4_perlang/fig12_var_capture_{E,U}.{png,pdf}
  figures/v4_perlang/fig13_top_pairs_by_excess.{png,pdf}
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
V3 = SP / "artifacts" / "geometry" / "v4_perlang"
FIG = SP / "artifacts" / "figures" / "v4_perlang"
D = 4096


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    groups = json.loads((V3 / "groups.json").read_text())
    per_lang = json.loads((V3 / "per_lang_summary.json").read_text())
    LANGS = list(groups["languages"].keys())
    N = len(LANGS)
    out_summary = {}

    for matrix in ("E", "U"):
        print(f"\n=== {matrix} ===", flush=True)
        bases = {L: np.load(V3 / f"pc_basis_{L}_{matrix}.npy") for L in LANGS}
        eigs = {L: np.load(V3 / f"pc_eigvals_{L}_{matrix}.npy") for L in LANGS}
        ksig = {L: per_lang["languages"][f"{L}_{matrix}"]["spectrum"]["k_significant"]
                  for L in LANGS}
        mp_edge = {L: per_lang["languages"][f"{L}_{matrix}"]["spectrum"]["mp_upper_edge"]
                     for L in LANGS}

        sum_cos2 = np.zeros((N, N), dtype=np.float64)
        excess = np.zeros((N, N), dtype=np.float64)
        excess_per_min = np.zeros((N, N), dtype=np.float64)
        var_capture = np.zeros((N, N), dtype=np.float64)

        for i, Li in enumerate(LANGS):
            Bi = bases[Li]
            ei = eigs[Li]
            Ki = Bi.shape[0]
            for j, Lj in enumerate(LANGS):
                Bj = bases[Lj]
                Kj = Bj.shape[0]
                # principal-angle cosines: singular values of Bi Bj^T
                M = Bi @ Bj.T
                cos_angles = np.linalg.svd(M, compute_uv=False)
                k_min = min(Ki, Kj)
                s2 = float((cos_angles[:k_min] ** 2).sum())
                sum_cos2[i, j] = s2
                rand_baseline = Ki * Kj / D
                excess[i, j] = s2 - rand_baseline
                excess_per_min[i, j] = (s2 - rand_baseline) / k_min
                # variance capture of L_i by L_j: weighted by L_i's eigenvalues
                # ||B_j e_i,k||² = column-k norm squared of (M^T) = sum over rows of M^T squared = sum cols of M squared
                # Actually: B_j e_i,k = j-th projection of e_i,k. Compute (B_j @ B_i^T) — that's M^T. Each column k has ||B_j e_i,k||² = (M^T)[:, k].T @ (M^T)[:, k] = sum_l M[l,k]^2
                col_norm_sq = (M ** 2).sum(axis=0)   # shape (Kj... wait, M is (Ki, Kj), M^T is (Kj, Ki)).
                # Let me redo: B_j e_i,k = (B_j) @ (Bi^T)[:, k]) = M^T[:, k] where M^T = B_j B_i^T (Kj × Ki).
                # ||B_j e_i,k||² = sum_l (M^T[l, k])² = sum_l M[k, l]² = row-k norm squared of M.
                row_norm_sq = (M ** 2).sum(axis=1)   # shape (Ki,)
                # Weight by L_i's eigenvalues
                total_var_i = float(ei.sum())
                if total_var_i > 0:
                    var_capture[i, j] = float((ei * row_norm_sq).sum() / total_var_i)
                else:
                    var_capture[i, j] = 0.0

        np.save(V3 / f"shared_dims_excess_{matrix}.npy", excess)
        np.save(V3 / f"shared_dims_sum_cos2_{matrix}.npy", sum_cos2)
        np.save(V3 / f"var_capture_{matrix}.npy", var_capture)

        # ── Heatmaps ────────────────────────────────────────────────
        # Excess shared dims (corrected for K-baseline)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = excess.copy(); np.fill_diagonal(disp, np.nan)
        vmax = float(np.nanmax(np.abs(disp)))
        im = ax.imshow(disp, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(LANGS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(LANGS, fontsize=8)
        plt.colorbar(im, ax=ax, label="excess shared dims (Σ cos² − K_i K_j / d)")
        for i in range(N):
            for j in range(N):
                if i == j: continue
                v = excess[i, j]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                         fontsize=6, color="black" if abs(v) < vmax/2 else "white")
        ax.set_title(f"Shared dimensions above chance (Σ cos² − random) — {matrix}\n"
                      f"positive = subspaces share genuine structure, negative = below chance")
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"fig11_shared_dims_excess_{matrix}.{ext}",
                         dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Variance capture
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = var_capture.copy(); np.fill_diagonal(disp, np.nan)
        im = ax.imshow(disp, cmap="viridis",
                        vmin=0, vmax=float(np.nanmax(disp)))
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(LANGS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(LANGS, fontsize=8)
        plt.colorbar(im, ax=ax, label="fraction of L_i's variance captured by L_j's subspace")
        for i in range(N):
            for j in range(N):
                if i == j: continue
                v = var_capture[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6, color="white" if v < 0.5 else "black")
        ax.set_xlabel("L_j (containing subspace)")
        ax.set_ylabel("L_i (source variance)")
        ax.set_title(f"Variance capture: fraction of L_i's variance inside L_j's K_sig subspace — {matrix}")
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"fig12_var_capture_{matrix}.{ext}",
                         dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Top pairs ─────────────────────────────────────────────────
        # Excess shared dims, ranking upper-triangle pairs
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                Ki = ksig[LANGS[i]]
                Kj = ksig[LANGS[j]]
                pairs.append((LANGS[i], LANGS[j],
                              Ki, Kj,
                              sum_cos2[i, j], Ki * Kj / D,
                              excess[i, j], excess_per_min[i, j],
                              var_capture[i, j], var_capture[j, i]))
        out_summary[matrix] = {
            "labels": LANGS,
            "ksig": [ksig[L] for L in LANGS],
            "mp_upper_edge": [mp_edge[L] for L in LANGS],
            "shared_dims_excess_matrix": excess.tolist(),
            "var_capture_i_in_j_matrix": var_capture.tolist(),
        }

        # Print top-N pairs by excess
        print(f"\nTop 10 pairs by excess shared dims ({matrix}):")
        pairs.sort(key=lambda p: -p[6])
        for p in pairs[:10]:
            Li, Lj, Ki, Kj, sc, _, ex, ex_n, vc_ij, vc_ji = p
            kmin = min(Ki, Kj)
            kmax = max(Ki, Kj)
            print(f"  {Li:<22s} vs {Lj:<22s}  K=({Ki},{Kj})  Σcos²={sc:>6.1f}  "
                  f"random={Ki*Kj/D:>6.1f}  excess={ex:>+6.1f}  "
                  f"excess/K_min={ex_n:>+.3f}  var_cap_i→j={vc_ij:.3f}/var_cap_j→i={vc_ji:.3f}")

    (V3 / "shared_dims_summary.json").write_text(json.dumps(out_summary, indent=2))
    print("\n[done] shared-dims artefacts written")


if __name__ == "__main__":
    main()

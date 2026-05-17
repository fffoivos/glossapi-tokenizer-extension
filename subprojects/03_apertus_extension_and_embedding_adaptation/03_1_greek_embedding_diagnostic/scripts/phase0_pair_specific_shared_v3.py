"""v3 — pair-specific shared subspace via principal-angle decomposition + specificity check.

For each language pair (A, B) and each matrix:
  Step 1 — find the SHARED subspace between A and B.
    SVD(B_A B_B^T) = U Σ V^T  → cos(θ_k) = singular values, descending.
    Canonical directions in d-space:
      v_A,k = U^T_k @ B_A       (linear combo of A's PCs, lives in A's row space)
      v_B,k = V^T_k @ B_B       (lives in B's row space)
    "Shared direction" for the k-th angle: d_k = (v_A,k + v_B,k) / 2, renormalised.

  Step 2 — verify each shared direction is PAIR-SPECIFIC (not shared with others).
    For each language C: compute var_C(d_k) = d_k^T C_C d_k where C_C is
    rebuilt from C's PC basis + eigenvalues: var_C(d_k) = Σ_l λ_C,l (d_k · v_C,l)².
    Specificity score:
      s = min(var_A(d_k), var_B(d_k)) / max_{C ∉ {A,B}}( var_C(d_k) )

    s > 1  → genuinely pair-specific (A,B both high, others all lower)
    s ≈ 1  → universal direction
    s < 1  → there's a 3rd language that uses this direction even more strongly

  Step 3 — for the top pair-specific directions, decode exemplar tokens from A and B
    (top-20 by absolute projection onto d_k, separately per language).

Output:
  geometry/v3_perlang/pair_specific_{matrix}.json
    For every pair, the top-10 shared directions (with specificity scores +
    exemplar tokens from each language).
  figures/v3_perlang/fig14_pair_specificity_summary.png
    11×11 heatmap: count of pair-specific directions (s > 1) per pair.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
SP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic"
)
V3 = SP / "artifacts" / "geometry" / "v3_perlang"
FIG = SP / "artifacts" / "figures" / "v3_perlang"

COS_THRESHOLD = 0.5      # only consider directions with shared cosine > this
N_EXEMPLARS = 15
N_TOP_DIRECTIONS_TO_REPORT = 10


def load_decoder():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")


def var_along_from_rows(direction: np.ndarray, rows_centred: np.ndarray) -> float:
    """Empirical variance of language tokens along direction d.
    var(d) = mean((d · (M[c] − μ_C))²) over c ∈ language tokens.

    Uses the centred rows DIRECTLY — no truncation to K_sig PC basis.
    The previous version (`var_along(d, basis, eigvals)`) used only the
    saved K_sig PCs, which assigns zero variance to anything in C's
    sub-K_sig tail — biasing specificity scores upward. This fix
    eliminates that bias.
    """
    projs = rows_centred @ direction
    return float((projs ** 2).mean())


def main():
    groups = json.loads((V3 / "groups.json").read_text())
    LANGS = list(groups["languages"].keys())
    tok = load_decoder()

    summary_per_matrix = {}
    specificity_counts = {mat: np.zeros((len(LANGS), len(LANGS)), dtype=np.int32)
                            for mat in ("E", "U")}
    FIG.mkdir(parents=True, exist_ok=True)

    for matrix in ("E", "U"):
        print(f"\n=== {matrix} ===", flush=True)
        M = np.load(ROOT / "arrays" / f"{matrix}_fp32.npy", mmap_mode="r")

        bases = {L: np.load(V3 / f"pc_basis_{L}_{matrix}.npy") for L in LANGS}
        mus = {L: np.load(V3 / f"mu_{L}_{matrix}.npy") for L in LANGS}
        # Pre-load centred rows for every language so the row-based variance
        # check (replacing the v3 truncated-PC version) is fast.
        ids = {L: np.asarray(groups["languages"][L], dtype=np.int64) for L in LANGS}
        rows_centred = {L: (np.asarray(M[ids[L]], dtype=np.float64) - mus[L])
                          for L in LANGS}

        pair_records = []
        for i, A in enumerate(LANGS):
            for j, B in enumerate(LANGS):
                if j <= i: continue
                t0 = time.time()
                Ba, Bb = bases[A], bases[B]
                Ka, Kb = Ba.shape[0], Bb.shape[0]

                # Step 1 — SVD of Ba Bb^T
                Mab = Ba @ Bb.T          # (Ka, Kb)
                U, s_singular, Vt = np.linalg.svd(Mab, full_matrices=False)
                # Number of candidates above the cos-threshold
                n_candidates = int((s_singular > COS_THRESHOLD).sum())
                if n_candidates == 0:
                    continue

                # Canonical directions in d-space (rows are direction vectors)
                # v_A,k = U[:, k]^T @ Ba  =  (U.T @ Ba)[k, :]
                # v_B,k = Vt[k, :] @ Bb   =  (Vt @ Bb)[k, :]
                cv_A = (U.T @ Ba)[:n_candidates]        # (n_candidates, d)
                cv_B = (Vt @ Bb)[:n_candidates]

                # Shared direction d_k — average of cv_A,k and cv_B,k, then renorm
                d_shared = (cv_A + cv_B) / 2
                norms = np.linalg.norm(d_shared, axis=1, keepdims=True)
                d_shared = d_shared / np.maximum(norms, 1e-12)

                # Step 2 — specificity score per direction
                # var_C(d_k) computed from CENTRED ROWS DIRECTLY (not from
                # truncated PC basis). Honest empirical variance — no bias
                # from C's tail being truncated.
                var_per_lang = {}
                for L in LANGS:
                    # projs is (n_C, n_candidates) — vectorised across all directions
                    projs = rows_centred[L] @ d_shared.T
                    var_per_lang[L] = (projs ** 2).mean(axis=0)

                others = [L for L in LANGS if L not in (A, B)]
                var_A_arr = var_per_lang[A]
                var_B_arr = var_per_lang[B]
                others_max = np.max(np.stack([var_per_lang[L] for L in others]), axis=0)
                specificity = np.minimum(var_A_arr, var_B_arr) / np.maximum(others_max, 1e-30)

                n_specific = int((specificity > 1.0).sum())
                specificity_counts[matrix][i, j] = n_specific
                specificity_counts[matrix][j, i] = n_specific

                # Step 3 — top N pair-specific directions: rank by specificity score
                top_idx = np.argsort(-specificity)[:N_TOP_DIRECTIONS_TO_REPORT]
                dir_records = []
                rows_A = rows_centred[A]
                rows_B = rows_centred[B]
                for kk in top_idx:
                    d = d_shared[kk]
                    score = float(specificity[kk])
                    cos_angle = float(s_singular[kk])
                    var_a, var_b = float(var_A_arr[kk]), float(var_B_arr[kk])
                    # Exemplars from each language
                    proj_A = rows_A @ d
                    proj_B = rows_B @ d
                    top_A_idx = np.argsort(-np.abs(proj_A))[:N_EXEMPLARS]
                    top_B_idx = np.argsort(-np.abs(proj_B))[:N_EXEMPLARS]
                    ex_A = [{
                        "token_id": int(ids[A][i]),
                        "decoded": tok.decode([int(ids[A][i])]),
                        "score": float(proj_A[i]),
                    } for i in top_A_idx]
                    ex_B = [{
                        "token_id": int(ids[B][i]),
                        "decoded": tok.decode([int(ids[B][i])]),
                        "score": float(proj_B[i]),
                    } for i in top_B_idx]
                    others_max_lang = others[int(np.argmax(np.array([var_per_lang[L][kk] for L in others])))]
                    dir_records.append({
                        "principal_angle_index": int(kk),
                        "cos_principal_angle": cos_angle,
                        "specificity_score": score,
                        "var_A": var_a, "var_B": var_b,
                        "max_other_lang": others_max_lang,
                        "max_other_var": float(others_max[kk]),
                        "exemplars_A": ex_A,
                        "exemplars_B": ex_B,
                    })

                pair_records.append({
                    "A": A, "B": B,
                    "n_candidates_above_cos_threshold": n_candidates,
                    "n_pair_specific_directions": n_specific,
                    "top_directions": dir_records,
                })
                print(f"  {A:<22s} ↔ {B:<22s}  cand={n_candidates:>3d}  "
                      f"specific={n_specific:>3d}  ({time.time()-t0:.0f}s)", flush=True)

        summary_per_matrix[matrix] = {
            "labels": LANGS,
            "specificity_counts_matrix": specificity_counts[matrix].tolist(),
            "pair_records": pair_records,
        }

    (V3 / "pair_specific_shared_summary.json").write_text(
        json.dumps(summary_per_matrix, indent=2, ensure_ascii=False))

    # ── Plot: 11×11 heatmap of pair-specific direction counts ──────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, matrix in zip(axes, ("E", "U")):
        mat = specificity_counts[matrix]
        disp = mat.astype(float)
        np.fill_diagonal(disp, np.nan)
        im = ax.imshow(disp, cmap="viridis", vmin=0, vmax=np.nanmax(disp))
        ax.set_xticks(range(len(LANGS)))
        ax.set_yticks(range(len(LANGS)))
        ax.set_xticklabels(LANGS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(LANGS, fontsize=8)
        plt.colorbar(im, ax=ax, label="# pair-specific shared directions")
        for i in range(len(LANGS)):
            for j in range(len(LANGS)):
                if i == j: continue
                v = mat[i, j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                         color="white" if v < disp[~np.isnan(disp)].mean() else "black")
        ax.set_title(f"{matrix} matrix")
    fig.suptitle("Figure 14 — Count of pair-specific shared directions (specificity score > 1)")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig14_pair_specificity_count.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n[done]")


if __name__ == "__main__":
    main()

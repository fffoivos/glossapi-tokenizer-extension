"""v3 — language-discriminant directions via generalised eigendecomposition.

For each language L and each matrix:
  1. Build the in-group covariance C_L from L's centred rows directly.
  2. Build the background covariance C_bg from the union of all OTHER languages'
     tokens (10 languages).
  3. Add a small Tikhonov regulariser to C_bg to make it strictly invertible.
  4. Solve the generalised eigenproblem  C_L v = λ C_bg v.

Eigenvectors with the largest λ are the directions where L has unusually high
variance compared to the background — L's "most meaningful" directions.
Eigenvalues near 1 are directions L shares with the background; eigenvalues
near 0 are directions where L is *less* varied than the background.

For each top direction we save:
  - the eigenvalue λ (the "L-specificity strength")
  - the d-dim unit vector
  - the top-20 L tokens with the largest absolute projection onto v (the
    "exemplars" of this direction — decoded to readable text for inspection)

Output:
  geometry/v4_perlang/discriminant_{L}_{matrix}.npz
    eigvals, eigvecs (top-K)
  geometry/v4_perlang/discriminant_summary.json
    per-language top-10 directions with exemplar tokens + their scores
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
SP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic"
)
V3 = SP / "artifacts" / "geometry" / "v4_perlang"

K_OUT = 64
N_EXEMPLARS = 20
N_TOP_DIRECTIONS_TO_REPORT = 10


def cov_centred(rows: np.ndarray, mu: np.ndarray) -> np.ndarray:
    centred = rows - mu
    return centred.T @ centred / rows.shape[0]


def load_decoder():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")


def main():
    groups = json.loads((V3 / "groups.json").read_text())
    LANGS = list(groups["languages"].keys())
    tok = load_decoder()

    summary = {}
    for matrix in ("E", "U"):
        print(f"\n=== {matrix} ===", flush=True)
        M = np.load(ROOT / "arrays" / f"{matrix}_fp32.npy", mmap_mode="r")
        d = M.shape[1]

        for L in LANGS:
            t0 = time.time()
            ids_L = np.asarray(groups["languages"][L], dtype=np.int64)
            rows_L = np.asarray(M[ids_L], dtype=np.float64)
            mu_L = rows_L.mean(axis=0)
            C_L = cov_centred(rows_L, mu_L)

            # Background: union of all OTHER languages' token ids
            bg_set = set()
            for L2 in LANGS:
                if L2 != L:
                    bg_set.update(groups["languages"][L2])
            bg_set -= set(int(x) for x in ids_L)   # ensure disjoint
            bg_ids = np.asarray(sorted(bg_set), dtype=np.int64)
            rows_bg = np.asarray(M[bg_ids], dtype=np.float64)
            mu_bg = rows_bg.mean(axis=0)
            C_bg = cov_centred(rows_bg, mu_bg)

            # Tikhonov regularise C_bg to ensure positive definite
            eps = 1e-6 * np.trace(C_bg) / d
            C_bg_reg = C_bg + eps * np.eye(d)

            # Solve generalised eigenproblem: C_L v = λ C_bg v
            # Returns ascending; reverse for descending λ
            eigvals, eigvecs = eigh(C_L, C_bg_reg)
            order = np.argsort(-eigvals)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            # Keep top-K
            eigvals_top = eigvals[:K_OUT].astype(np.float64)
            eigvecs_top = eigvecs[:, :K_OUT].astype(np.float32)

            np.savez(
                V3 / f"discriminant_{L}_{matrix}.npz",
                eigvals=eigvals_top,
                eigvecs=eigvecs_top.T,        # rows are directions
                n_L=len(ids_L), n_bg=len(bg_ids), eps=eps,
            )

            # For each of the top-N reported directions, exemplar tokens within L
            centred_L = rows_L - mu_L
            scores = centred_L @ eigvecs_top    # (n_L, K_OUT)

            dir_records = []
            for k in range(N_TOP_DIRECTIONS_TO_REPORT):
                vec_scores = scores[:, k]
                ord_pos = np.argsort(-vec_scores)
                ord_neg = np.argsort(vec_scores)
                exemplars_pos = []
                exemplars_neg = []
                for i in ord_pos[:N_EXEMPLARS]:
                    tid = int(ids_L[i])
                    exemplars_pos.append({
                        "token_id": tid,
                        "decoded": tok.decode([tid]),
                        "score": float(vec_scores[i]),
                    })
                for i in ord_neg[:N_EXEMPLARS]:
                    tid = int(ids_L[i])
                    exemplars_neg.append({
                        "token_id": tid,
                        "decoded": tok.decode([tid]),
                        "score": float(vec_scores[i]),
                    })
                dir_records.append({
                    "rank": k + 1,
                    "eigenvalue": float(eigvals_top[k]),
                    "interpretation": (
                        "L-specific (λ >> 1)" if eigvals_top[k] > 5 else
                        ("L-specific (λ > 1)" if eigvals_top[k] > 1 else
                         ("shared with background" if eigvals_top[k] > 0.5 else
                          "L-suppressed (λ < 1)"))),
                    "exemplars_high_positive": exemplars_pos,
                    "exemplars_high_negative": exemplars_neg,
                })

            summary[f"{L}_{matrix}"] = {
                "language": L, "matrix": matrix,
                "n_L": len(ids_L), "n_background": len(bg_ids),
                "regulariser_eps": float(eps),
                "top_64_eigenvalues": eigvals_top.tolist(),
                "top_10_directions": dir_records,
            }
            print(f"  {L:<22s}  top-5 λ = {eigvals_top[:5].round(2).tolist()}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    (V3 / "discriminant_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[done]")


if __name__ == "__main__":
    main()

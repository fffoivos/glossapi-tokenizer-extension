"""v3 — per-language geometry + pairwise comparisons for 11 PMI-attributed languages.

For each (language, matrix ∈ {E, U}):
  §A centroid + global offset
  §B full d=4096 covariance eigendecomposition (Cov = X^T X / n)
     → eigenvalues, MP edge (Marchenko-Pastur), K_significant, top-PC shares,
        cumulative variance, participation ratio, shape anisotropy κ
  §C top-K_significant PC basis (for Mahalanobis + direction cosines)
  §D in-group Mahalanobis distribution + hull quantiles
  §E direction-cosine artifacts (cos to top-K PCs per token)

Pairwise (11×11 per matrix):
  §F centroid cosines + pooled-σ distances
  §G hull-overlap matrix: fraction of L_j tokens with m ≤ L_i's q25 (within L_i's subspace)
     — generalises the v2.1 "infiltrators" check

Output dir: artifacts/geometry/v4_perlang/
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
SP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic"
)
V3 = SP / "artifacts" / "geometry" / "v4_perlang"

QUANTILES = [10, 25, 50, 75, 90]


def mp_edge(eigvals: np.ndarray, n: int, d: int) -> tuple[float, float]:
    """Marchenko-Pastur upper edge for the empirical eigvalues of (1/n) X^T X
    where X is (n, d). MP aspect ratio c = d / n (NOT min/max — for n < d the
    nonzero spectrum still has upper edge σ² (1 + √c)² with c > 1).

    σ² estimated as the median of the lower-half of the empirical spectrum.
    This is a heuristic — for c ≫ 1 the MP density is heavily skewed and the
    median-of-bulk is a biased estimator; still, the same heuristic applied
    to every language gives a uniform comparison.
    """
    bulk = eigvals[eigvals < np.median(eigvals)]
    if bulk.size == 0:
        bulk = eigvals
    sigma_sq = float(np.median(bulk))
    c = d / n                               # MP aspect ratio: d ÷ samples
    edge = sigma_sq * (1 + np.sqrt(c)) ** 2
    return sigma_sq, float(edge)


def spectrum_stats(eigvals_sorted_desc: np.ndarray, n: int, d: int) -> dict:
    s = eigvals_sorted_desc
    sigma_sq, edge = mp_edge(s, n, d)
    k_sig = int((s > edge).sum())
    total_var = float(s.sum())
    cum_var = np.cumsum(s) / total_var if total_var > 0 else np.zeros_like(s)
    k_95pct = int(np.searchsorted(cum_var, 0.95)) + 1
    top1 = float(s[0] / s.sum()) if s.sum() > 0 else float("nan")
    top10 = float(s[:10].sum() / s.sum()) if s.sum() > 0 else float("nan")
    n_eig = s.size
    pr = float(((s.sum()) ** 2) / (n_eig * (s ** 2).sum())) if (s ** 2).sum() > 0 else float("nan")
    kappa = float(((n_eig * (s ** 2).sum()) / (s.sum() ** 2) - 1) / max(n_eig - 1, 1))
    return {
        "n": int(n), "d": int(d),
        "method": "full_cov_eigh" if n > d else "full_svd_rank_deficient",
        "n_eigvals_computed": int(n_eig),
        "sigma_sq_estimate": sigma_sq,
        "mp_upper_edge": edge,
        "k_significant": k_sig,
        "k_95pct_var": k_95pct,
        "total_var": total_var,
        "top_1_pc_share": top1,
        "top_10_pc_share": top10,
        "cumulative_variance_to_k_sig": float(cum_var[max(k_sig - 1, 0)]) if k_sig > 0 else float("nan"),
        "participation_ratio": pr,
        "shape_anisotropy_kappa": kappa,
    }


def compute_per_lang(M: np.ndarray, ids: np.ndarray, name: str, matrix: str) -> dict:
    n = ids.size
    d = M.shape[1]
    rows = np.asarray(M[ids], dtype=np.float64)
    mu = rows.mean(axis=0).astype(np.float64)
    centred = rows - mu

    if n < d:
        # rank-deficient: full SVD on centred / sqrt(n)
        _, sv, Vt = np.linalg.svd(centred / np.sqrt(n), full_matrices=False)
        eigvals = (sv ** 2).astype(np.float64)
        pc_basis_full = Vt
    else:
        Cov = (centred.T @ centred) / n
        eigvals_asc, eigvecs = np.linalg.eigh(Cov)
        # Sort descending
        idx = np.argsort(-eigvals_asc)
        eigvals = eigvals_asc[idx]
        pc_basis_full = eigvecs[:, idx].T   # (d, d) row = eigenvector

    stats = spectrum_stats(eigvals, n, d)
    # Use the formal MP K_sig for reporting but enforce a working-floor so
    # rank-deficient languages (Georgian K_sig=0 / Thai K_sig=1) still
    # produce a usable PC basis for downstream pairwise comparisons.
    K_FLOOR = 32
    k_sig = max(stats["k_significant"], K_FLOOR)
    stats["k_significant_formal"] = int(stats["k_significant"])
    stats["k_working"] = int(k_sig)
    stats["k_floor_applied"] = int(stats["k_significant"]) < K_FLOOR
    pc_basis = pc_basis_full[:k_sig].astype(np.float32)
    eig_sig = eigvals[:k_sig].astype(np.float32)

    # Mahalanobis in the K_sig subspace
    z = (rows - mu) @ pc_basis.T            # (n, K_sig)
    inv_lambda = 1.0 / np.maximum(eig_sig, 1e-12)
    m_sq = (z ** 2) * inv_lambda
    m = np.sqrt(m_sq.sum(axis=1))

    # In-group hull quantiles
    quantile_values = {f"q{q}": float(np.percentile(m, q)) for q in QUANTILES}
    quantile_values["mean"] = float(m.mean())
    quantile_values["std"] = float(m.std())
    quantile_values["min"] = float(m.min())
    quantile_values["max"] = float(m.max())

    # Direction cosines (per token vs top-K PCs)
    deltas = rows - mu
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    units = deltas / np.maximum(norms, 1e-12)
    cos_all = (units @ pc_basis.T).astype(np.float32)

    # Save artefacts
    np.save(V3 / f"mu_{name}_{matrix}.npy", mu.astype(np.float32))
    np.save(V3 / f"pc_basis_{name}_{matrix}.npy", pc_basis)
    np.save(V3 / f"pc_eigvals_{name}_{matrix}.npy", eig_sig)
    np.savez(V3 / f"distance_{name}_{matrix}.npz",
              ids=ids, mahalanobis=m, euclid=norms.ravel())
    np.savez(V3 / f"direction_cosines_{name}_{matrix}.npz",
              ids=ids, cos_top_k=cos_all)
    np.savez(V3 / f"spectrum_{name}_{matrix}.npz", eigvals=eigvals.astype(np.float64))

    return {
        "name": name, "matrix": matrix, "spectrum": stats,
        "hull_quantiles": quantile_values,
        "median_row_norm": float(np.median(np.linalg.norm(rows, axis=1))),
        "centroid_norm": float(np.linalg.norm(mu)),
    }


def main():
    V3.mkdir(parents=True, exist_ok=True)
    groups = json.loads((V3 / "groups.json").read_text())
    LANGS = list(groups["languages"].keys())
    print(f"Processing {len(LANGS)} languages × 2 matrices", flush=True)

    summary = {"languages": {}, "pairwise": {}}
    mus = {"E": {}, "U": {}}
    pcs = {"E": {}, "U": {}}
    eigs = {"E": {}, "U": {}}
    qs = {"E": {}, "U": {}}

    for matrix in ("E", "U"):
        print(f"\n=== {matrix} matrix ===", flush=True)
        M = np.load(ROOT / "arrays" / f"{matrix}_fp32.npy", mmap_mode="r")
        d = M.shape[1]
        for lang in LANGS:
            ids = np.asarray(groups["languages"][lang], dtype=np.int64)
            t0 = time.time()
            r = compute_per_lang(M, ids, lang, matrix)
            mus[matrix][lang] = np.load(V3 / f"mu_{lang}_{matrix}.npy")
            pcs[matrix][lang] = np.load(V3 / f"pc_basis_{lang}_{matrix}.npy")
            eigs[matrix][lang] = np.load(V3 / f"pc_eigvals_{lang}_{matrix}.npy")
            qs[matrix][lang] = r["hull_quantiles"]
            print(f"  {lang:<22s} n={r['spectrum']['n']:>5d}  "
                  f"K_sig={r['spectrum']['k_significant']:>4d}  "
                  f"PR={r['spectrum']['participation_ratio']:.3f}  "
                  f"κ={r['spectrum']['shape_anisotropy_kappa']:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            summary["languages"][f"{lang}_{matrix}"] = r

    # Pairwise (centroid + hull overlap)
    print("\n=== Pairwise comparisons ===", flush=True)
    for matrix in ("E", "U"):
        M = np.load(ROOT / "arrays" / f"{matrix}_fp32.npy", mmap_mode="r")
        cent_cos = np.zeros((len(LANGS), len(LANGS)), dtype=np.float32)
        cent_dist = np.zeros((len(LANGS), len(LANGS)), dtype=np.float32)
        hull_overlap = np.zeros((len(LANGS), len(LANGS)), dtype=np.float32)   # frac of j inside i's q25
        for i, Li in enumerate(LANGS):
            mu_i = mus[matrix][Li]
            pc_i = pcs[matrix][Li]
            eig_i = eigs[matrix][Li]
            q25_i = qs[matrix][Li]["q25"]
            for j, Lj in enumerate(LANGS):
                mu_j = mus[matrix][Lj]
                # centroid cosine + pooled-σ distance
                num = float(mu_i @ mu_j)
                cn = float(np.linalg.norm(mu_i) * np.linalg.norm(mu_j)) + 1e-12
                cent_cos[i, j] = num / cn
                cent_dist[i, j] = float(np.linalg.norm(mu_i - mu_j))
                # hull overlap: project L_j's tokens into L_i's K_sig subspace, compute Mahalanobis to μ_i
                ids_j = np.asarray(groups["languages"][Lj], dtype=np.int64)
                rows_j = np.asarray(M[ids_j], dtype=np.float64)
                z = (rows_j - mu_i) @ pc_i.T
                inv_lambda = 1.0 / np.maximum(eig_i, 1e-12)
                m_j_into_i = np.sqrt(((z ** 2) * inv_lambda).sum(axis=1))
                hull_overlap[i, j] = float((m_j_into_i <= q25_i).mean())
        np.save(V3 / f"pairwise_centroid_cosine_{matrix}.npy", cent_cos)
        np.save(V3 / f"pairwise_centroid_distance_{matrix}.npy", cent_dist)
        np.save(V3 / f"pairwise_hull_overlap_{matrix}.npy", hull_overlap)
        # JSON labels
        summary["pairwise"][matrix] = {
            "labels": LANGS,
            "centroid_cosine": cent_cos.tolist(),
            "centroid_distance": cent_dist.tolist(),
            "hull_overlap_frac_j_within_i_q25": hull_overlap.tolist(),
        }
        print(f"  {matrix} pairwise matrices done", flush=True)

    (V3 / "per_lang_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[done] artefacts under artifacts/geometry/v4_perlang/")


if __name__ == "__main__":
    main()

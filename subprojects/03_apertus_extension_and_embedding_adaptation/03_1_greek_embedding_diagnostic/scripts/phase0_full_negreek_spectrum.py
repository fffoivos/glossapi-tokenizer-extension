"""Recompute the ¬Greek spectrum FULLY (all 4096 eigenvalues) via the
4096×4096 covariance eigendecomposition.

The original phase0_greek_vs_not_geometry.py used randomized_svd(n_components=500)
for ¬Greek (n=126,990 ≫ d=4,096). Stats derived from that truncated slice
(sigma_sq estimate, MP edge, K_significant, total_var, top-K share fractions,
participation ratio, shape anisotropy κ) are biased:
  - sigma_sq taken from the median of a top-500 spectrum overestimates the
    noise floor (truly-noisy directions are absent from the slice).
  - PR and κ depend on Σλ and Σλ² over the full spectrum, not the slice.
  - top-1/top-10 PC share is divided by total_var, which the truncated sum
    underestimates.

This script computes the d×d covariance Cov = X^T X / n where X is the
centred ¬Greek embedding block, then eigh — gives all 4096 eigenvalues.
Greek's spectrum is already full (n < d, computed by full_svd).

Overwrites geometry/v2/spectrum_not_greek_{E,U}.{json,npz} and updates
cross_group_summary.json with the corrected ¬Greek block.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
V2 = ROOT / "geometry" / "v2"


def mp_edge_from_full_spectrum(eigvals: np.ndarray, n: int, d: int) -> tuple[float, float]:
    """sigma_sq from the bulk of the empirical eigenvalues + MP edge for n×d real matrix."""
    # Bulk = eigenvalues plausibly in the MP bulk (below the median is mostly bulk
    # for high-rank cases, since spikes are the top tail).
    bulk = eigvals[eigvals < np.median(eigvals)]
    if bulk.size == 0:
        bulk = eigvals
    sigma_sq = float(np.median(bulk))
    q = min(d, n) / max(d, n)
    edge = sigma_sq * (1 + np.sqrt(q)) ** 2
    return sigma_sq, float(edge)


def spectrum_stats(eigvals: np.ndarray, n: int, d: int) -> dict:
    n_eig = eigvals.size
    sigma_sq, edge = mp_edge_from_full_spectrum(eigvals, n, d)
    k_sig = int((eigvals > edge).sum())
    total_var = float(eigvals.sum())
    cum_var = np.cumsum(eigvals[::-1])[::-1]  # ensure descending order
    # Actually re-sort descending to be sure
    s = np.sort(eigvals)[::-1]
    cum_var = np.cumsum(s) / s.sum() if s.sum() > 0 else np.zeros_like(s)
    k_95pct = int(np.searchsorted(cum_var, 0.95)) + 1
    top1_share = float(s[0] / s.sum()) if s.sum() > 0 else float("nan")
    top10_share = float(s[:10].sum() / s.sum()) if s.sum() > 0 else float("nan")
    pr = float(((s.sum()) ** 2) / (n_eig * (s ** 2).sum()))
    kappa = float(((n_eig * (s ** 2).sum()) / (s.sum() ** 2) - 1) / max(n_eig - 1, 1))
    return {
        "n": int(n), "d": int(d), "method": "full_cov_eigh",
        "n_eigvals_computed": int(n_eig),
        "sigma_sq_estimate": sigma_sq,
        "mp_upper_edge": edge,
        "k_significant": k_sig,
        "k_95pct_var": k_95pct,
        "total_var": total_var,
        "top_1_pc_share": top1_share,
        "top_10_pc_share": top10_share,
        "cumulative_variance_to_k_sig": float(cum_var[max(k_sig - 1, 0)]) if k_sig > 0 else float("nan"),
        "participation_ratio": pr,
        "shape_anisotropy_kappa": kappa,
    }, s.astype(np.float64)


def process(matrix: str):
    print(f"\n=== {matrix} ===", flush=True)
    M = np.load(ROOT / "arrays" / f"{matrix}_fp32.npy", mmap_mode="r")
    groups = json.loads((ROOT / "geometry" / "groups_greek_vs_not.json").read_text())
    not_ids = np.asarray(groups["not_Greek"], dtype=np.int64)
    mu = np.load(V2 / f"mu_not_greek_{matrix}.npy")

    n = int(not_ids.size)
    d = int(mu.size)
    print(f"  shape ({matrix}, ¬Greek): n={n}, d={d}", flush=True)

    t0 = time.time()
    # Build the d×d covariance — much cheaper than n×d SVD when d ≪ n.
    # Stream the dot product in float64 to avoid mantissa loss.
    rows = np.asarray(M[not_ids], dtype=np.float64)
    rows -= mu.astype(np.float64)
    print(f"  centred matrix prepared in {time.time()-t0:.1f}s", flush=True)
    t1 = time.time()
    Cov = (rows.T @ rows) / n        # (d, d)
    print(f"  Cov X^T X / n done in {time.time()-t1:.1f}s; symmetric={np.allclose(Cov, Cov.T, atol=1e-8)}", flush=True)
    t2 = time.time()
    eigvals = np.linalg.eigvalsh(Cov)   # all d eigenvalues, ascending
    print(f"  eigh done in {time.time()-t2:.1f}s; n_eigvals={eigvals.size}; "
          f"min={eigvals.min():.6f}, max={eigvals.max():.4f}", flush=True)

    stats, sorted_desc = spectrum_stats(eigvals, n, d)
    print(f"  stats: K_sig={stats['k_significant']}, MP_edge={stats['mp_upper_edge']:.4f}, "
          f"sigma_sq={stats['sigma_sq_estimate']:.4f}, top_1={stats['top_1_pc_share']:.4f}, "
          f"PR={stats['participation_ratio']:.4f}", flush=True)

    # Write outputs (overwrite truncated v2 files)
    np.savez(V2 / f"spectrum_not_greek_{matrix}.npz", eigvals=sorted_desc)
    (V2 / f"spectrum_not_greek_{matrix}.json").write_text(json.dumps(stats, indent=2))
    print(f"  wrote spectrum_not_greek_{matrix}.{{json,npz}} (FULL spectrum)", flush=True)
    return stats


def main():
    out = {}
    for m in ("E", "U"):
        out[m] = process(m)
    # Update cross_group_summary.json (¬Greek block only)
    summary = json.loads((V2 / "cross_group_summary.json").read_text())
    for m in ("E", "U"):
        summary["matrices"][m]["not_greek_spectrum"] = out[m]
    (V2 / "cross_group_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[done] ¬Greek spectrum overwritten; cross_group_summary.json refreshed.")


if __name__ == "__main__":
    main()

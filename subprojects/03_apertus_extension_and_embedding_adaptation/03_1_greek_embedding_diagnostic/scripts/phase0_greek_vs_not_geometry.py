"""v2 §3.1-§3.6 + §3.9 + §3.10 — Greek-vs-¬Greek manifold diagnostic.

Inputs (already on disk):
  arrays/{E_fp32,U_fp32}.npy
  geometry/groups_greek_vs_not.json

Outputs (under geometry/v2/):
  centroids.json                  per-group + global centroids (norms, distances)
  distance_{E,U}.npz              per-token Euclidean + Mahalanobis (group own)
  spectrum_{group}_{E,U}.npz      eigenvalues (full or top-N) + MP edge
  spectrum_{group}_{E,U}.json     MP summary, K_sig, PR, anisotropy
  hull_{E,U}.json                 within-group hull occupancy + top-20 outliers
  infiltrators_{E,U}.json         §3.9 + §3.10 — ¬Greek into Greek hull (top-1000)
  cross_group_summary.csv         flat per-(group, matrix) table

§3.5 MP edge formula:
  Marchenko-Pastur upper edge for an n × d real Gaussian matrix:
  λ_+ = σ² × (1 + sqrt(q))²  where q = min(d, n) / max(d, n)
  σ² estimated as the median of the lower-half of the empirical
  eigenvalue spectrum.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
ARRAY_DIR = ROOT / "arrays"
GEOM_DIR = ROOT / "geometry"
V2_DIR = GEOM_DIR / "v2"
CLASS_PATH = Path(
    "/home/foivos/runs/apertus_greek_diagnostic_20260511_v2/token_classification.jsonl"
)

K_PC_NOT_GREEK = 500    # cap for the rank-d-deficient ¬Greek spectrum
TOP_N_INFILTRATORS = 1000
TOP_N_OUTLIERS = 20
SEED = 20260512


def load_decoded(ids: list[int]) -> dict[int, dict]:
    want = set(ids)
    out: dict[int, dict] = {}
    with CLASS_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            tid = int(r["id"])
            if tid in want:
                out[tid] = {
                    "raw_token": r.get("raw_token", ""),
                    "decoded_text": r.get("decoded_text", ""),
                }
                if len(out) == len(want):
                    break
    return out


def mp_edge(eigvals: np.ndarray, n: int, d: int) -> tuple[float, float]:
    """Return (sigma_sq, mp_upper_edge) — MP estimator from the empirical spectrum."""
    bulk = eigvals[eigvals < np.median(eigvals) * 2]      # rough bulk extraction
    if bulk.size == 0:
        bulk = eigvals
    sigma_sq = float(np.median(bulk))
    q = min(d, n) / max(d, n)
    edge = sigma_sq * (1 + np.sqrt(q)) ** 2
    return sigma_sq, float(edge)


def compute_spectrum_full(X: np.ndarray, name: str) -> dict:
    """Centred X has shape (n, d). Returns the per-group spectrum + meta.

    For n ≤ d (rank-deficient case, e.g. Greek): full SVD via numpy.
    For n > d (full-rank): randomised SVD on top-K_PC_NOT_GREEK is enough
    for the MP edge + significant-PC count (the bulk eigenvalues we don't
    explicitly compute are all below the MP edge by construction).
    """
    n, d = X.shape
    print(f"    [{name}] shape={X.shape}", flush=True)
    t0 = time.time()
    if n <= d:
        # full SVD; only n singular values are nonzero
        _, sv, _ = np.linalg.svd(X / np.sqrt(n), full_matrices=False)
        eigvals = (sv ** 2).astype(np.float64)
        method = "full_svd"
        k_computed = n
    else:
        k = min(K_PC_NOT_GREEK, d - 1)
        _, sv, _ = randomized_svd(X / np.sqrt(n), n_components=k,
                                   random_state=SEED, n_iter=5)
        eigvals = (sv ** 2).astype(np.float64)
        method = f"randomized_svd_top_{k}"
        k_computed = k
    print(f"    [{name}] {method} done in {time.time()-t0:.1f}s; "
          f"computed {k_computed} eigenvalues", flush=True)

    sigma_sq, edge = mp_edge(eigvals, n, d)
    k_sig = int((eigvals > edge).sum())

    total_var = float(eigvals.sum())
    cum_var = np.cumsum(eigvals) / total_var if total_var > 0 else np.zeros_like(eigvals)
    top1_share = float(eigvals[0] / total_var) if total_var > 0 else float("nan")
    top10_share = float(eigvals[:10].sum() / total_var) if total_var > 0 else float("nan")
    k_95pct = int(np.searchsorted(cum_var, 0.95)) + 1
    # participation ratio (use the eigvals we have; for full-rank case
    # this is an upper bound on PR — the noise tail we didn't compute
    # would lower it slightly)
    n_eig = eigvals.size
    pr = float(((eigvals.sum()) ** 2) / (n_eig * (eigvals ** 2).sum()))
    kappa = float(((n_eig * (eigvals ** 2).sum()) / (eigvals.sum() ** 2) - 1)
                   / max(n_eig - 1, 1))
    return {
        "n": int(n), "d": int(d), "method": method,
        "n_eigvals_computed": int(k_computed),
        "eigvals": eigvals,
        "sigma_sq_estimate": sigma_sq,
        "mp_upper_edge": edge,
        "k_significant": k_sig,
        "k_95pct_var": k_95pct,
        "total_var": total_var,
        "top_1_pc_share": top1_share,
        "top_10_pc_share": top10_share,
        "cumulative_variance_to_k_sig": float(cum_var[max(k_sig - 1, 0)]) if eigvals.size else float("nan"),
        "participation_ratio": pr,
        "shape_anisotropy_kappa": kappa,
    }


def hull_from_mahalanobis(m: np.ndarray) -> dict:
    """Empirical hull occupancy stats; σ = std of in-group Mahalanobis."""
    sigma = float(np.std(m))
    if sigma == 0:
        sigma = 1.0
    return {
        "mahalanobis_std": sigma,
        "frac_within_0_5_sigma": float((m <= 0.5 * sigma).mean()),
        "frac_within_1_sigma":   float((m <= 1.0 * sigma).mean()),
        "frac_within_2_sigma":   float((m <= 2.0 * sigma).mean()),
        "frac_within_3_sigma":   float((m <= 3.0 * sigma).mean()),
    }


def mahalanobis_in_subspace(rows: np.ndarray, mu: np.ndarray,
                             pc_basis: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance using the given PC basis + eigenvalues."""
    z = (rows - mu) @ pc_basis.T            # (n, K)
    inv_lambda = 1.0 / np.maximum(eigvals, 1e-12)
    m_sq = (z ** 2) * inv_lambda            # (n, K)
    return np.sqrt(m_sq.sum(axis=1))


def top_pc_contributions(rows: np.ndarray, mu: np.ndarray,
                          pc_basis: np.ndarray, eigvals: np.ndarray,
                          n_top: int = 3) -> np.ndarray:
    """For each row, return the top-n_top PC indices contributing most
    to its Mahalanobis distance."""
    z = (rows - mu) @ pc_basis.T
    contrib = (z ** 2) / np.maximum(eigvals, 1e-12)
    return np.argsort(-contrib, axis=1)[:, :n_top]


def process_matrix(M: np.ndarray, name: str, groups: dict, decoded: dict):
    """Compute everything for one matrix."""
    print(f"\n=== {name} ===", flush=True)
    M = np.asarray(M)
    greek_ids = np.asarray(groups["Greek"], dtype=np.int64)
    not_ids = np.asarray(groups["not_Greek"], dtype=np.int64)
    all_ids = np.asarray(groups["all_classified"], dtype=np.int64)

    # Centroids
    mu_classified = M[all_ids].mean(axis=0).astype(np.float32)
    mu_greek = M[greek_ids].mean(axis=0).astype(np.float32)
    mu_not = M[not_ids].mean(axis=0).astype(np.float32)
    centroids = {
        "all_classified_norm": float(np.linalg.norm(mu_classified)),
        "greek_norm": float(np.linalg.norm(mu_greek)),
        "not_greek_norm": float(np.linalg.norm(mu_not)),
        "greek_distance_to_global": float(np.linalg.norm(mu_greek - mu_classified)),
        "not_greek_distance_to_global": float(np.linalg.norm(mu_not - mu_classified)),
        "greek_vs_not_greek_distance": float(np.linalg.norm(mu_greek - mu_not)),
        "greek_median_row_norm": float(np.median(np.linalg.norm(M[greek_ids], axis=1))),
        "not_greek_median_row_norm": float(np.median(np.linalg.norm(M[not_ids], axis=1))),
    }
    np.save(V2_DIR / f"mu_greek_{name}.npy", mu_greek)
    np.save(V2_DIR / f"mu_not_greek_{name}.npy", mu_not)
    np.save(V2_DIR / f"mu_classified_{name}.npy", mu_classified)
    print(f"  centroids:", centroids, flush=True)

    # Per-group spectrum
    print("  Greek spectrum:", flush=True)
    X_g = (M[greek_ids] - mu_greek).astype(np.float64)
    spec_g = compute_spectrum_full(X_g, "Greek")
    print("  ¬Greek spectrum:", flush=True)
    X_n = (M[not_ids] - mu_not).astype(np.float64)
    spec_n = compute_spectrum_full(X_n, "not_Greek")

    # Save spectra
    np.savez(V2_DIR / f"spectrum_greek_{name}.npz", eigvals=spec_g["eigvals"])
    np.savez(V2_DIR / f"spectrum_not_greek_{name}.npz", eigvals=spec_n["eigvals"])
    spec_g_json = {k: v for k, v in spec_g.items() if k != "eigvals"}
    spec_n_json = {k: v for k, v in spec_n.items() if k != "eigvals"}
    (V2_DIR / f"spectrum_greek_{name}.json").write_text(json.dumps(spec_g_json, indent=2))
    (V2_DIR / f"spectrum_not_greek_{name}.json").write_text(json.dumps(spec_n_json, indent=2))

    # We need the PC basis for Mahalanobis. Recompute the basis (with right-singular-vector matrix)
    # for both groups at K_sig.
    k_sig_g = max(spec_g["k_significant"], 1)
    k_sig_n = min(max(spec_n["k_significant"], 1), 500)
    print(f"  Greek K_sig: {k_sig_g}; ¬Greek K_sig: {k_sig_n}", flush=True)

    # Greek basis: redo SVD to get right-singular vectors (numpy.linalg.svd above
    # returned them as Vt; for ¬Greek we need a fresh randomised_svd call)
    Ug, Sg, Vg = np.linalg.svd(X_g / np.sqrt(X_g.shape[0]), full_matrices=False)
    pc_greek = Vg[:k_sig_g].astype(np.float32)        # (K_sig_g, d)
    eig_greek = (Sg[:k_sig_g] ** 2).astype(np.float32)
    np.save(V2_DIR / f"pc_basis_greek_{name}.npy", pc_greek)
    np.save(V2_DIR / f"pc_eigvals_greek_{name}.npy", eig_greek)

    Un, Sn, Vn = randomized_svd(X_n / np.sqrt(X_n.shape[0]),
                                 n_components=k_sig_n,
                                 random_state=SEED, n_iter=5)
    pc_not = Vn.astype(np.float32)
    eig_not = (Sn ** 2).astype(np.float32)
    np.save(V2_DIR / f"pc_basis_not_greek_{name}.npy", pc_not)
    np.save(V2_DIR / f"pc_eigvals_not_greek_{name}.npy", eig_not)

    # §3.2 distance distributions (within-group, own metric)
    m_greek_own = mahalanobis_in_subspace(M[greek_ids], mu_greek, pc_greek, eig_greek)
    eu_greek = np.linalg.norm(M[greek_ids] - mu_greek, axis=1)
    m_not_own = mahalanobis_in_subspace(M[not_ids], mu_not, pc_not, eig_not)
    eu_not = np.linalg.norm(M[not_ids] - mu_not, axis=1)
    np.savez(V2_DIR / f"distance_greek_{name}.npz",
              ids=greek_ids, euclid=eu_greek, mahalanobis=m_greek_own)
    np.savez(V2_DIR / f"distance_not_greek_{name}.npz",
              ids=not_ids, euclid=eu_not, mahalanobis=m_not_own)

    # §3.4 hull occupancy + top-20 outliers (within-Greek)
    greek_hull = hull_from_mahalanobis(m_greek_own)
    outlier_idx = np.argsort(-m_greek_own)[:TOP_N_OUTLIERS]
    top_pcs_outliers = top_pc_contributions(M[greek_ids[outlier_idx]],
                                             mu_greek, pc_greek, eig_greek)
    outliers = []
    for rank, idx in enumerate(outlier_idx):
        tid = int(greek_ids[idx])
        dec = decoded.get(tid, {})
        outliers.append({
            "rank": rank + 1, "id": tid,
            "raw_token": dec.get("raw_token", ""),
            "decoded_text": dec.get("decoded_text", ""),
            "mahalanobis": float(m_greek_own[idx]),
            "euclid": float(eu_greek[idx]),
            "top_3_pcs": [int(x) for x in top_pcs_outliers[rank]],
        })
    greek_hull["top_20_outliers"] = outliers
    (V2_DIR / f"hull_greek_{name}.json").write_text(json.dumps(greek_hull, indent=2))

    # §3.9 + §3.10 infiltrators: ¬Greek tokens scored against Greek's hull
    print("  infiltrators (¬Greek into Greek hull):", flush=True)
    m_not_into_greek = mahalanobis_in_subspace(M[not_ids], mu_greek, pc_greek, eig_greek)
    sigma_g = greek_hull["mahalanobis_std"]
    inf_summary = {
        "matrix": name,
        "k_sig_used": int(k_sig_g),
        "mahalanobis_std_greek_in_group": float(sigma_g),
        "frac_negreek_within_0_5_sigma": float((m_not_into_greek <= 0.5 * sigma_g).mean()),
        "frac_negreek_within_1_sigma":   float((m_not_into_greek <= 1.0 * sigma_g).mean()),
        "frac_negreek_within_2_sigma":   float((m_not_into_greek <= 2.0 * sigma_g).mean()),
        "frac_negreek_within_3_sigma":   float((m_not_into_greek <= 3.0 * sigma_g).mean()),
        "count_negreek_within_1_sigma": int((m_not_into_greek <= 1.0 * sigma_g).sum()),
        "count_negreek_within_2_sigma": int((m_not_into_greek <= 2.0 * sigma_g).sum()),
    }

    # Top-N infiltrators
    top_n = min(TOP_N_INFILTRATORS, m_not_into_greek.size)
    infil_idx = np.argsort(m_not_into_greek)[:top_n]
    top_pcs_infil = top_pc_contributions(M[not_ids[infil_idx]], mu_greek,
                                          pc_greek, eig_greek)
    # Decode infiltrator ids (need to load on demand — we only loaded outliers' ids earlier)
    decoded_inf = load_decoded([int(not_ids[i]) for i in infil_idx])
    src_lookup = groups["source_group_of_negreek"]
    src_counts: dict[str, int] = {}
    infiltrators_top1000 = []
    for rank, idx in enumerate(infil_idx):
        tid = int(not_ids[idx])
        src = src_lookup.get(str(tid), src_lookup.get(tid, "?"))
        dec = decoded_inf.get(tid, {})
        m = float(m_not_into_greek[idx])
        infiltrators_top1000.append({
            "rank": rank + 1, "id": tid,
            "raw_token": dec.get("raw_token", ""),
            "decoded_text": dec.get("decoded_text", ""),
            "source_group": src,
            "mahalanobis_to_greek": m,
            "top_3_pcs": [int(x) for x in top_pcs_infil[rank]],
        })
        if m <= sigma_g:
            src_counts[src] = src_counts.get(src, 0) + 1
    inf_summary["count_within_1_sigma_by_source_group"] = src_counts
    # Top-20 visible at top-level
    inf_summary["top_20_infiltrators"] = infiltrators_top1000[:20]
    (V2_DIR / f"infiltrators_{name}.json").write_text(json.dumps(inf_summary, indent=2))

    # Write top-1000 CSV
    with (V2_DIR / f"infiltrators_top1000_{name}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "raw_token", "decoded_text", "source_group",
                    "mahalanobis_to_greek"])
        for r in infiltrators_top1000:
            w.writerow([r["rank"], r["id"], r["raw_token"], r["decoded_text"],
                        r["source_group"], r["mahalanobis_to_greek"]])
    # Save percentile mapping: where do the 1000 sit in Greek's in-group Mahalanobis?
    greek_sorted = np.sort(m_greek_own)
    pcts = np.searchsorted(greek_sorted, m_not_into_greek[infil_idx]) / max(greek_sorted.size, 1) * 100
    np.savez(V2_DIR / f"infiltrators_top1000_{name}_distance.npz",
              ids=not_ids[infil_idx],
              mahalanobis_to_greek=m_not_into_greek[infil_idx],
              percentile_in_greek_distribution=pcts)

    return {
        "matrix": name,
        "centroids": centroids,
        "greek_spectrum": spec_g_json,
        "not_greek_spectrum": spec_n_json,
        "greek_hull": {k: v for k, v in greek_hull.items() if k != "top_20_outliers"},
        "infiltrators_summary": {k: v for k, v in inf_summary.items()
                                  if k not in ("top_20_infiltrators",)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", choices=["E", "U", "both"], default="both")
    args = ap.parse_args()

    V2_DIR.mkdir(parents=True, exist_ok=True)
    groups = json.loads((GEOM_DIR / "groups_greek_vs_not.json").read_text())
    print(f"groups: Greek n={groups['n_greek']}, ¬Greek n={groups['n_not_greek']}",
          flush=True)

    # Pre-load outlier decoded info (greek-only) — small
    decoded_greek = load_decoded(list(groups["Greek"]))

    summary = {"strict_greek_n": groups["n_greek"], "matrices": {}}
    if args.matrix in ("E", "both"):
        E = np.load(ARRAY_DIR / "E_fp32.npy", mmap_mode="r")
        summary["matrices"]["E"] = process_matrix(E, "E", groups, decoded_greek)
    if args.matrix in ("U", "both"):
        U = np.load(ARRAY_DIR / "U_fp32.npy", mmap_mode="r")
        summary["matrices"]["U"] = process_matrix(U, "U", groups, decoded_greek)

    (V2_DIR / "cross_group_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[done] wrote per-matrix artefacts under geometry/v2/", flush=True)


if __name__ == "__main__":
    main()

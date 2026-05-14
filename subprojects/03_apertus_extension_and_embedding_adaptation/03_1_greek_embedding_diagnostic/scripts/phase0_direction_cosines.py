"""v2 §3.3 — direction-cosine artifacts for Greek and ¬Greek.

For each token in a group:
  v_t = (M[t] − μ_g) / ‖M[t] − μ_g‖          (unit direction)
  c_t,k = v_t · e_g,k                         (cosine to k-th PC of group)

Save per-token cosines for k = 1..K_used (top-K_significant per group).
Also produce two summaries: |cos(v_t, top-1 PC)| distribution + mean
|cos| heatmap across top-K PCs (the Fig 4 + Fig 5 ingredients).

Outputs:
  geometry/v2/direction_cosines_{group}_{matrix}.npz
  geometry/v2/direction_cosines_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
V2 = ROOT / "geometry" / "v2"

K_HEATMAP = 32  # number of top PCs to summarise in the heatmap


def process_group(M: np.ndarray, ids: np.ndarray, mu: np.ndarray,
                   pc_basis: np.ndarray, group_name: str, matrix: str):
    print(f"  {group_name} {matrix}: ids n={ids.size}, K_used={pc_basis.shape[0]}", flush=True)
    rows = np.asarray(M[ids])
    deltas = rows - mu                              # (n, d)
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    units = deltas / np.maximum(norms, 1e-12)       # (n, d) unit vectors
    # Project onto top-K PC axes
    K = pc_basis.shape[0]
    cos_all = units @ pc_basis.T                     # (n, K)
    np.savez(V2 / f"direction_cosines_{group_name}_{matrix}.npz",
              ids=ids, delta_norms=norms.ravel(),
              cos_top_k=cos_all.astype(np.float32))
    # Summary
    abs_cos = np.abs(cos_all)
    return {
        "group": group_name, "matrix": matrix, "n": int(ids.size),
        "K_used": int(K),
        "mean_abs_cos_top1": float(abs_cos[:, 0].mean()),
        "median_abs_cos_top1": float(np.median(abs_cos[:, 0])),
        "mean_abs_cos_top10_mean": float(abs_cos[:, :min(10, K)].mean()),
        "mean_abs_cos_per_pc_top32": [float(abs_cos[:, k].mean()) if k < K else None
                                        for k in range(min(K_HEATMAP, K))],
    }


def main():
    groups = json.loads((ROOT / "geometry" / "groups_greek_vs_not.json").read_text())
    greek_ids = np.asarray(groups["Greek"], dtype=np.int64)
    not_ids = np.asarray(groups["not_Greek"], dtype=np.int64)

    summary = {"by_matrix_group": {}}
    for matrix in ("E", "U"):
        print(f"\n=== {matrix} ===", flush=True)
        M = np.load(ROOT / "arrays" / f"{matrix}_fp32.npy", mmap_mode="r")
        mu_g = np.load(V2 / f"mu_greek_{matrix}.npy")
        mu_n = np.load(V2 / f"mu_not_greek_{matrix}.npy")
        pc_g = np.load(V2 / f"pc_basis_greek_{matrix}.npy")
        pc_n = np.load(V2 / f"pc_basis_not_greek_{matrix}.npy")
        summary["by_matrix_group"][f"greek_{matrix}"] = process_group(
            M, greek_ids, mu_g, pc_g, "greek", matrix)
        summary["by_matrix_group"][f"not_greek_{matrix}"] = process_group(
            M, not_ids, mu_n, pc_n, "not_greek", matrix)

    (V2 / "direction_cosines_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[done]")


if __name__ == "__main__":
    main()

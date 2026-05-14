"""Phase 0 §2.1–§2.3 — per-matrix global summary, per-group centroids
+ top-K PCs, cross-group geometry.

Inputs:
  arrays/E_fp32.npy          (V, D) float32   — from extract_embeddings.py
  arrays/U_fp32.npy          (V, D) float32
  token_classification.jsonl  — from Phase A v2

Outputs (geometry/):
  centroids_E.npy            (n_groups+1, D)  — incl. "all-vocab"
  centroids_U.npy
  pc_basis_E_<group>.npy     (K, D)
  pc_singvals_E_<group>.npy  (K,)
  same for U
  pair_cos_E.npy             (n_groups, n_groups)
  pair_cos_U.npy
  pair_dist_pooled_E.npy     pooled-σ-normalised pairwise centroid distance
  pair_dist_pooled_U.npy
  anisotropy_E.json
  anisotropy_U.json
  global_stats.json
  group_index.json           — token-id -> group_idx + group_name list
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Order matters: this is the column order in the §6.1.2 contrast table.
# Group names come from the Phase A v2 classifier — note capitalisation.
GROUPS_OF_INTEREST = [
    "English-baseline",
    "Greek",
    "Cyrillic",
    "German",
    "French",
    "CJK",
    "structural_non_linguistic",
]

ARRAY_DIR = Path("/home/foivos/runs/apertus_embedding_init_test_20260512/arrays")
GEOM_DIR = Path("/home/foivos/runs/apertus_embedding_init_test_20260512/geometry")
CLASSIFICATION_PATH = Path(
    "/home/foivos/runs/apertus_greek_diagnostic_20260511_v2/token_classification.jsonl"
)


def load_groups(classification_path: Path, vocab_size: int):
    """Return ids_per_group: {group_name: np.ndarray(int64)} + classified mask."""
    ids_per_group: dict[str, list[int]] = defaultdict(list)
    classified = np.zeros(vocab_size, dtype=bool)
    excluded_buckets = {"special", "byte_fragment", "whitespace_only", "digits_only"}
    print(f"[load] {classification_path}", flush=True)
    with classification_path.open() as f:
        for line in f:
            r = json.loads(line)
            tid = int(r["id"])
            bucket = r.get("bucket", "")
            groups = r.get("groups", []) or []
            if bucket in excluded_buckets:
                continue
            classified[tid] = True
            for g in groups:
                ids_per_group[g].append(tid)
    out = {g: np.asarray(sorted(set(ids)), dtype=np.int64) for g, ids in ids_per_group.items()}
    return out, classified


def randomised_svd_top_k(X_centred: np.ndarray, k: int):
    """Compute top-K SVD of X_centred (n, d), return (U_basis: (k, d), s: (k,))."""
    from sklearn.utils.extmath import randomized_svd
    n, d = X_centred.shape
    n_components = min(k, min(n, d) - 1)
    if n_components < 1:
        return np.zeros((1, d), dtype=np.float32), np.zeros(1, dtype=np.float32)
    _, s, vt = randomized_svd(X_centred, n_components=n_components,
                               random_state=20260512, n_iter=4)
    return vt.astype(np.float32), s.astype(np.float32)


def random_pair_cosine_median(matrix: np.ndarray, ids: np.ndarray, n_pairs: int = 10000,
                               seed: int = 42) -> float:
    if ids.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    a_ix = rng.choice(ids, size=n_pairs, replace=True)
    b_ix = rng.choice(ids, size=n_pairs, replace=True)
    keep = a_ix != b_ix
    a = matrix[a_ix[keep]]
    b = matrix[b_ix[keep]]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    cos = (a_norm * b_norm).sum(axis=1)
    return float(np.median(cos))


def process_matrix(M: np.ndarray, name: str, groups: dict[str, np.ndarray],
                   classified_ids: np.ndarray, k_pc: int):
    """Compute §2.1 global + §2.2 per-group stats for one matrix; save outputs."""
    print(f"\n[{name}] processing matrix shape {M.shape}", flush=True)
    t0 = time.time()
    # §2.1 — global stats on the classified subset (pitfall #5: exclude special/byte/ws/digits).
    M_classified = M[classified_ids]
    mu_global = M_classified.mean(axis=0).astype(np.float32)
    norms = np.linalg.norm(M, axis=1)
    cos_random_pair = random_pair_cosine_median(M, classified_ids, n_pairs=10000)
    print(f"[{name}] ||mu_global||={np.linalg.norm(mu_global):.4f}  "
          f"median||row||={np.median(norms):.4f}  "
          f"random_pair_cos_median={cos_random_pair:.4f}", flush=True)

    # §2.2 — per-group centroids + top-K PCs.
    centroids = {"all_classified": mu_global}
    pc_bases: dict[str, np.ndarray] = {}
    pc_sings: dict[str, np.ndarray] = {}
    anisotropy: dict[str, dict] = {}

    for g in GROUPS_OF_INTEREST + ["all_classified"]:
        ids = groups.get(g) if g in groups else classified_ids
        if g == "all_classified":
            ids = classified_ids
        if ids is None or ids.size == 0:
            print(f"[{name}] group={g} empty; skip", flush=True)
            continue
        rows = M[ids]
        mu = rows.mean(axis=0).astype(np.float32)
        centred = (rows - mu).astype(np.float32)
        vt, s = randomised_svd_top_k(centred, k_pc)
        # Variance share: s² gives squared singular values; trace ≈ sum(s_i²) approximation
        # over top-K. Use full per-row variance for the trace baseline.
        within_var = float((centred ** 2).sum())
        top_k_var = float((s ** 2).sum())
        top1_share = float((s[0] ** 2) / within_var) if within_var > 0 else float("nan")
        d_to_global = float(np.linalg.norm(mu - mu_global))
        centroids[g] = mu
        pc_bases[g] = vt
        pc_sings[g] = s
        anisotropy[g] = {
            "n_tokens": int(ids.size),
            "trace_total": within_var,
            "trace_top_k": top_k_var,
            "top_k_var_share": top_k_var / within_var if within_var > 0 else float("nan"),
            "top_1_pc_share": top1_share,
            "centroid_distance_to_mu_global": d_to_global,
            "centroid_norm": float(np.linalg.norm(mu)),
            "median_row_norm": float(np.median(np.linalg.norm(rows, axis=1))),
        }
        print(f"[{name}] group={g:<28s} n={ids.size:>6d}  "
              f"top1_pc_share={top1_share:.4f}  ||mu-mu_global||={d_to_global:.4f}",
              flush=True)
        np.save(GEOM_DIR / f"pc_basis_{name}_{g}.npy", vt)
        np.save(GEOM_DIR / f"pc_singvals_{name}_{g}.npy", s)

    # Save centroids (in GROUPS_OF_INTEREST order + all_classified).
    centroid_order = GROUPS_OF_INTEREST + ["all_classified"]
    centroid_array = np.stack(
        [centroids[g] for g in centroid_order if g in centroids],
        axis=0,
    )
    np.save(GEOM_DIR / f"centroids_{name}.npy", centroid_array)
    (GEOM_DIR / f"centroid_order_{name}.json").write_text(json.dumps(
        [g for g in centroid_order if g in centroids]
    ))

    # §2.3 — cross-group geometry.
    pair_cos = np.zeros((len(centroid_order), len(centroid_order)), dtype=np.float32)
    pair_dist_pooled = np.zeros_like(pair_cos)
    for i, gi in enumerate(centroid_order):
        for j, gj in enumerate(centroid_order):
            if gi not in centroids or gj not in centroids:
                continue
            ci, cj = centroids[gi], centroids[gj]
            ai = anisotropy.get(gi, {})
            aj = anisotropy.get(gj, {})
            ti = ai.get("trace_total", 0.0) / max(ai.get("n_tokens", 1), 1)
            tj = aj.get("trace_total", 0.0) / max(aj.get("n_tokens", 1), 1)
            denom = np.sqrt((ti + tj) / 2) + 1e-12
            d = float(np.linalg.norm(ci - cj))
            pair_dist_pooled[i, j] = d / denom
            num = float((ci * cj).sum())
            cn = float(np.linalg.norm(ci) * np.linalg.norm(cj)) + 1e-12
            pair_cos[i, j] = num / cn
    np.save(GEOM_DIR / f"pair_cos_{name}.npy", pair_cos)
    np.save(GEOM_DIR / f"pair_dist_pooled_{name}.npy", pair_dist_pooled)

    (GEOM_DIR / f"anisotropy_{name}.json").write_text(json.dumps(anisotropy, indent=2))
    print(f"[{name}] complete in {time.time()-t0:.0f}s", flush=True)

    return {
        "mu_global_norm": float(np.linalg.norm(mu_global)),
        "median_row_norm": float(np.median(norms)),
        "random_pair_cos_median": cos_random_pair,
        "n_classified": int(classified_ids.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-pc", type=int, default=64)
    args = ap.parse_args()

    GEOM_DIR.mkdir(parents=True, exist_ok=True)
    E = np.load(ARRAY_DIR / "E_fp32.npy", mmap_mode="r")
    U = np.load(ARRAY_DIR / "U_fp32.npy", mmap_mode="r")
    V = E.shape[0]
    groups, classified_mask = load_groups(CLASSIFICATION_PATH, V)
    classified_ids = np.where(classified_mask)[0].astype(np.int64)
    print(f"[init] V={V} classified={classified_ids.size}", flush=True)
    for g in GROUPS_OF_INTEREST:
        n = groups.get(g, np.array([])).size
        print(f"  group={g:<30s} n={n}", flush=True)

    # Save group index for downstream reuse.
    group_index = {g: groups[g].tolist() for g in GROUPS_OF_INTEREST if g in groups}
    group_index["_classified_ids"] = classified_ids.tolist()
    (GEOM_DIR / "group_index.json").write_text(json.dumps(group_index))

    stats_E = process_matrix(np.asarray(E), "E", groups, classified_ids, args.k_pc)
    stats_U = process_matrix(np.asarray(U), "U", groups, classified_ids, args.k_pc)

    (GEOM_DIR / "global_stats.json").write_text(json.dumps({
        "E": stats_E, "U": stats_U,
        "groups_of_interest": GROUPS_OF_INTEREST,
        "k_pc": args.k_pc,
    }, indent=2))
    print("\n[done] phase0 §2.1–§2.3 complete; outputs under geometry/")


if __name__ == "__main__":
    main()

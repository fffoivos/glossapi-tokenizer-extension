"""v2 §3.11 — Local clustering within Greek.

Cluster the 1,494 Greek embeddings to reveal sub-structure. Two methods:
  - HDBSCAN (density-based, variable cluster sizes, can produce noise points)
  - k-means with k ∈ {8, 16, 32}

For each cluster, list the 20 nearest members to the cluster centroid
(cosine distance) — qualitative inspection by surface form.

Outputs:
  geometry/v2/greek_clusters_{E,U}.json   machine-readable
  geometry/v2/greek_clusters_{E,U}.md     pre-rendered human-readable gallery
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
try:
    import hdbscan  # type: ignore
    HAVE_HDBSCAN = True
except ImportError:
    HAVE_HDBSCAN = False

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
V2_DIR = ROOT / "geometry" / "v2"
CLASS_PATH = Path(
    "/home/foivos/runs/apertus_greek_diagnostic_20260511_v2/token_classification.jsonl"
)

K_VALUES = [8, 16, 32]
TOP_PER_CLUSTER = 20
SEED = 20260512


def load_decoded(ids):
    want = set(int(x) for x in ids)
    out = {}
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


def cluster_one(X_proj: np.ndarray, ids: np.ndarray, decoded: dict, method: str,
                params: dict):
    """Cluster + emit member lists per cluster, ordered by cosine to cluster centroid."""
    if method == "kmeans":
        km = KMeans(n_clusters=params["k"], random_state=SEED, n_init=10)
        labels = km.fit_predict(X_proj)
        centroids = km.cluster_centers_
    elif method == "hdbscan":
        if not HAVE_HDBSCAN:
            return None
        clusterer = hdbscan.HDBSCAN(min_cluster_size=params["min_cluster_size"],
                                     metric="euclidean")
        labels = clusterer.fit_predict(X_proj)
        # cluster centroids = mean of members
        unique_labels = sorted(set(labels) - {-1})
        centroids = np.array([X_proj[labels == lbl].mean(axis=0) for lbl in unique_labels])
        # remap labels to dense indexing (skip noise)
        remap = {old: new for new, old in enumerate(unique_labels)}
        # keep noise points labelled -1
        labels = np.array([-1 if lbl == -1 else remap[lbl] for lbl in labels])
    else:
        raise ValueError(method)

    clusters = {}
    for cid in range(len(centroids)):
        mask = labels == cid
        if not mask.any():
            continue
        members_proj = X_proj[mask]
        members_ids = ids[mask]
        c = centroids[cid]
        # rank by cosine to centroid (in projected subspace)
        c_norm = np.linalg.norm(c) + 1e-12
        cosines = (members_proj @ c) / (np.linalg.norm(members_proj, axis=1) * c_norm + 1e-12)
        order = np.argsort(-cosines)
        top = order[:TOP_PER_CLUSTER]
        clusters[cid] = {
            "n": int(mask.sum()),
            "top_members": [
                {"id": int(members_ids[i]),
                 "raw_token": decoded.get(int(members_ids[i]), {}).get("raw_token", ""),
                 "decoded_text": decoded.get(int(members_ids[i]), {}).get("decoded_text", ""),
                 "cos_to_cluster_centroid": float(cosines[i])}
                for i in top
            ],
        }
    n_noise = int((labels == -1).sum()) if (labels == -1).any() else 0
    return {"method": method, "params": params,
             "n_clusters": len(clusters), "n_noise": n_noise,
             "clusters": clusters}


def render_markdown(out_dict: dict, matrix: str) -> str:
    lines = [f"# Greek-token clustering ({matrix} matrix)", ""]
    for label, payload in out_dict.items():
        if payload is None:
            continue
        lines.append(f"## {payload['method']} (params={payload['params']})")
        lines.append(f"- n_clusters: {payload['n_clusters']}; n_noise: {payload['n_noise']}")
        lines.append("")
        for cid, c in sorted(payload["clusters"].items(), key=lambda kv: -kv[1]["n"]):
            lines.append(f"### Cluster {cid}  (n={c['n']})")
            lines.append("")
            for i, m in enumerate(c["top_members"]):
                lines.append(f"  {i+1:>2}. `{m['raw_token']}` → "
                              f"`{m['decoded_text']}`  (cos={m['cos_to_cluster_centroid']:+.3f})")
            lines.append("")
    return "\n".join(lines)


def process_matrix(matrix: str):
    print(f"=== {matrix} clustering ===", flush=True)
    M = np.load(ROOT / "arrays" / f"{matrix}_fp32.npy", mmap_mode="r")
    groups = json.loads((ROOT / "geometry" / "groups_greek_vs_not.json").read_text())
    greek_ids = np.asarray(groups["Greek"], dtype=np.int64)
    mu = np.load(V2_DIR / f"mu_greek_{matrix}.npy")
    pc = np.load(V2_DIR / f"pc_basis_greek_{matrix}.npy")
    eig = np.load(V2_DIR / f"pc_eigvals_greek_{matrix}.npy")
    # Use the full Greek subspace (top-K_sig PCs) — anything beyond is noise
    rows = np.asarray(M[greek_ids])
    z = (rows - mu) @ pc.T                    # (1494, K_sig)
    z_scaled = z / np.sqrt(np.maximum(eig, 1e-12))   # whitened — equal-variance per PC
    print(f"  greek_subspace shape: {z_scaled.shape}", flush=True)

    decoded = load_decoded(greek_ids.tolist())

    out = {}
    for k in K_VALUES:
        out[f"kmeans_k{k}"] = cluster_one(z_scaled, greek_ids, decoded, "kmeans", {"k": k})
    if HAVE_HDBSCAN:
        out["hdbscan_min15"] = cluster_one(z_scaled, greek_ids, decoded, "hdbscan",
                                            {"min_cluster_size": 15})
        out["hdbscan_min30"] = cluster_one(z_scaled, greek_ids, decoded, "hdbscan",
                                            {"min_cluster_size": 30})
    else:
        print("  hdbscan not available; skipping density-based clustering")

    (V2_DIR / f"greek_clusters_{matrix}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (V2_DIR / f"greek_clusters_{matrix}.md").write_text(render_markdown(out, matrix))
    print(f"  wrote greek_clusters_{matrix}.{{json,md}}", flush=True)


def main():
    V2_DIR.mkdir(parents=True, exist_ok=True)
    for m in ("E", "U"):
        process_matrix(m)


if __name__ == "__main__":
    main()

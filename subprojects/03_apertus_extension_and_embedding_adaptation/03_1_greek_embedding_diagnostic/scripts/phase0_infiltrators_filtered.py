"""v2.1 — Re-run §3.9 + §3.10 with untrained-floor filtered out.

The v2 finding was that the top-1000 ¬Greek tokens nearest μ_Greek are
dominated by Mistral-inherited specials (`<|fim_begin|>`, `[AVAILABLE_TOOLS]`)
and structural LaTeX tokens with near-zero row norms. Filter those out by
applying the absolute U-floor from Phase A v2's floors.json, then redo
the Mahalanobis-to-μ_Greek analysis.

Also: switch hull-occupancy bands to **quantile-of-Greek-distance** rather
than std-of-m. Bands reported:
  q10/q25/q50/q75/q90/q99 of Greek's own in-group Mahalanobis.

For ¬Greek: fraction of (filtered) ¬Greek tokens with Mahalanobis below
each Greek quantile.

Outputs:
  geometry/v2_1/infiltrators_filtered_{E,U}.json
  geometry/v2_1/infiltrators_filtered_top1000_{E,U}.csv
  geometry/v2_1/hull_quantiles_{E,U}.json
  geometry/v2_1/filter_summary.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
V2 = ROOT / "geometry" / "v2"
V21 = ROOT / "geometry" / "v2_1"
FLOORS = Path("/home/foivos/runs/apertus_greek_diagnostic_20260511_v2/floors.json")
CLASS_PATH = Path(
    "/home/foivos/runs/apertus_greek_diagnostic_20260511_v2/token_classification.jsonl"
)

TOP_N_INFILTRATORS = 1000
QUANTILES = [10, 25, 50, 75, 90, 99]


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


def mahalanobis_in_subspace(rows: np.ndarray, mu: np.ndarray,
                             pc_basis: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
    z = (rows - mu) @ pc_basis.T
    inv_lambda = 1.0 / np.maximum(eigvals, 1e-12)
    m_sq = (z ** 2) * inv_lambda
    return np.sqrt(m_sq.sum(axis=1))


def top_pc_contributions(rows: np.ndarray, mu: np.ndarray,
                          pc_basis: np.ndarray, eigvals: np.ndarray,
                          n_top: int = 3) -> np.ndarray:
    z = (rows - mu) @ pc_basis.T
    contrib = (z ** 2) / np.maximum(eigvals, 1e-12)
    return np.argsort(-contrib, axis=1)[:, :n_top]


def process(matrix: str):
    print(f"\n=== {matrix} ===", flush=True)
    M = np.load(ROOT / "arrays" / f"{matrix}_fp32.npy", mmap_mode="r")
    M = np.asarray(M)
    groups = json.loads((ROOT / "geometry" / "groups_greek_vs_not.json").read_text())
    greek_ids = np.asarray(groups["Greek"], dtype=np.int64)
    not_ids = np.asarray(groups["not_Greek"], dtype=np.int64)
    src_lookup = groups["source_group_of_negreek"]

    # Filter at p1 of classified row norms (per-matrix). Phase A v2's
    # absolute_floor_median_U (0.4566) catches only the very-untrained
    # tail; p1 is a more useful lower-bound for "near-untrained".
    norms = np.linalg.norm(M, axis=1)
    all_ids = np.asarray(groups["all_classified"], dtype=np.int64)
    floor_value = float(np.percentile(norms[all_ids], 1.0))
    floor_kind = f"p1_of_classified_{matrix}_norms"
    print(f"  floor (matrix {matrix}): {floor_value:.4f}  [{floor_kind}]", flush=True)

    # Filter ¬Greek by row-norm
    n_norms = norms[not_ids]
    keep_mask = n_norms > floor_value
    filtered_ids = not_ids[keep_mask]
    print(f"  ¬Greek before filter: {not_ids.size}", flush=True)
    print(f"  ¬Greek after filter:  {filtered_ids.size}  "
          f"(dropped {not_ids.size - filtered_ids.size})", flush=True)

    # Load Greek subspace
    mu_greek = np.load(V2 / f"mu_greek_{matrix}.npy")
    pc_greek = np.load(V2 / f"pc_basis_greek_{matrix}.npy")
    eig_greek = np.load(V2 / f"pc_eigvals_greek_{matrix}.npy")
    k_sig = pc_greek.shape[0]
    print(f"  K_sig (Greek): {k_sig}", flush=True)

    # Compute Greek's own Mahalanobis (re-derived here so we have full quantiles)
    m_greek_own = mahalanobis_in_subspace(M[greek_ids], mu_greek, pc_greek, eig_greek)
    print(f"  Greek m: mean={m_greek_own.mean():.2f}, min={m_greek_own.min():.2f}, "
          f"max={m_greek_own.max():.2f}", flush=True)

    greek_quantiles = {f"q{q}": float(np.percentile(m_greek_own, q)) for q in QUANTILES}
    print(f"  Greek quantiles: {greek_quantiles}", flush=True)

    # Now compute Mahalanobis for FILTERED ¬Greek
    m_negreek = mahalanobis_in_subspace(M[filtered_ids], mu_greek, pc_greek, eig_greek)
    print(f"  filtered ¬Greek m: mean={m_negreek.mean():.2f}, "
          f"min={m_negreek.min():.2f}, p1={np.percentile(m_negreek, 1):.2f}", flush=True)

    # Hull occupancy in quantile terms (fraction of ¬Greek below each Greek quantile)
    hull = {"matrix": matrix,
            "filter": {"kind": floor_kind, "value": floor_value,
                        "n_dropped": int(not_ids.size - filtered_ids.size)},
            "greek_n": int(greek_ids.size),
            "negreek_filtered_n": int(filtered_ids.size),
            "k_sig_greek": int(k_sig),
            "greek_mahalanobis_quantiles": greek_quantiles}
    for q in QUANTILES:
        v = greek_quantiles[f"q{q}"]
        hull[f"frac_negreek_below_greek_q{q}"] = float((m_negreek <= v).mean())
        hull[f"count_negreek_below_greek_q{q}"] = int((m_negreek <= v).sum())
    # Also: where do filtered ¬Greek tokens' m values sit on Greek's CDF?
    greek_sorted = np.sort(m_greek_own)
    pcts_full = np.searchsorted(greek_sorted, m_negreek) / max(greek_sorted.size, 1) * 100
    hull["filtered_negreek_mahalanobis_quantiles"] = {
        f"p{p}": float(np.percentile(m_negreek, p)) for p in (1, 5, 25, 50, 75, 95, 99)
    }
    hull["filtered_negreek_percentile_in_greek_distribution_quantiles"] = {
        f"p{p}": float(np.percentile(pcts_full, p)) for p in (1, 5, 25, 50, 75, 95, 99)
    }

    # Top-1000 infiltrators (after filter)
    top_n = min(TOP_N_INFILTRATORS, m_negreek.size)
    infil_idx = np.argsort(m_negreek)[:top_n]
    infil_ids = filtered_ids[infil_idx]
    pcts_top1000 = np.searchsorted(greek_sorted, m_negreek[infil_idx]) / max(greek_sorted.size, 1) * 100
    decoded = load_decoded(infil_ids.tolist())
    top_pcs = top_pc_contributions(M[infil_ids], mu_greek, pc_greek, eig_greek)

    src_within_q50 = {}
    src_within_q25 = {}
    rows_out = []
    for rank, idx in enumerate(infil_idx):
        tid = int(filtered_ids[idx])
        m = float(m_negreek[idx])
        src = src_lookup.get(str(tid), src_lookup.get(tid, "?"))
        dec = decoded.get(tid, {})
        rows_out.append({
            "rank": rank + 1, "id": tid,
            "raw_token": dec.get("raw_token", ""),
            "decoded_text": dec.get("decoded_text", ""),
            "source_group": src,
            "mahalanobis_to_greek": m,
            "row_norm": float(norms[tid]),
            "percentile_in_greek_distribution": float(pcts_top1000[rank]),
            "top_3_pcs": [int(x) for x in top_pcs[rank]],
        })
        if m <= greek_quantiles["q50"]:
            src_within_q50[src] = src_within_q50.get(src, 0) + 1
        if m <= greek_quantiles["q25"]:
            src_within_q25[src] = src_within_q25.get(src, 0) + 1

    hull["top_1000_source_group_below_greek_q50"] = src_within_q50
    hull["top_1000_source_group_below_greek_q25"] = src_within_q25
    hull["top_20_filtered_infiltrators"] = rows_out[:20]

    (V21 / f"infiltrators_filtered_{matrix}.json").write_text(json.dumps(hull, indent=2))
    with (V21 / f"infiltrators_filtered_top1000_{matrix}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "id", "raw_token", "decoded_text", "source_group",
                    "row_norm", "mahalanobis_to_greek", "percentile_in_greek_distribution"])
        for r in rows_out:
            w.writerow([r["rank"], r["id"], r["raw_token"], r["decoded_text"],
                        r["source_group"], r["row_norm"], r["mahalanobis_to_greek"],
                        r["percentile_in_greek_distribution"]])

    # Save the filtered distance arrays for plotting
    np.savez(V21 / f"distance_filtered_{matrix}.npz",
              greek_ids=greek_ids,
              greek_mahalanobis=m_greek_own,
              filtered_ids=filtered_ids,
              filtered_negreek_mahalanobis=m_negreek,
              top1000_ids=infil_ids,
              top1000_mahalanobis=m_negreek[infil_idx],
              top1000_percentile_in_greek=pcts_top1000,
              greek_quantiles=np.asarray([greek_quantiles[f"q{q}"] for q in QUANTILES]),
              quantile_labels=np.asarray(QUANTILES))

    # Quantile-based hull file (for plotting + harmonisation with v2)
    quant_hull = {
        "matrix": matrix,
        "greek_quantiles": greek_quantiles,
        "fraction_of_greek_below_each_greek_quantile": {
            f"q{q}": float((m_greek_own <= greek_quantiles[f"q{q}"]).mean()) for q in QUANTILES
        },
        "fraction_of_filtered_negreek_below_each_greek_quantile": {
            f"q{q}": hull[f"frac_negreek_below_greek_q{q}"] for q in QUANTILES
        },
        "count_of_filtered_negreek_below_each_greek_quantile": {
            f"q{q}": hull[f"count_negreek_below_greek_q{q}"] for q in QUANTILES
        },
    }
    (V21 / f"hull_quantiles_{matrix}.json").write_text(json.dumps(quant_hull, indent=2))
    return hull


def main():
    V21.mkdir(parents=True, exist_ok=True)
    out = {}
    for m in ("E", "U"):
        out[m] = process(m)
    summary = {
        "matrices": {
            mat: {
                "filter_kind": v["filter"]["kind"],
                "filter_value": v["filter"]["value"],
                "n_dropped_negreek": v["filter"]["n_dropped"],
                "negreek_filtered_n": v["negreek_filtered_n"],
                "greek_quantiles": v["greek_mahalanobis_quantiles"],
                "frac_negreek_below_greek_q50": v["frac_negreek_below_greek_q50"],
                "count_negreek_below_greek_q50": v["count_negreek_below_greek_q50"],
                "frac_negreek_below_greek_q25": v["frac_negreek_below_greek_q25"],
                "count_negreek_below_greek_q25": v["count_negreek_below_greek_q25"],
                "frac_negreek_below_greek_q10": v["frac_negreek_below_greek_q10"],
                "count_negreek_below_greek_q10": v["count_negreek_below_greek_q10"],
            }
            for mat, v in out.items()
        }
    }
    (V21 / "filter_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[done] geometry/v2_1/ written", flush=True)


if __name__ == "__main__":
    main()

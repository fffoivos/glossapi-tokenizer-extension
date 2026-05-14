"""Recompute Greek-row diagnostics using the strict 1,494-token set
from base_greek_tokens.jsonl (decoded-as-Greek-only).

Drops the 13 mixed-script tokens (μm, μg, -α, Aβ, etc.) that Phase A v2's
byte-level classifier had included.

Updates affected geometry artefacts (Greek centroids + PC bases +
anisotropy) and re-runs the linear classifier with strict Greek replacing
broad Greek. Other groups unchanged.

Outputs:
  geometry/strict_greek_ids.json       (1,494 token ids)
  geometry/strict_greek_diff.json      (broad vs strict deltas, summary)
  geometry/centroids_{E,U}.npy         (Greek row overwritten with strict)
  geometry/pc_basis_{E,U}_Greek.npy    (overwritten)
  geometry/pc_singvals_{E,U}_Greek.npy
  geometry/anisotropy_{E,U}.json       (Greek row overwritten)
  geometry/linear_classifier_{E,U}.json  (recomputed with strict Greek)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import randomized_svd

ARRAY_DIR = Path("/home/foivos/runs/apertus_embedding_init_test_20260512/arrays")
GEOM_DIR = Path("/home/foivos/runs/apertus_embedding_init_test_20260512/geometry")
BASE_GREEK = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/"
    "tokenizer_analysis/inspection/base/greek_tokens/base_greek_tokens.jsonl"
)
GROUPS_OF_INTEREST = [
    "English-baseline", "Greek", "Cyrillic", "German", "French", "CJK",
    "structural_non_linguistic",
]
K_PC = 64
SEED = 20260512


def load_strict_greek() -> list[int]:
    ids = []
    with BASE_GREEK.open() as f:
        for line in f:
            ids.append(int(json.loads(line)["id"]))
    return sorted(set(ids))


def compute_group(matrix: np.ndarray, ids: np.ndarray, mu_global: np.ndarray):
    rows = matrix[ids]
    mu = rows.mean(axis=0).astype(np.float32)
    centred = (rows - mu).astype(np.float32)
    n_components = min(K_PC, min(centred.shape) - 1)
    _, s, vt = randomized_svd(centred, n_components=n_components,
                               random_state=SEED, n_iter=4)
    within_var = float((centred ** 2).sum())
    top_k_var = float((s ** 2).sum())
    return {
        "mu": mu,
        "pc_basis": vt.astype(np.float32),
        "pc_singvals": s.astype(np.float32),
        "stats": {
            "n_tokens": int(ids.size),
            "trace_total": within_var,
            "trace_top_k": top_k_var,
            "top_k_var_share": top_k_var / within_var if within_var > 0 else float("nan"),
            "top_1_pc_share": float((s[0] ** 2) / within_var) if within_var > 0 else float("nan"),
            "centroid_distance_to_mu_global": float(np.linalg.norm(mu - mu_global)),
            "centroid_norm": float(np.linalg.norm(mu)),
            "median_row_norm": float(np.median(np.linalg.norm(rows, axis=1))),
        },
    }


def linear_classifier(matrix: np.ndarray, group_ids: dict[str, np.ndarray], focus: str):
    X_rows, y_labels = [], []
    for g in GROUPS_OF_INTEREST:
        if g not in group_ids:
            continue
        for tid in group_ids[g]:
            X_rows.append(int(tid))
            y_labels.append(g)
    X = matrix[np.asarray(X_rows, dtype=np.int64)]
    y = np.asarray(y_labels)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, stratify=y_enc, test_size=0.2,
                                            random_state=SEED)
    clf = LogisticRegression(solver="saga", max_iter=2000, n_jobs=-1, random_state=SEED)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    cr = classification_report(yte, yhat, target_names=le.classes_.tolist(),
                                 output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, yhat).tolist()
    focus_idx = int(np.where(le.classes_ == focus)[0][0])
    w_focus = clf.coef_[focus_idx]

    # Greek-direction alignment
    centroid_order = json.loads((GEOM_DIR / "centroid_order_E.json").read_text())
    cents = np.load(GEOM_DIR / f"centroids_{'E' if matrix.shape == np.load(ARRAY_DIR/'E_fp32.npy', mmap_mode='r').shape else 'U'}.npy")
    return {
        "test_accuracy": float(clf.score(Xte, yte)),
        "class_order": le.classes_.tolist(),
        "per_class_f1": {cls: cr[cls]["f1-score"] for cls in le.classes_},
        "per_class_support": {cls: cr[cls]["support"] for cls in le.classes_},
        "per_class_precision": {cls: cr[cls]["precision"] for cls in le.classes_},
        "per_class_recall": {cls: cr[cls]["recall"] for cls in le.classes_},
        "confusion_matrix": cm,
        "macro_f1": cr["macro avg"]["f1-score"],
        "fit_seconds": int(time.time() - t0),
        "weight_vector_norm_focus": float(np.linalg.norm(w_focus)),
        "weight_vector_focus": w_focus.tolist(),   # for downstream alignment
    }


def main():
    GEOM_DIR.mkdir(exist_ok=True)
    strict_ids = np.asarray(load_strict_greek(), dtype=np.int64)
    print(f"[init] strict Greek: {strict_ids.size} tokens")
    (GEOM_DIR / "strict_greek_ids.json").write_text(json.dumps(strict_ids.tolist()))

    # Load broad Greek for diff reporting
    broad_classification = Path(
        "/home/foivos/runs/apertus_greek_diagnostic_20260511_v2/token_classification.jsonl"
    )
    broad_greek = set()
    with broad_classification.open() as f:
        for line in f:
            r = json.loads(line)
            if "Greek" in (r.get("groups") or []):
                broad_greek.add(int(r["id"]))
    strict_set = set(strict_ids.tolist())
    print(f"[diff] broad-only ({len(broad_greek - strict_set)}): "
          f"{sorted(broad_greek - strict_set)}")
    print(f"[diff] strict-only ({len(strict_set - broad_greek)}): "
          f"{sorted(strict_set - broad_greek)}")

    # Existing groups (load to overlay)
    group_index = json.loads((GEOM_DIR / "group_index.json").read_text())
    group_index_for_classifier = {
        g: np.asarray(group_index[g], dtype=np.int64)
        for g in GROUPS_OF_INTEREST if g in group_index and g != "Greek"
    }
    group_index_for_classifier["Greek"] = strict_ids

    diff = {"strict_n": int(strict_ids.size), "broad_n": len(broad_greek)}

    for name in ("E", "U"):
        print(f"\n[{name}] recomputing Greek row + classifier")
        M = np.load(ARRAY_DIR / f"{name}_fp32.npy", mmap_mode="r")
        # Mu_global stays the same (computed on all classified, Greek shift is < 1%)
        aniso_path = GEOM_DIR / f"anisotropy_{name}.json"
        aniso = json.loads(aniso_path.read_text())
        broad_greek_stats = aniso["Greek"]

        mu_global = np.load(GEOM_DIR / f"centroids_{name}.npy")[
            json.loads((GEOM_DIR / f"centroid_order_{name}.json").read_text()).index("all_classified")
        ]
        g_out = compute_group(np.asarray(M), strict_ids, mu_global)

        # Overwrite Greek row in centroids array
        order = json.loads((GEOM_DIR / f"centroid_order_{name}.json").read_text())
        cents = np.load(GEOM_DIR / f"centroids_{name}.npy")
        cents[order.index("Greek")] = g_out["mu"]
        np.save(GEOM_DIR / f"centroids_{name}.npy", cents)

        np.save(GEOM_DIR / f"pc_basis_{name}_Greek.npy", g_out["pc_basis"])
        np.save(GEOM_DIR / f"pc_singvals_{name}_Greek.npy", g_out["pc_singvals"])

        aniso["Greek"] = g_out["stats"]
        aniso_path.write_text(json.dumps(aniso, indent=2))

        print(f"  Greek strict: n={g_out['stats']['n_tokens']}")
        print(f"    median_row_norm:                broad={broad_greek_stats['median_row_norm']:.4f}  strict={g_out['stats']['median_row_norm']:.4f}")
        print(f"    centroid_distance_to_mu_global: broad={broad_greek_stats['centroid_distance_to_mu_global']:.4f}  strict={g_out['stats']['centroid_distance_to_mu_global']:.4f}")
        print(f"    top_1_pc_share:                 broad={broad_greek_stats['top_1_pc_share']:.4f}  strict={g_out['stats']['top_1_pc_share']:.4f}")
        diff[name] = {
            "broad": {k: broad_greek_stats[k] for k in
                      ("median_row_norm","centroid_distance_to_mu_global","top_1_pc_share","centroid_norm")},
            "strict": {k: g_out["stats"][k] for k in
                       ("median_row_norm","centroid_distance_to_mu_global","top_1_pc_share","centroid_norm")},
        }

        # Linear classifier with strict Greek
        print(f"  fitting linear classifier on {name} (strict Greek)...")
        clf_out = linear_classifier(np.asarray(M), group_index_for_classifier, "Greek")
        (GEOM_DIR / f"linear_classifier_{name}.json").write_text(json.dumps(
            {k: v for k, v in clf_out.items() if k != "weight_vector_focus"},
            indent=2
        ))
        # Save the Greek weight vector for downstream cos alignment
        np.save(GEOM_DIR / f"linear_classifier_{name}_greek_weight.npy",
                 np.asarray(clf_out["weight_vector_focus"], dtype=np.float32))
        print(f"    Greek F1: {clf_out['per_class_f1']['Greek']:.4f}  "
              f"macro_F1: {clf_out['macro_f1']:.4f}  fit_seconds={clf_out['fit_seconds']}")

    (GEOM_DIR / "strict_greek_diff.json").write_text(json.dumps(diff, indent=2))
    print("\n[done] strict Greek recomputation complete")


if __name__ == "__main__":
    main()

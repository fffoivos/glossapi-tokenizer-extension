"""Phase 0 §2.7.10 — Linear language-classifier probe.

Multinomial logistic regression on E (and U) to predict group membership.
Outputs per-group F1 + confusion matrix + the "Greek direction" weight vector.

Output:
  geometry/linear_classifier_E.json
  geometry/linear_classifier_U.json
  geometry/language_direction_alignment_<group>_E.json
  geometry/language_direction_alignment_<group>_U.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

GROUPS_OF_INTEREST = [
    "English-baseline", "Greek", "Cyrillic", "German", "French", "CJK",
    "structural_non_linguistic",
]

ARRAY_DIR = Path("/home/foivos/runs/apertus_embedding_init_test_20260512/arrays")
GEOM_DIR = Path("/home/foivos/runs/apertus_embedding_init_test_20260512/geometry")


def build_dataset(matrix: np.ndarray, group_index: dict):
    X_rows, y_labels = [], []
    for g in GROUPS_OF_INTEREST:
        ids = group_index.get(g, [])
        if not ids:
            continue
        for tid in ids:
            X_rows.append(tid)
            y_labels.append(g)
    ids = np.asarray(X_rows, dtype=np.int64)
    X = matrix[ids]
    return X, np.asarray(y_labels), ids


def cos_align(a: np.ndarray, b: np.ndarray) -> float:
    n_a = np.linalg.norm(a) + 1e-12
    n_b = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (n_a * n_b))


def process(matrix: np.ndarray, name: str, group_index: dict, focus_group: str,
            seed: int):
    t0 = time.time()
    X, y, _ = build_dataset(np.asarray(matrix), group_index)
    print(f"[{name}] dataset: X={X.shape}, n_classes={len(set(y))}", flush=True)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, stratify=y_enc, test_size=0.2,
                                            random_state=seed)
    clf = LogisticRegression(solver="saga", max_iter=2000,
                              multi_class="multinomial", n_jobs=-1,
                              random_state=seed)
    print(f"[{name}] fitting (saga, max_iter=2000) ...", flush=True)
    clf.fit(Xtr, ytr)
    score = float(clf.score(Xte, yte))
    print(f"[{name}] test accuracy = {score:.4f}", flush=True)

    yhat = clf.predict(Xte)
    cm = confusion_matrix(yte, yhat).tolist()
    cls_report = classification_report(yte, yhat, target_names=le.classes_.tolist(),
                                         output_dict=True, zero_division=0)
    per_class = {cls: cls_report.get(cls, {}) for cls in le.classes_}

    # Find the focus group's class index.
    focus_idx = int(np.where(le.classes_ == focus_group)[0][0]) if focus_group in le.classes_ else None
    direction_meta = {}
    if focus_idx is not None:
        w_focus = clf.coef_[focus_idx]
        # Compare to centroid-displacement direction and to other directions.
        centroid_order = json.loads((GEOM_DIR / f"centroid_order_{name}.json").read_text())
        centroids = np.load(GEOM_DIR / f"centroids_{name}.npy")
        mu_focus = centroids[centroid_order.index(focus_group)]
        mu_global = centroids[centroid_order.index("all_classified")]
        disp = mu_focus - mu_global
        direction_meta = {
            "weight_vector_norm": float(np.linalg.norm(w_focus)),
            "cos_with_centroid_displacement": cos_align(w_focus, disp),
            "cos_with_mu_focus": cos_align(w_focus, mu_focus),
            "cos_with_mu_global": cos_align(w_focus, mu_global),
        }
        (GEOM_DIR / f"language_direction_alignment_{focus_group}_{name}.json").write_text(
            json.dumps(direction_meta, indent=2)
        )

    out = {
        "matrix": name,
        "test_accuracy": score,
        "class_order": le.classes_.tolist(),
        "per_class_f1": {cls: v.get("f1-score", None) for cls, v in per_class.items()},
        "per_class_support": {cls: v.get("support", None) for cls, v in per_class.items()},
        "per_class_precision": {cls: v.get("precision", None) for cls, v in per_class.items()},
        "per_class_recall": {cls: v.get("recall", None) for cls, v in per_class.items()},
        "confusion_matrix": cm,
        "macro_f1": float(cls_report.get("macro avg", {}).get("f1-score", float("nan"))),
        "focus_group_alignment": direction_meta,
        "fit_seconds": int(time.time() - t0),
    }
    out_path = GEOM_DIR / f"linear_classifier_{name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{name}] wrote {out_path}; macro_f1={out['macro_f1']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus-group", default="Greek")
    ap.add_argument("--seed", type=int, default=20260512)
    args = ap.parse_args()

    group_index = json.loads((GEOM_DIR / "group_index.json").read_text())
    # group_index values are lists of int (token-id) per group + _classified_ids.
    group_index_for_classifier = {g: group_index[g] for g in GROUPS_OF_INTEREST if g in group_index}

    E = np.load(ARRAY_DIR / "E_fp32.npy", mmap_mode="r")
    U = np.load(ARRAY_DIR / "U_fp32.npy", mmap_mode="r")

    process(np.asarray(E), "E", group_index_for_classifier, args.focus_group, args.seed)
    process(np.asarray(U), "U", group_index_for_classifier, args.focus_group, args.seed)
    print("[done] linear classifier complete")


if __name__ == "__main__":
    main()

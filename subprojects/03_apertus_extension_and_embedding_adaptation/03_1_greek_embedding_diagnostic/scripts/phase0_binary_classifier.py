"""v2 §3.7 — binary Greek vs ¬Greek logistic regression on E and U.

Plan said binary; v1 was 7-class. This is the binary version.

Class balance: Greek has 1,494 tokens, ¬Greek has 126,990. We balance
training by sampling a stratified subset of ¬Greek (n=10,000) so the
classifier sees a meaningful negative class without choking on saga
with 130k samples (would be ~hours).

Output:
  geometry/v2/linear_classifier_binary_{E,U}.json
  geometry/v2/linear_classifier_binary_weight_greek_{E,U}.npy
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
V2 = ROOT / "geometry" / "v2"
SEED = 20260513
NEG_SAMPLES = 10_000


def run(matrix: str):
    print(f"\n=== {matrix} binary classifier ===", flush=True)
    M = np.load(ROOT / "arrays" / f"{matrix}_fp32.npy", mmap_mode="r")
    groups = json.loads((ROOT / "geometry" / "groups_greek_vs_not.json").read_text())
    greek_ids = np.asarray(groups["Greek"], dtype=np.int64)
    not_ids = np.asarray(groups["not_Greek"], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    neg_pick = rng.choice(not_ids, size=min(NEG_SAMPLES, not_ids.size), replace=False)

    X = np.vstack([np.asarray(M[greek_ids]), np.asarray(M[neg_pick])])
    y = np.concatenate([np.ones(greek_ids.size), np.zeros(neg_pick.size)]).astype(np.int8)
    print(f"  X.shape={X.shape}, n_pos={int(y.sum())}, n_neg={int((1-y).sum())}", flush=True)

    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=SEED)
    t0 = time.time()
    clf = LogisticRegression(solver="saga", max_iter=2000, n_jobs=-1,
                              random_state=SEED, class_weight="balanced")
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    fit_seconds = int(time.time() - t0)

    cr = classification_report(yte, yhat, target_names=["not_Greek", "Greek"],
                                 output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, yhat).tolist()

    w_greek = clf.coef_[0]   # binary → coef_ is (1, d)
    # Compare weight direction to centroid displacement + top-1 PC of Greek
    mu_greek = np.load(V2 / f"mu_greek_{matrix}.npy")
    mu_not = np.load(V2 / f"mu_not_greek_{matrix}.npy")
    mu_classified = np.load(V2 / f"mu_classified_{matrix}.npy")
    pc_greek_top1 = np.load(V2 / f"pc_basis_greek_{matrix}.npy")[0]

    def cos(a, b):
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (na * nb))

    out = {
        "matrix": matrix,
        "task": "binary Greek vs ¬Greek logistic regression",
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "neg_sample_size": int(NEG_SAMPLES),
        "test_accuracy": float(clf.score(Xte, yte)),
        "macro_f1": cr["macro avg"]["f1-score"],
        "greek_f1": cr["Greek"]["f1-score"],
        "greek_precision": cr["Greek"]["precision"],
        "greek_recall": cr["Greek"]["recall"],
        "greek_support": cr["Greek"]["support"],
        "negreek_f1": cr["not_Greek"]["f1-score"],
        "confusion_matrix": cm,
        "fit_seconds": fit_seconds,
        "weight_vector_norm": float(np.linalg.norm(w_greek)),
        "cos_weight_vs_mu_greek_minus_mu_negreek": cos(w_greek, mu_greek - mu_not),
        "cos_weight_vs_mu_greek_minus_mu_global": cos(w_greek, mu_greek - mu_classified),
        "cos_weight_vs_top1_pc_greek": cos(w_greek, pc_greek_top1),
    }
    (V2 / f"linear_classifier_binary_{matrix}.json").write_text(json.dumps(out, indent=2))
    np.save(V2 / f"linear_classifier_binary_weight_greek_{matrix}.npy",
             w_greek.astype(np.float32))
    print(f"  test_acc={out['test_accuracy']:.4f}, macro_f1={out['macro_f1']:.4f}, "
          f"Greek F1={out['greek_f1']:.4f}, fit={fit_seconds}s")
    print(f"  cos(w, μ_Greek − μ_¬Greek) = {out['cos_weight_vs_mu_greek_minus_mu_negreek']:.4f}")
    print(f"  cos(w, μ_Greek − μ_global) = {out['cos_weight_vs_mu_greek_minus_mu_global']:.4f}")
    print(f"  cos(w, top-1 PC of Greek)  = {out['cos_weight_vs_top1_pc_greek']:.4f}")


def main():
    for m in ("E", "U"):
        run(m)
    print("\n[done]")


if __name__ == "__main__":
    main()

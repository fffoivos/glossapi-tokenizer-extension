#!/usr/bin/env python3
"""Train two binary per-line heads on the unified 2,000-doc LLM silver (struct_lines):
  bib head  = (label==1)  over BIB_FEATS (22)  → candidate span_line_lr_struct_model.json
  toc head  = (label==2)  over TOC_FEATS (27)  → toc_line_lr_model.json
Both via the existing numpy LR (line_lr.fit_lr); standardize→dot→sigmoid, Rust-portable. The ToC head's
probabilities are front-gated (struct_lines.apply_toc_gate) for decoding/eval. Reports per-line PR per
class on the doc-grouped TEST, plus the DEPLOYED bib model (trained on the old span_dataset) evaluated on
the same silver test split for comparison only; this does not authorize production.

  python train_struct.py            # train + report
  python train_struct.py rebuild    # rebuild the feature matrix first
"""

import json
import os
import sys
import collections
import numpy as np
import line_lr as L
import struct_lines as SL

HERE = os.path.dirname(os.path.abspath(__file__))
BIB_COST = 1.0
TOC_COST = 1.0


def train_head(bundle, keys, y, cost_fp):
    idx = [bundle["col"][k] for k in keys]
    X = bundle["X"][:, idx].astype(np.float64)
    tr = bundle["tr"]
    w = np.where(y == 1, 1.0, cost_fp)
    mu, sd, W, b = L.fit_lr(X[tr], y[tr], w[tr])
    P = L._prob(X, mu, sd, W, b)
    model = {
        "features": list(keys),
        "mu": [round(float(x), 5) for x in mu],
        "sd": [round(float(x), 5) for x in sd],
        "weight": [round(float(x), 5) for x in W],
        "bias": round(float(b), 5),
        "cost_fp": cost_fp,
        "note": "per-line LR; standardize x then sigmoid(w·z+b); decode via decode_spans hysteresis.",
    }
    return P, model


def apply_model(bundle, model):
    """Apply an exported model JSON to the current matrix (used to score the OLD bib model on the new test)."""
    idx = [bundle["col"][k] for k in model["features"]]
    X = bundle["X"][:, idx].astype(np.float64)
    mu = np.array(model["mu"])
    sd = np.array(model["sd"])
    W = np.array(model["weight"])
    b = model["bias"]
    return L._prob(X, mu, sd, W, b)


def pline_of(P, rows):
    pl = collections.defaultdict(list)
    for (doc, _), pp in zip(rows, P):
        pl[doc].append(float(pp))
    return {k: np.array(v) for k, v in pl.items()}


def recall_at_prec(P, y, target=0.97):
    o = np.argsort(-P)
    ct = cf = 0
    tot = int(y.sum())
    best = 0.0
    thr = 1.0
    for j in o:
        if y[j] == 1:
            ct += 1
        else:
            cf += 1
        if ct and ct / (ct + cf) >= target:
            best = ct / tot
            thr = float(P[j])
    return best, thr


def sweep(name, P, y, gate_mask=None):
    if gate_mask is not None:
        P = P * gate_mask
    print(
        f"\n[{name}] per-line PR on TEST (raw, pre-smoothing):   ({int(y.sum()):,} positives / {len(y):,} lines)"
    )
    print("  thr   precision  recall  flagged%")
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        pred = P >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        pr = tp / (tp + fp) if tp + fp else 1.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {thr:.2f}   {pr:.3f}      {rc:.3f}   {100 * pred.mean():.1f}%")
    rec, thr = recall_at_prec(P, y, 0.97)
    print(f"  → recall@prec0.97: {rec:.3f}  (thr≈{thr:.3f})")
    return rec


def main():
    data = SL.load()
    bundle = SL.build_matrix(data, rebuild="rebuild" in sys.argv)
    te = ~bundle["tr"]
    gate = SL.toc_gate(data)
    # per-row gate mask aligned to bundle rows (for the raw test sweep)
    gmask = np.array([gate[doc][i] for doc, i in bundle["rows"]], dtype=bool)

    # ── bib head (new gold) ──
    Pb, mb = train_head(bundle, SL.BIB_FEATS, bundle["y_bib"], BIB_COST)
    json.dump(
        mb,
        open(f"{HERE}/span_line_lr_struct_model.json", "w"),
        ensure_ascii=False,
        indent=1,
    )
    # ── toc head ──
    Pt, mt = train_head(bundle, SL.TOC_FEATS, bundle["y_toc"], TOC_COST)
    json.dump(
        mt, open(f"{HERE}/toc_line_lr_model.json", "w"), ensure_ascii=False, indent=1
    )

    print(
        f"trained 2 heads on {int(bundle['tr'].sum()):,} train / {int(te.sum()):,} test lines"
    )
    rb = sweep("BIB (new gold)", Pb[te], bundle["y_bib"][te])
    sweep("TOC (front-gated)", Pt[te], bundle["y_toc"][te], gate_mask=gmask[te])

    # ── no-regression preview: DEPLOYED bib model on the SAME new test split ──
    old_path = f"{HERE}/span_line_lr_model.json"
    if os.path.exists(old_path):
        old = json.load(open(old_path))
        if all(k in bundle["col"] for k in old["features"]):
            Po = apply_model(bundle, old)
            ro = sweep(
                "BIB (deployed/old-data model) on new test", Po[te], bundle["y_bib"][te]
            )
            print(
                f"\nREGRESSION PREVIEW (recall@prec0.97 on new test): new-gold bib {rb:.3f}  vs  deployed {ro:.3f}  "
                f"→ Δ {rb - ro:+.3f}   ({'new ≥ deployed (gate trends OK)' if rb >= ro else 'new < deployed — investigate before promoting'})"
            )
        else:
            print(
                "\n(deployed bib model features not all present in matrix — skipping regression preview)"
            )

    print(
        "\ntop ToC weights (standardized):",
        json.dumps(
            dict(
                sorted(
                    zip(mt["features"], [round(x, 2) for x in mt["weight"]]),
                    key=lambda kv: -abs(kv[1]),
                )[:10]
            ),
            ensure_ascii=False,
        ),
    )
    print(
        "candidate models → span_line_lr_struct_model.json (bib), toc_line_lr_model.json (toc)"
    )


if __name__ == "__main__":
    main()

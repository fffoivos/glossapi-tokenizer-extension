#!/usr/bin/env python3
"""Tier-1 interpretable per-line in-bib classifier (context-rich logistic regression, numpy, no
sklearn — Rust-portable). Own-line signals (span_signals.line_signals) + CONTEXT features (neighbor
entry/year/page density, distance to a bibliography header above, document position). Feed-forward by
design — the contiguity smoother (decode_spans.py) owns the sequential dependency.

Featurizes ONCE; ablation reuses columns. Exposes:
  build_matrix(data)           -> bundle (Xall, yall, is_tr, rows, col)  [rows aligned to lines]
  train_and_predict(data, ...) -> (pline {doc_id: np.array}, model, bundle); fits on TRAIN only
Run directly: per-line PR sweep + precision-first op point + leave-one-feature-out ablation.
"""
import json, os, collections
import numpy as np
import span_signals as S
import span_seq_data as D
HERE = os.path.dirname(os.path.abspath(__file__))

OWN = ["is_entry", "author", "greek_author", "gr_edmark", "year_paren", "page", "place", "locator",
       "latin_frac", "is_bib_header", "short_line", "url", "num_prefix"]
# context block; under_bib_block + lead_num + runpur12 added to separate footnote/inline citations
# from true bibliography entries (deterministic context separators, AUC 0.86-0.94 — see determinism_probe.py)
CTX = ["ed3", "ed8", "runpur12", "yd8", "pd8", "dist_bib_hdr", "under_bib_block", "lead_num", "pos"]
FEATS = OWN + CTX
import re as _re
_LEADNUM = _re.compile(r"^\s*\[?(\d{1,4})\]?[.)]")


def _roll(vals, i, k):
    a, b = max(0, i - k), min(len(vals), i + k + 1)
    return sum(vals[a:b]) / (b - a) if b > a else 0.0


def doc_features(d):
    lines = d["lines"]
    sig = [S.line_signals(t) for _, t in lines]
    ent = [s["is_entry"] for s in sig]; yr = [s["year_bare"] for s in sig]; pg = [s["page"] for s in sig]
    hdr = [s["is_bib_header"] for s in sig]
    N = d["N"] or 1
    dist = [1.0] * len(lines); last = None
    sect = ""; under = [0.0] * len(lines)   # header-scoping: are we inside a bibliography section?
    for i, (_, t) in enumerate(lines):
        if hdr[i]:
            last = i
        dist[i] = min(i - last, 50) / 50.0 if last is not None else 1.0
        if S.ATX_HEADER.search(t):
            fam = S.header_family(t)
            sect = fam if fam else "other"
        under[i] = 1.0 if sect == "bib" else 0.0
    out = []
    for i, ((abs_idx, txt), s) in enumerate(zip(lines, sig)):
        m = _LEADNUM.match(txt)
        f = {k: s[k] for k in OWN}
        f.update(ed3=_roll(ent, i, 3), ed8=_roll(ent, i, 8), runpur12=_roll(ent, i, 12),
                 yd8=_roll(yr, i, 8), pd8=_roll(pg, i, 8), dist_bib_hdr=dist[i],
                 under_bib_block=under[i], lead_num=min(float(m.group(1)) if m else 0.0, 300) / 300.0,
                 pos=abs_idx / N)
        out.append(f)
    return out


MCACHE = f"{HERE}/units/SPAN_linemat.npz"


def build_matrix(data, rebuild=False):
    col = {k: j for j, k in enumerate(FEATS)}
    if os.path.exists(MCACHE) and not rebuild:
        z = np.load(MCACHE, allow_pickle=True)
        docids = list(z["docids"])
        rows = [(docids[di], li) for di, li in z["rowdoc"]]
        return dict(X=z["X"], y=z["y"], tr=z["tr"], rows=rows, col=col)
    X, y, tr, rowdoc = [], [], [], []
    docids = list(data.docs.keys()); didx = {d: i for i, d in enumerate(docids)}
    for doc_id, d in data.docs.items():
        feats = doc_features(d)
        gold = set()
        for s in d["spans"]:
            a, b = s["start_line"], s["end_line"]
            if isinstance(a, int) and isinstance(b, int) and b >= a:
                gold.update(range(a, b + 1))
        istr = d["split"] == "train"
        for i, ((abs_idx, _), f) in enumerate(zip(d["lines"], feats)):
            X.append([f[k] for k in FEATS]); y.append(1 if abs_idx in gold else 0)
            tr.append(istr); rowdoc.append((didx[doc_id], i))
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int8); tr = np.asarray(tr, bool)
    rowdoc = np.asarray(rowdoc, np.int32)
    np.savez(MCACHE, X=X, y=y, tr=tr, rowdoc=rowdoc, docids=np.array(docids, dtype="U64"))
    rows = [(docids[di], li) for di, li in rowdoc]
    return dict(X=X, y=y, tr=tr, rows=rows, col=col)


def fit_lr(X, y, w=None, iters=1500, lr=0.3, l2=1e-2):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = ((X - mu) / sd).astype(np.float64)
    n, k = Z.shape; W = np.zeros(k); b = 0.0
    w = np.ones(n) if w is None else np.asarray(w, float); w = w / w.mean()
    y = np.asarray(y, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ W + b)))
        W -= lr * (Z.T @ (w * (p - y)) / n + l2 * W)
        b -= lr * (w * (p - y)).mean()
    return mu, sd, W, b


def _prob(X, mu, sd, W, b):
    return 1 / (1 + np.exp(-(((X - mu) / sd) @ W + b)))


def _fit_cols(bundle, keys, cost_fp=1.0):
    idx = [bundle["col"][k] for k in keys]
    X = bundle["X"][:, idx]; y = bundle["y"]; tr = bundle["tr"]
    w = np.where(y == 1, 1.0, cost_fp)
    mu, sd, W, b = fit_lr(X[tr], y[tr], w[tr])
    P = _prob(X, mu, sd, W, b)
    model = {"features": keys, "mu": [round(float(x), 5) for x in mu], "sd": [round(float(x), 5) for x in sd],
             "weight": [round(float(x), 5) for x in W], "bias": round(float(b), 5), "cost_fp": cost_fp,
             "note": "per-line in-bib LR; standardize x then sigmoid(w·z+b); decode via decode_spans hysteresis."}
    return P, model


def train_pline(data, bundle=None, extra_name=None, extra_vec=None, extra_mat=None, extra_keys=None,
                cost_fp=1.0, iters=1500):
    """Fit the line LR on FEATS (+ optional extra per-row column(s): p_win for synergy, or a frozen
    line-embedding block for the DL-augmented arm) and return (pline {doc_id: array}, model). Reuses the
    cached feature matrix; no re-featurization."""
    if bundle is None:
        bundle = build_matrix(data)
    idx = [bundle["col"][k] for k in FEATS]
    X = bundle["X"][:, idx].astype(np.float64)
    keys = list(FEATS)
    if extra_vec is not None:
        X = np.column_stack([X, np.asarray(extra_vec, np.float64)])
        keys = keys + [extra_name]
    if extra_mat is not None:
        X = np.column_stack([X, np.asarray(extra_mat, np.float64)])
        keys = keys + (extra_keys or [f"emb{j}" for j in range(extra_mat.shape[1])])
    y = bundle["y"]; tr = bundle["tr"]
    w = np.where(y == 1, 1.0, cost_fp)
    mu, sd, W, b = fit_lr(X[tr], y[tr], w[tr], iters=iters)
    P = _prob(X, mu, sd, W, b)
    pline = collections.defaultdict(list)
    for (doc_id, _), pp in zip(bundle["rows"], P):
        pline[doc_id].append(float(pp))
    pline = {k: np.array(v) for k, v in pline.items()}
    model = {"features": keys, "mu": [round(float(x), 5) for x in mu], "sd": [round(float(x), 5) for x in sd],
             "weight": [round(float(x), 5) for x in W], "bias": round(float(b), 5), "cost_fp": cost_fp}
    return pline, model


def train_and_predict(data, keys=FEATS, cost_fp=1.0, export=True, bundle=None):
    if bundle is None:
        bundle = build_matrix(data)
    P, model = _fit_cols(bundle, keys, cost_fp)
    pline = collections.defaultdict(list)
    for (doc_id, _), pp in zip(bundle["rows"], P):
        pline[doc_id].append(float(pp))
    pline = {k: np.array(v) for k, v in pline.items()}
    if export:
        json.dump(model, open(f"{HERE}/span_line_lr_model.json", "w"), ensure_ascii=False, indent=1)
    return pline, model, bundle


def _recall_at_prec(P, y, target=0.97):
    order = np.argsort(-P); tot = int(y.sum()); ct = cf = 0; best = 0.0; thr = 1.0
    for j in order:
        if y[j] == 1: ct += 1
        else: cf += 1
        if ct / (ct + cf) >= target:
            best = ct / tot; thr = float(P[j])
    return best, thr


def main():
    data = D.load()
    bundle = build_matrix(data)
    P, model = _fit_cols(bundle, FEATS, cost_fp=1.0)
    json.dump(model, open(f"{HERE}/span_line_lr_model.json", "w"), ensure_ascii=False, indent=1)
    tr = bundle["tr"]; y = bundle["y"]; te = ~tr
    yte, Pte = y[te], P[te]
    print(f"line model: {len(FEATS)} features, {int(tr.sum()):,} train / {int(te.sum()):,} test lines")
    print("\nPER-LINE precision/recall sweep on TEST (raw, pre-smoothing):")
    print("  thr   precision  recall  flagged%")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        pred = Pte >= thr
        tp = int((pred & (yte == 1)).sum()); fp = int((pred & (yte == 0)).sum()); fn = int((~pred & (yte == 1)).sum())
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        print(f"  {thr:.2f}   {pr:.3f}      {rc:.3f}   {100*pred.mean():.1f}%")
    rec, thr = _recall_at_prec(Pte, yte, 0.97)
    print(f"\nprecision-first op point (line-prec≥0.97): thr≈{thr:.3f}  recall={rec:.3f}")
    print("\nweights (standardized): " + json.dumps(dict(zip(FEATS, [round(x, 2) for x in model["weight"]])), ensure_ascii=False))
    import sys
    if "ablate" in sys.argv:
        print("\nleave-one-feature-out ablation (Δ recall@prec0.97 on test):")
        abl = []
        for drop in FEATS:
            P2, _ = _fit_cols(bundle, [k for k in FEATS if k != drop])
            r2, _ = _recall_at_prec(P2[te], yte, 0.97)
            abl.append((drop, r2 - rec))
        for k, dv in sorted(abl, key=lambda x: x[1]):
            print(f"  drop {k:<14} Δrecall {dv:+.3f}")


if __name__ == "__main__":
    main()

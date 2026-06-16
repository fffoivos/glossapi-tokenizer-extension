#!/usr/bin/env python3
"""Contiguity smoother: turn per-line in-bib probabilities into clean multi-span (start_line,end_line)
output. Hysteresis run-length (primary) — open a run at p≥θ_hi, extend while p≥θ_lo, merge runs within
G present lines, drop runs shorter than L_min. Tuned PRECISION-FIRST: grid-search θ on TRAIN to
MAXIMIZE recall subject to line-level precision ≥ target (default 0.97); spans shrink inward rather
than bleed into prose. Frozen params → span_smooth_params.json.

Run directly: trains the interpretable line-LR, tunes the smoother on train, reports line-level
precision/recall + decoded-span stats on the held-out test.
"""
import json, os, collections
import numpy as np
import span_seq_data as D
import line_lr as L
HERE = os.path.dirname(os.path.abspath(__file__))
PCACHE = f"{HERE}/units/SPAN_pline_lineLR.npz"


def hysteresis(p, theta_hi, theta_lo, gap, lmin):
    """p: per-present-line probs (1D). Returns list of (start_idx,end_idx) over present-line indices."""
    n = len(p); runs = []; inrun = False; start = 0
    for i in range(n):
        if not inrun:
            if p[i] >= theta_hi:
                inrun = True; start = i
        elif p[i] < theta_lo:
            runs.append((start, i - 1)); inrun = False
    if inrun:
        runs.append((start, n - 1))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s + 1 >= lmin]


def decode_doc(d, p, params):
    """Return spans as absolute (start_line,end_line) for one doc."""
    runs = hysteresis(p, **params)
    lines = d["lines"]
    return [(lines[s][0], lines[e][0]) for s, e in runs]


def _gold_lineset(d):
    g = set()
    for s in d["spans"]:
        a, b = s["start_line"], s["end_line"]
        if isinstance(a, int) and isinstance(b, int) and b >= a:
            g.update(range(a, b + 1))
    return g


def _pred_lineset(d, spans):
    g = set()
    for s, e in spans:
        g.update(range(s, e + 1))
    # restrict to present lines (we only ever remove present lines)
    present = {a for a, _ in d["lines"]}
    return g & present


def line_pr(data, pline, params, which="test"):
    """Aggregate line-level precision/recall over docs in the split (precision = removed lines that are
    truly bib; the FP / prose-amputation rate is 1-precision)."""
    tp = fp = fn = 0
    for doc_id, d in data.docs.items():
        if d["split"] != which:
            continue
        spans = decode_doc(d, pline[doc_id], params)
        pred = _pred_lineset(d, spans)
        gold = _gold_lineset(d) & {a for a, _ in d["lines"]}
        tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, tp, fp, fn


def tune(data, pline, target=0.97):
    """Grid search hysteresis params on TRAIN for max recall s.t. line-precision ≥ target."""
    best = None
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        for tl in (0.3, 0.4, 0.5, 0.6):
            if tl >= th:
                continue
            for gap in (2, 4, 8):
                for lmin in (2, 3, 5):
                    params = dict(theta_hi=th, theta_lo=tl, gap=gap, lmin=lmin)
                    prec, rec, *_ = line_pr(data, pline, params, "train")
                    if prec >= target and (best is None or rec > best[1]):
                        best = (params, rec, prec)
    if best is None:  # nothing reaches target; fall back to the most precise config
        for th in (0.95, 0.9):
            params = dict(theta_hi=th, theta_lo=th - 0.1, gap=2, lmin=5)
            prec, rec, *_ = line_pr(data, pline, params, "train")
            if best is None or prec > best[2]:
                best = (params, rec, prec)
    return best


def get_pline(data, synergy=False, rebuild=False):
    cache = PCACHE.replace(".npz", "_syn.npz") if synergy else PCACHE
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    bundle = L.build_matrix(data)
    if synergy:
        import window_clf as W
        pw, _, _ = W.predict(data, export=True)
        lpw = W.line_pwin(data, pw)
        vec = np.array([lpw[doc][i] for doc, i in bundle["rows"]])
        pline, model = L.train_pline(data, bundle, "pwin", vec)
        json.dump(model, open(f"{HERE}/span_line_lr_syn_model.json", "w"), ensure_ascii=False, indent=1)
    else:
        pline, model = L.train_pline(data, bundle)
        json.dump(model, open(f"{HERE}/span_line_lr_model.json", "w"), ensure_ascii=False, indent=1)
    np.savez(cache, **{k: v for k, v in pline.items()})
    return pline


def main():
    import sys
    synergy = "synergy" in sys.argv
    data = D.load()
    pline = get_pline(data, synergy=synergy, rebuild="rebuild" in sys.argv)
    print("MODE:", "two-stage (line+window p_win)" if synergy else "line-only")
    best = tune(data, pline, target=0.97)
    params, _, _ = best
    json.dump(params, open(f"{HERE}/span_smooth_params.json", "w"), indent=1)
    tr_prec, tr_rec, *_ = line_pr(data, pline, params, "train")
    te_prec, te_rec, tp, fp, fn = line_pr(data, pline, params, "test")
    print(f"tuned hysteresis: {params}")
    print(f"TRAIN  line-precision {tr_prec:.3f}  recall {tr_rec:.3f}")
    print(f"TEST   line-precision {te_prec:.3f}  recall {te_rec:.3f}   (TP {tp:,} FP {fp:,} FN {fn:,})")
    print(f"       => prose-amputation (FP) rate = {1-te_prec:.3f}")
    # decoded-span / multi-span stats on test
    nsp = collections.Counter()
    for doc_id, d in data.docs.items():
        if d["split"] != "test":
            continue
        nsp[min(len(decode_doc(d, pline[doc_id], params)), 5)] += 1
    print("decoded spans per test doc:", dict(sorted(nsp.items())))


if __name__ == "__main__":
    main()

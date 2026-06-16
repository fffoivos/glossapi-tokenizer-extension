#!/usr/bin/env python3
"""Operating-point frontier for the line classifier. The smoother was tuned for strict bib-precision
≥0.97 (treating footnote/inline citations as errors), but the REAL constraint is prose-protection —
don't delete running main text. Since ~97% of 'false positives' are citation reference-mass, tuning
against prose-protection lets us recover recall at no real cost. This maps recall vs prose-protection
across hysteresis thresholds and picks the operating point that maximizes recall subject to
prose-protection ≥ target (default 0.995), then reports the held-out test numbers there."""
import os, json, collections
import numpy as np
import span_seq_data as D
import span_signals as S
import decode_spans as DS
import failure_analysis as FA
HERE = os.path.dirname(os.path.abspath(__file__))
PROSE = {"prose", "footnote_prose"}   # strictly running main text (the costly removal)


def precompute(data):
    """Per doc: pline (cached), and per-line (gold flag, is_prose flag), aligned to d['lines']."""
    pline = DS.get_pline(data)
    info = {}
    for doc_id, d in data.docs.items():
        gold = set()
        for s in d["spans"]:
            a, b = s.get("start_line"), s.get("end_line")
            if isinstance(a, int) and isinstance(b, int) and b >= a:
                gold.update(range(a, b + 1))
        N = d["N"] or 1
        gflag = np.zeros(len(d["lines"]), bool); pflag = np.zeros(len(d["lines"]), bool)
        for i, (abs_idx, t) in enumerate(d["lines"]):
            gflag[i] = abs_idx in gold
            if not gflag[i]:
                pflag[i] = FA.categorize_fp(t, S.line_signals(t), abs_idx / N) in PROSE
        info[doc_id] = (pline[doc_id], gflag, pflag)
    return info


def eval_params(data, info, params, which):
    rem = tp = fn = prose_rem = 0
    for doc_id, d in data.docs.items():
        if d["split"] != which:
            continue
        p, gflag, pflag = info[doc_id]
        pred = np.zeros(len(d["lines"]), bool)
        for s, e in DS.hysteresis(p, **params):
            pred[s:e + 1] = True
        rem += int(pred.sum()); tp += int((pred & gflag).sum())
        fn += int((~pred & gflag).sum()); prose_rem += int((pred & pflag).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    pp = 1 - prose_rem / rem if rem else 1.0
    return recall, pp, prose_rem, rem


def main():
    data = D.load()
    info = precompute(data)
    grid = [dict(theta_hi=th, theta_lo=tl, gap=g, lmin=lm)
            for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95) for tl in (0.3, 0.4, 0.5, 0.6, 0.7)
            for g in (2, 4) for lm in (2, 3, 5) if tl < th]
    DEFAULT_FLOOR = 0.999   # conservative: honor "don't remove main text"; dial down for more recall
    print("recall/prose-protection frontier — tuned on TRAIN, reported on held-out TEST:")
    rows = [(eval_params(data, info, p, "train"), p) for p in grid]
    frontier = {}
    for floor in (0.999, 0.997, 0.995, 0.99):
        ok = [(r[0], p) for r, p in rows if r[1] >= floor]
        if not ok:
            continue
        p = max(ok, key=lambda x: x[0])[1]
        rec_te, pp_te, pr_te, rem_te = eval_params(data, info, p, "test")
        frontier[str(floor)] = dict(params=p, test_recall=round(rec_te, 3), test_prose_protection=round(pp_te, 4),
                                    test_prose_lines=pr_te, test_removed=rem_te)
        print(f"  prose-protection floor {floor}:  TEST recall {rec_te:.3f}  prose-protection {pp_te:.4f}  "
              f"({pr_te} prose / {rem_te:,} removed)   {p}")
    best = frontier[str(DEFAULT_FLOOR)]["params"]
    json.dump(best, open(f"{HERE}/span_smooth_params.json", "w"), indent=1)
    json.dump(frontier, open(f"{HERE}/results_operating_point.json", "w"), indent=1)
    print(f"\nDEFAULT op point (prose-protection≥{DEFAULT_FLOOR}, conservative): {best}")
    print("→ written to span_smooth_params.json; dial the floor down in results_operating_point.json for more recall.")


if __name__ == "__main__":
    main()

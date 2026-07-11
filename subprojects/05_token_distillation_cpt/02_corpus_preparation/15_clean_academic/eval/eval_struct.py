#!/usr/bin/env python3
"""Precision-first operating-point frontier on the unified LLM-silver labels. Reuses
operating_point.eval_params + failure_analysis.categorize_fp (prose vs citation FP is class-agnostic — the
costly error is removing running MAIN TEXT, for either head). For each class: tune hysteresis on TRAIN to
max recall s.t. prose-protection ≥ floor; report on the held-out TEST. The ToC head is front-gated.

  python eval_struct.py
"""

import os
import json
import numpy as np
import operating_point as OP
import failure_analysis as FA
import span_signals as S
import struct_lines as SL
import train_struct as T

HERE = os.path.dirname(os.path.abspath(__file__))

GRID = [
    dict(theta_hi=th, theta_lo=tl, gap=g, lmin=lm)
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    for tl in (0.3, 0.4, 0.5, 0.6, 0.7)
    for g in (2, 4, 8)
    for lm in (2, 3, 5)
    if tl < th
]


def precompute(data, pline):
    """Per doc: (pline, gold-flag, is-prose-flag) aligned to d['lines']. Mirrors operating_point.precompute
    but with an injected per-class pline and the per-class gold (set via struct_lines.set_class)."""
    info = {}
    for doc_id, d in data.docs.items():
        gold = set()
        for s in d["spans"]:
            a, b = s.get("start_line"), s.get("end_line")
            if isinstance(a, int) and isinstance(b, int) and b >= a:
                gold.update(range(a, b + 1))
        N = d["N"] or 1
        gflag = np.zeros(len(d["lines"]), bool)
        pflag = np.zeros(len(d["lines"]), bool)
        for i, (abs_idx, t) in enumerate(d["lines"]):
            gflag[i] = abs_idx in gold
            if not gflag[i]:
                pflag[i] = (
                    FA.categorize_fp(t, S.line_signals(t), abs_idx / N) in OP.PROSE
                )
        info[doc_id] = (pline[doc_id], gflag, pflag)
    return info


def frontier(cls, data, info):
    rows = [(OP.eval_params(data, info, p, "train"), p) for p in GRID]
    res = {}
    print(
        f"\n[{cls.upper()}]  recall / prose-protection frontier (tuned TRAIN → TEST):"
    )
    for floor in (0.999, 0.997, 0.995, 0.99):
        ok = [(r[0], p) for r, p in rows if r[1] >= floor]
        if not ok:
            continue
        p = max(ok, key=lambda x: x[0])[1]
        rec, pp, pr, rem = OP.eval_params(data, info, p, "test")
        res[str(floor)] = dict(
            params=p,
            test_recall=round(rec, 3),
            test_prose_protection=round(pp, 4),
            test_prose_lines=int(pr),
            test_removed=int(rem),
        )
        print(
            f"  prose-protection floor {floor}:  TEST recall {rec:.3f}  prose-protection {pp:.4f}  "
            f"({pr} prose / {rem:,} removed lines)   {p}"
        )
    return res


def main():
    data = SL.load()
    bundle = SL.build_matrix(data)
    Pb, _ = T.train_head(bundle, SL.BIB_FEATS, bundle["y_bib"], T.BIB_COST)
    Pt, _ = T.train_head(bundle, SL.TOC_FEATS, bundle["y_toc"], T.TOC_COST)
    gate = SL.toc_gate(data)
    pl = {
        "bib": T.pline_of(Pb, bundle["rows"]),
        "toc": SL.apply_toc_gate(T.pline_of(Pt, bundle["rows"]), gate),
    }
    out = {}
    for cls in ("bib", "toc"):
        SL.set_class(data, cls)
        out[cls] = frontier(cls, data, precompute(data, pl[cls]))
    json.dump(
        out,
        open(f"{HERE}/results_struct_operating_point.json", "w"),
        indent=1,
        ensure_ascii=False,
    )
    print("\nwrote results_struct_operating_point.json")
    print(
        "NOTE: prose-protection ≠ raw line-precision — it credits only running-main-text removals as the "
        "costly error (citation/footnote/list FPs are cheap & reversible)."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Composition of what the classifier REMOVES (held-out test) — the policy view. Every removed line is
either a true bibliography entry (correct) or, if not gold, categorized by type (footnote citation /
inline citation in prose / reference table-row / web source / ToC / caption / header / running prose).
This separates the only error that actually hurts training — deleting RUNNING PROSE — from removing
'reference mass' (footnote/inline/table citations) that a corpus cleaner may well WANT to strip.

Reports:
  prose_protection_precision = 1 − (running-prose lines removed / all removed)   ← the metric that matters
  strict_bib_precision       = true-bib lines / all removed                       ← the binary view
  reference_mass_precision   = (bib + footnote/inline/table/web citations) / all removed  ← policy view
"""
import os, re, json, collections
import numpy as np
import span_seq_data as D
import span_signals as S
import decode_spans as DS
import failure_analysis as FA
HERE = os.path.dirname(os.path.abspath(__file__))

CITATION = {"footnote_citation", "entry_like(non-target)", "table", "web/url", "entry_frontmatter", "inline_cite_prose"}
PROSE = {"prose", "footnote_prose", "short/numeric"}
STRUCTURE = {"toc", "caption", "header"}


def main():
    data = D.load()
    pline = DS.get_pline(data)
    params = json.load(open(f"{HERE}/span_smooth_params.json"))
    comp = collections.Counter()
    n_removed = n_bib = 0
    for doc_id, d in data.docs.items():
        if d["split"] != "test":
            continue
        gold = set()
        for s in d["spans"]:
            a, b = s.get("start_line"), s.get("end_line")
            if isinstance(a, int) and isinstance(b, int) and b >= a:
                gold.update(range(a, b + 1))
        pred = set()
        for s, e in DS.decode_doc(d, pline[doc_id], params):
            pred.update(range(s, e + 1))
        present = {a for a, _ in d["lines"]}
        gold &= present; pred &= present
        N = d["N"] or 1
        txtmap = {a: t for a, t in d["lines"]}
        for abs_idx in pred:
            n_removed += 1
            if abs_idx in gold:
                n_bib += 1; comp["bibliography_entry (correct)"] += 1
            else:
                t = txtmap[abs_idx]
                comp[FA.categorize_fp(t, S.line_signals(t), abs_idx / N)] += 1

    prose = sum(v for k, v in comp.items() if k in PROSE)
    cit = sum(v for k, v in comp.items() if k in CITATION)
    struct = sum(v for k, v in comp.items() if k in STRUCTURE)
    print(f"=== composition of {n_removed:,} REMOVED lines (held-out test) ===")
    for k, v in comp.most_common():
        print(f"   {k:<32} {v:6,}  ({100*v/n_removed:.1f}%)")
    pp = 1 - prose / n_removed
    sb = n_bib / n_removed
    rm = (n_bib + cit) / n_removed
    print(f"\nstrict_bib_precision      = {sb:.3f}   (true bibliography / removed)")
    print(f"reference_mass_precision  = {rm:.3f}   (bib + footnote/inline/table/web citations / removed)")
    print(f"PROSE-PROTECTION precision = {pp:.4f}  (1 − running-prose removed / removed)  ← the metric that matters")
    print(f"   running prose wrongly removed: {prose} of {n_removed:,} lines = {100*prose/n_removed:.2f}%")
    json.dump({"n_removed": n_removed, "strict_bib_precision": round(sb, 3),
               "reference_mass_precision": round(rm, 3), "prose_protection_precision": round(pp, 4),
               "composition": dict(comp)}, open(f"{HERE}/results_composition.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

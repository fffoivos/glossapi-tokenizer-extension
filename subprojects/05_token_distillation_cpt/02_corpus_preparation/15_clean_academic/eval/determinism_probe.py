#!/usr/bin/env python3
"""Can DETERMINISTIC features separate the confusable 'false' lines (citation-shaped non-bibliography:
footnote citations, inline cites, reference table-rows) from true bibliography ENTRIES? Both groups are
locally identical (author+year+title), so the separators must be deterministic CONTEXT. Measures
single-feature separability (AUC) of candidate deterministic features on the held-out TEST split.

Groups (test, is_entry lines only — i.e. already citation-shaped): positives = true bib entries
(gold), negatives = citation-shaped non-gold lines (the confusions). High AUC ⇒ that feature alone
would help the model tell them apart."""
import os, re, json
import numpy as np
import span_seq_data as D
import span_signals as S
HERE = os.path.dirname(os.path.abspath(__file__))
LEADNUM = re.compile(r"^\s*\[?(\d{1,4})\]?[.)]")
TAB = re.compile(r"\t|\|")


def section_type_stream(lines):
    """For each present line, the family of the most-recent header above: 'bib'|'cv'|'app'|'other'|''."""
    cur = ""
    out = []
    for _, t in lines:
        if S.ATX_HEADER.search(t):
            fam = S.header_family(t)
            cur = fam if fam else "other"
        out.append(cur)
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return 0.5
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float)  # ranks 0..n-1
    rp = r[:len(pos)].sum()
    a = (rp - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return max(a, 1 - a), a  # symmetric separability, and signed


def main():
    data = D.load()
    feats = {k: ([], []) for k in
             ["runpur_pm12", "dist_bib_hdr", "under_bib_block", "pos", "lead_num", "has_tab", "ed8(current)"]}
    npos = nneg = 0
    for doc_id, d in data.docs.items():
        if d["split"] != "test":
            continue
        lines = d["lines"]
        sig = [S.line_signals(t) for _, t in lines]
        ent = [s["is_entry"] for s in sig]
        sect = section_type_stream(lines)
        N = d["N"] or 1
        gold = set()
        for s in d["spans"]:
            a, b = s.get("start_line"), s.get("end_line")
            if isinstance(a, int) and isinstance(b, int) and b >= a:
                gold.update(range(a, b + 1))
        # distance to nearest bib header above (present lines)
        dist = []; last = None
        for i, (_, t) in enumerate(lines):
            if S.is_bib_header(t):
                last = i
            dist.append(min(i - last, 60) / 60.0 if last is not None else 1.0)
        for i, (abs_idx, t) in enumerate(lines):
            if not ent[i]:                      # only citation-SHAPED lines (the confusable set)
                continue
            isbib = abs_idx in gold
            a, b = max(0, i - 12), min(len(lines), i + 13)
            runpur = sum(ent[a:b]) / (b - a)
            m = LEADNUM.match(t)
            vals = {"runpur_pm12": runpur, "dist_bib_hdr": dist[i],
                    "under_bib_block": 1.0 if sect[i] == "bib" else 0.0, "pos": abs_idx / N,
                    "lead_num": float(m.group(1)) if m else 0.0, "has_tab": 1.0 if TAB.search(t) else 0.0,
                    "ed8(current)": sum(ent[max(0, i - 8):i + 9]) / (min(len(lines), i + 9) - max(0, i - 8))}
            for k, v in vals.items():
                feats[k][0 if isbib else 1].append(v)
            npos += isbib; nneg += not isbib

    print(f"TEST citation-shaped lines: {npos:,} true-bib entries (pos) vs {nneg:,} non-bib citations (neg)\n")
    print("single-feature separability (AUC; 1.0 = perfectly separates true-bib from confusable citation):")
    rows = []
    for k, (p, n) in feats.items():
        a, signed = auc(p, n)
        mp, mn = np.mean(p), np.mean(n)
        rows.append((a, k, mp, mn))
    for a, k, mp, mn in sorted(rows, reverse=True):
        print(f"  {k:<18} AUC {a:.3f}   mean(true-bib)={mp:7.3f}  mean(citation)={mn:7.3f}")
    json.dump({k: round(auc(p, n)[0], 3) for k, (p, n) in feats.items()},
              open(f"{HERE}/results_determinism_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score Eval A — detector end-matter bibliography BOUNDARY vs Opus truth.
(1) found/not confusion matrix (detector_header_found vs annotator has_end_bibliography) → P/R/F1.
(2) boundary START-LINE error for docs both found (|detector_start − annotator_start|) → localisation.
Breakdowns by source + stratum. Writes eval/CONFUSION_MATRIX_A.md + eval/results_A.json.
"""
import glob, json, os, collections, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    man = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{HERE}/units/A_boundary_manifest.jsonl")}
    raw = json.load(open(f"{HERE}/annotations_scale_A/all.json"))
    ann = {a["unit_id"]: a for a in (raw["annotations"] if isinstance(raw, dict) else raw)}

    rows = []
    for uid, m in man.items():
        a = ann.get(uid)
        if not a:
            continue
        det_found = bool(m["detector_header_found"])
        true_found = bool(a["has_end_bibliography"])
        det_start = m.get("detector_bib_start_line")
        ann_start = a.get("bibliography_start_line")
        rows.append(dict(uid=uid, source=m["source"], stratum=m["stratum"], det_found=det_found,
                         true_found=true_found, det_start=det_start, ann_start=ann_start,
                         tail0=m["tail_starts_at_line"]))

    # (1) found/not confusion (only where, if detector found, the bib is inside the shown tail)
    usable = [r for r in rows if not (r["det_found"] and r["det_start"] is not None and r["det_start"] < r["tail0"])]
    TP = sum(1 for r in usable if r["det_found"] and r["true_found"])
    FP = sum(1 for r in usable if r["det_found"] and not r["true_found"])
    FN = sum(1 for r in usable if not r["det_found"] and r["true_found"])
    TN = sum(1 for r in usable if not r["det_found"] and not r["true_found"])
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

    # (2) boundary localisation where both found and both have line numbers
    errs = []
    for r in rows:
        if r["det_found"] and r["true_found"] and r["det_start"] is not None and r["ann_start"] not in (None, -1):
            errs.append(r["det_start"] - r["ann_start"])
    abserr = [abs(e) for e in errs]

    res = dict(n=len(rows), usable=len(usable), TP=TP, FP=FP, FN=FN, TN=TN,
               precision=prec, recall=rec, f1=f1,
               boundary_n=len(errs),
               boundary_median_abs=(st.median(abserr) if abserr else None),
               boundary_within_5=(sum(1 for e in abserr if e <= 5) / len(abserr) if abserr else None),
               boundary_within_20=(sum(1 for e in abserr if e <= 20) / len(abserr) if abserr else None),
               boundary_signed_median=(st.median(errs) if errs else None))
    json.dump(res, open(f"{HERE}/results_A.json", "w"), indent=1)

    L = [f"# Eval A — end-matter bibliography boundary ({len(rows)} doc tails)\n",
         "## (1) Found / not-found confusion (detector vs Opus)\n",
         "| | truth: has bib | truth: no bib |", "|---|---:|---:|",
         f"| detector: found | {TP} (TP) | {FP} (FP) |", f"| detector: not found | {FN} (FN) | {TN} (TN) |", "",
         f"**precision = {prec:.3f}  recall = {rec:.3f}  F1 = {f1:.3f}**  (n usable = {len(usable)})\n",
         "## (2) Boundary localisation (docs both found)\n",
         f"- n = {len(errs)}; median |error| = {res['boundary_median_abs']} lines; "
         f"within 5 lines = {None if res['boundary_within_5'] is None else round(res['boundary_within_5']*100)}%; "
         f"within 20 = {None if res['boundary_within_20'] is None else round(res['boundary_within_20']*100)}%; "
         f"signed median = {res['boundary_signed_median']} (─ = detector starts too early)\n"]
    # breakdown
    by = collections.defaultdict(lambda: collections.Counter())
    for r in usable:
        cls = "TP" if (r["det_found"] and r["true_found"]) else "FP" if r["det_found"] else "FN" if r["true_found"] else "TN"
        by[r["source"]][cls] += 1
    L.append("## By source\n| source | TP | FP | FN | TN |\n|---|---:|---:|---:|---:|")
    for s, c in by.items():
        L.append(f"| {s} | {c['TP']} | {c['FP']} | {c['FN']} | {c['TN']} |")
    open(f"{HERE}/CONFUSION_MATRIX_A.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score Eval-B: Opus annotations vs the detector's β-gate decision → confusion matrix.

truth_bib   = annotation.is_reference_list (any reference/bibliography list, incl. sub-lists,
              archival, web, further-reading). NOT-bib = colophon / CV publication list / prose /
              TOC / appendix / apparatus.
gate_bib    = manifest.gate_decision starts with "bib".
Outputs eval/results_B.json + eval/CONFUSION_MATRIX.md, with per-stratum / per-style / per-feature
error breakdowns (the tuning to-do), Cohen's κ on the double-annotated subset, and a verbatim-quote
hallucination check.
"""
import glob, json, os, re, collections, math

HERE = os.path.dirname(os.path.abspath(__file__))
BIB_KINDS = {"end_bibliography", "chapter_bibliography", "subdivided_sublist",
             "archival_primary_sources", "web_sources", "further_reading"}


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def load_annotations(paths):
    out = {}
    for p in paths:
        try:
            arr = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print(f"  WARN bad json {p}: {e}")
            continue
        for a in (arr if isinstance(arr, list) else [arr]):
            if "unit_id" in a:
                out[a["unit_id"]] = a
    return out


def truth_is_bib(a):
    # primary signal is the explicit boolean; fall back to kind
    if isinstance(a.get("is_reference_list"), bool):
        return a["is_reference_list"]
    return a.get("kind") in BIB_KINDS


def main():
    manifest = {json.loads(l)["unit_id"]: json.loads(l)
                for l in open(f"{HERE}/units/B_betagate_manifest.jsonl", encoding="utf-8") if l.strip()}
    units = {}
    for p in glob.glob(f"{HERE}/units/B_betagate/unit_*.json"):
        u = json.load(open(p, encoding="utf-8"))
        units[u["unit_id"]] = u
    prim = load_annotations(sorted(glob.glob(f"{HERE}/annotations/B_batch*.json")))
    dbl = load_annotations([f"{HERE}/annotations/B_double.json"])

    rows = []
    halluc = 0
    for uid, m in manifest.items():
        a = prim.get(uid)
        if not a:
            print(f"  MISSING annotation {uid}")
            continue
        tb = truth_is_bib(a)
        gb = m["gate_decision"].startswith("bib")
        # hallucination check: evidence_quote ⊂ section text (whitespace-normalised)
        sect = norm(units[uid]["section_numbered"]) if uid in units else ""
        eq = norm(a.get("evidence_quote", ""))
        ok_quote = (len(eq) >= 8 and eq[:60] in sect) or (len(eq) < 8)
        if not ok_quote:
            halluc += 1
        rows.append(dict(uid=uid, truth_bib=tb, gate_bib=gb, stratum=m["stratum"],
                         gate=m["gate_decision"], features=m["features"],
                         kind=a.get("kind"), style=a.get("citation_style"), lang=a.get("language"),
                         script=a.get("script"), subj=a.get("subject_register"),
                         conf=a.get("confidence"), quote_ok=ok_quote, ann=a))

    # confusion matrix (gate vs truth)
    TP = sum(1 for r in rows if r["gate_bib"] and r["truth_bib"])
    FP = sum(1 for r in rows if r["gate_bib"] and not r["truth_bib"])
    FN = sum(1 for r in rows if not r["gate_bib"] and r["truth_bib"])
    TN = sum(1 for r in rows if not r["gate_bib"] and not r["truth_bib"])
    n = len(rows)
    prec = TP / (TP + FP) if TP + FP else float("nan")
    rec = TP / (TP + FN) if TP + FN else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    acc = (TP + TN) / n if n else float("nan")

    # Cohen's kappa on double-annotated subset (is_reference_list agreement)
    kappa = None
    pairs = [(truth_is_bib(prim[u]), truth_is_bib(dbl[u])) for u in dbl if u in prim]
    if pairs:
        po = sum(1 for x, y in pairs if x == y) / len(pairs)
        p1a = sum(1 for x, _ in pairs if x) / len(pairs)
        p1b = sum(1 for _, y in pairs if y) / len(pairs)
        pe = p1a * p1b + (1 - p1a) * (1 - p1b)
        kappa = (po - pe) / (1 - pe) if (1 - pe) else 1.0

    # error breakdowns
    def err_cls(r):
        if r["gate_bib"] and not r["truth_bib"]: return "FP"
        if not r["gate_bib"] and r["truth_bib"]: return "FN"
        return "ok"
    fps = [r for r in rows if err_cls(r) == "FP"]
    fns = [r for r in rows if err_cls(r) == "FN"]
    by_stratum = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        by_stratum[r["stratum"]][err_cls(r)] += 1

    res = dict(n=n, TP=TP, FP=FP, FN=FN, TN=TN, precision=prec, recall=rec, f1=f1, accuracy=acc,
               kappa=kappa, hallucinations=halluc,
               by_stratum={k: dict(v) for k, v in by_stratum.items()},
               FP_detail=[dict(uid=r["uid"], gate=r["gate"], kind=r["kind"], style=r["style"],
                               feat=r["features"], quote=r["ann"].get("evidence_quote")) for r in fps],
               FN_detail=[dict(uid=r["uid"], gate=r["gate"], kind=r["kind"], style=r["style"],
                               lang=r["lang"], feat=r["features"], quote=r["ann"].get("evidence_quote")) for r in fns])
    json.dump(res, open(f"{HERE}/results_B.json", "w"), ensure_ascii=False, indent=1)

    # markdown report
    L = []
    L.append("# Eval B — β-gate confusion matrix (Opus 4.8 annotation vs detector)\n")
    L.append(f"n={n} sections · hallucination-flagged quotes={halluc} · "
             f"inter-annotator κ (is_reference_list, {len(pairs)} double)={kappa:.2f}\n" if kappa is not None
             else f"n={n}\n")
    L.append("## Confusion matrix (rows = detector gate, cols = Opus truth)\n")
    L.append("| | truth: bibliography | truth: NOT bib |")
    L.append("|---|---:|---:|")
    L.append(f"| **gate: bib (flag/drop)** | {TP} (TP) | {FP} (FP) |")
    L.append(f"| **gate: kept-non-bib** | {FN} (FN) | {TN} (TN) |")
    L.append("")
    L.append(f"**precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  accuracy={acc:.3f}**\n")
    L.append("Precision = of sections the gate flags as bibliography, how many truly are. "
             "Recall = of true bibliographies, how many the gate flags.\n")
    L.append("## Error breakdown by sampling stratum\n")
    L.append("| stratum | FP | FN | ok |")
    L.append("|---|---:|---:|---:|")
    for st in ("bib_strong", "bib_weak", "kept_hasyear", "kept_noyear"):
        v = by_stratum.get(st, {})
        L.append(f"| {st} | {v.get('FP',0)} | {v.get('FN',0)} | {v.get('ok',0)} |")
    L.append("")
    if fps:
        L.append("## False positives (gate said bib, Opus says NOT) — over-removal risk\n")
        for r in fps:
            L.append(f"- `{r['uid']}` gate=`{r['gate']}` → Opus kind=**{r['kind']}** style={r['style']} "
                     f"feat={r['features']}  «{(r['ann'].get('evidence_quote') or '')[:80]}»")
        L.append("")
    if fns:
        L.append("## False negatives (gate kept, Opus says bibliography) — under-removal\n")
        for r in fns:
            L.append(f"- `{r['uid']}` gate=`{r['gate']}` → Opus kind=**{r['kind']}** style={r['style']} "
                     f"lang={r['lang']} feat={r['features']}  «{(r['ann'].get('evidence_quote') or '')[:80]}»")
        L.append("")
    # tuning signal: which annotation features separate the FN/FP from the correct calls
    L.append("## Tuning signal — feature profile of the errors\n")
    for label, grp in (("FN (missed bib)", fns), ("FP (false bib)", fps)):
        if not grp: continue
        stys = collections.Counter(r["style"] for r in grp)
        kinds = collections.Counter(r["kind"] for r in grp)
        noyear = sum(1 for r in grp if not r["features"].get("yc"))
        L.append(f"- **{label}** (n={len(grp)}): kinds={dict(kinds)}; styles={dict(stys)}; "
                 f"detector saw year_count=0 in {noyear}/{len(grp)}.")
    open(f"{HERE}/CONFUSION_MATRIX.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

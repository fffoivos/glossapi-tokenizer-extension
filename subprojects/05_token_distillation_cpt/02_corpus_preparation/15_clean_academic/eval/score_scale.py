#!/usr/bin/env python3
"""Score the scaled Eval-B set: confusion matrix on the FROZEN TEST split, prevalence-weighted
to the corpus, with bootstrap CIs. Plus per-stratum rates, FP/FN kind breakdown, and feature
separability re-measured on the larger TRAIN split.

Reads:
  units/B_scale_manifest.jsonl       (gate_decision, stratum, weight, split)
  annotations_scale/all.json         (the collected Opus annotations)
  units/B_scale/batch_*.json         (section text, for features + hallucination check)
Writes eval/CONFUSION_MATRIX_SCALE.md + eval/results_scale.json
"""
import glob, json, os, re, math, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
BIB_KINDS = {"end_bibliography", "chapter_bibliography", "subdivided_sublist",
             "archival_primary_sources", "web_sources", "further_reading"}


def truth_bib(a):
    if isinstance(a.get("is_reference_list"), bool):
        return a["is_reference_list"]
    return a.get("kind") in BIB_KINDS


def weighted_cm(rows):
    TP = sum(r["w"] for r in rows if r["gate"] and r["truth"])
    FP = sum(r["w"] for r in rows if r["gate"] and not r["truth"])
    FN = sum(r["w"] for r in rows if not r["gate"] and r["truth"])
    TN = sum(r["w"] for r in rows if not r["gate"] and not r["truth"])
    prec = TP / (TP + FP) if TP + FP else float("nan")
    rec = TP / (TP + FN) if TP + FN else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    acc = (TP + TN) / (TP + FP + FN + TN)
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=prec, recall=rec, f1=f1, accuracy=acc)


def bootstrap(rows, B=1000):
    # deterministic bootstrap via hash-seeded LCG (no Math.random)
    n = len(rows)
    precs, recs = [], []
    seed = 12345
    for b in range(B):
        samp = []
        s = seed + b * 2654435761
        for _ in range(n):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            samp.append(rows[s % n])
        cm = weighted_cm(samp)
        if cm["precision"] == cm["precision"]:
            precs.append(cm["precision"])
        if cm["recall"] == cm["recall"]:
            recs.append(cm["recall"])
    def ci(v):
        v = sorted(v)
        return (v[int(0.025 * len(v))], v[int(0.975 * len(v))]) if v else (float("nan"),) * 2
    return ci(precs), ci(recs)


def main():
    manifest = {json.loads(l)["unit_id"]: json.loads(l)
                for l in open(f"{HERE}/units/B_scale_manifest.jsonl", encoding="utf-8") if l.strip()}
    ann = {}
    raw = json.load(open(f"{HERE}/annotations_scale/all.json", encoding="utf-8"))
    for a in (raw["annotations"] if isinstance(raw, dict) else raw):
        if "unit_id" in a:
            ann[a["unit_id"]] = a
    text = {}
    for p in glob.glob(f"{HERE}/units/B_scale/batch_*.json"):
        for u in json.load(open(p, encoding="utf-8")):
            text[u["unit_id"]] = (u.get("header_line", "") + "\n" + u.get("section_text", ""))

    rows = []
    missing = 0
    halluc = 0
    for uid, m in manifest.items():
        a = ann.get(uid)
        if not a:
            missing += 1
            continue
        tb = truth_bib(a)
        gb = m["gate_decision"].startswith("bib")
        eq = re.sub(r"\s+", " ", a.get("evidence_quote", "")).strip().lower()
        body = re.sub(r"\s+", " ", text.get(uid, "")).lower()
        if len(eq) >= 10 and eq[:50] not in body:
            halluc += 1
        rows.append(dict(uid=uid, truth=tb, gate=gb, w=m["weight"], stratum=m["stratum"],
                         split=m["split"], gate_dec=m["gate_decision"], kind=a.get("kind"),
                         style=a.get("citation_style"), lang=a.get("language"), script=a.get("script")))

    test = [r for r in rows if r["split"] == "test"]
    train = [r for r in rows if r["split"] == "train"]
    cm_test = weighted_cm(test)
    cm_train = weighted_cm(train)
    cm_all = weighted_cm(rows)
    (plo, phi), (rlo, rhi) = bootstrap(test)

    # unweighted (raw) on test for reference
    for r in rows:
        r["w1"] = 1.0
    cm_test_raw = weighted_cm([dict(r, w=1.0) for r in test])

    # per-stratum rates (unweighted within stratum)
    strata = {}
    for st in ("bib_strong", "bib_weak", "kept_hasyear", "kept_noyear"):
        srows = [r for r in rows if r["stratum"] == st]
        n = len(srows)
        if st.startswith("bib"):
            fp = sum(1 for r in srows if not r["truth"]); strata[st] = dict(n=n, fp_rate=fp / n if n else 0)
        else:
            fn = sum(1 for r in srows if r["truth"]); strata[st] = dict(n=n, fn_rate=fn / n if n else 0)

    fps = [r for r in rows if r["gate"] and not r["truth"]]
    fns = [r for r in rows if not r["gate"] and r["truth"]]
    import collections
    fp_kinds = collections.Counter(r["kind"] for r in fps)
    fn_kinds = collections.Counter((r["kind"], r["style"]) for r in fns)
    fn_lang = collections.Counter(r["lang"] for r in fns)

    res = dict(n_total=len(rows), n_test=len(test), n_train=len(train), missing=missing,
               hallucination_flags=halluc, cm_test=cm_test, cm_test_raw=cm_test_raw, cm_train=cm_train,
               cm_all=cm_all, prec_ci=[plo, phi], rec_ci=[rlo, rhi], strata=strata,
               fp_kinds=dict(fp_kinds), fn_lang=dict(fn_lang))
    json.dump(res, open(f"{HERE}/results_scale.json", "w"), ensure_ascii=False, indent=1)

    L = [f"# Eval B at scale — β-gate confusion matrix ({len(rows)} sections, {len(test)} held-out test)\n"]
    L.append(f"annotations matched={len(rows)} (missing {missing}) · hallucination-flagged quotes={halluc}\n")
    L.append("## Held-out TEST, prevalence-weighted to the 94,282-section corpus\n")
    c = cm_test
    L.append("| | truth: bib | truth: NOT |")
    L.append("|---|---:|---:|")
    L.append(f"| gate: bib | {c['TP']:.0f} | {c['FP']:.0f} |")
    L.append(f"| gate: kept | {c['FN']:.0f} | {c['TN']:.0f} |\n")
    L.append(f"**precision = {c['precision']:.3f}  (95% CI {plo:.3f}–{phi:.3f})**")
    L.append(f"**recall = {c['recall']:.3f}  (95% CI {rlo:.3f}–{rhi:.3f})**")
    L.append(f"**F1 = {c['f1']:.3f} · accuracy = {c['accuracy']:.3f}**\n")
    L.append(f"(unweighted on the balanced test sample: precision {cm_test_raw['precision']:.3f}, "
             f"recall {cm_test_raw['recall']:.3f})\n")
    L.append("## Per-stratum error rates (all units)\n")
    L.append("| stratum | n | error rate |")
    L.append("|---|---:|---:|")
    for st, v in strata.items():
        rate = v.get("fp_rate", v.get("fn_rate"))
        L.append(f"| {st} | {v['n']} | {('FP ' if 'fp_rate' in v else 'FN ')}{rate:.3f} |")
    L.append("")
    L.append(f"## False-positive kinds (gate=bib, truly not): {dict(fp_kinds)}\n")
    L.append(f"## False-negative language (gate=kept, truly bib): {dict(fn_lang)}\n")
    open(f"{HERE}/CONFUSION_MATRIX_SCALE.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

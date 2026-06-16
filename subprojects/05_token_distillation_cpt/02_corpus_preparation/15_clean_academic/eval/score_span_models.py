#!/usr/bin/env python3
"""Unified PRECISION-FIRST scorer for the bibliography-span arms (Rust header→EOF baseline, interpretable
line-LR, two-stage synergy, and later the DL CRF). Three views per the existing eval contract, scored
per doc on the doc-grouped TEST split:
  1. LINE precision/recall (HEADLINE) — fraction of REMOVED lines that are truly bib (FP = prose
     amputation); + recall.
  2. DETECTION precision/recall/F1 + Cohen's κ (per doc: a bib region present or not).
  3. Line-IoU (median, %≥0.8) and signed Δstart/Δend with the PROSE-EATING directions (Δstart<0 / Δend>0)
     split out from benign under-capture.
Doc-level paired bootstrap (precision-weighted Fβ=0.5) of each arm vs the Rust baseline, on the shared
greek_phd+openarchives docs. Slices by metadata. Writes results_span_models.json.
"""
import json, os, collections, statistics as st
import numpy as np
import span_seq_data as D
import decode_spans as DS
HERE = os.path.dirname(os.path.abspath(__file__))


def present(d):
    return {a for a, _ in d["lines"]}


def gold_lineset(d):
    g = set()
    for s in d["spans"]:
        a, b = s["start_line"], s["end_line"]
        if isinstance(a, int) and isinstance(b, int) and b >= a:
            g.update(range(a, b + 1))
    return g & present(d)


def regions(spans):
    sp = sorted((s, e) for s, e in spans if e >= s)
    out = []
    for s, e in sp:
        if out and s <= out[-1][1] + 10:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def fbeta(tp, fp, fn, beta=0.5):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0


def doc_metrics(d, pred_spans):
    """Return (tp,fp,fn, gold_has, pred_has, deltas[list of (dstart,dend,iou)])."""
    pres = present(d)
    gold = gold_lineset(d)
    pred = set()
    for s, e in pred_spans:
        pred.update(range(s, e + 1))
    pred &= pres
    tp = len(pred & gold); fp = len(pred - gold); fn = len(gold - pred)
    gold_regs = regions([(s["start_line"], s["end_line"]) for s in d["spans"]
                          if isinstance(s.get("start_line"), int) and isinstance(s.get("end_line"), int)
                          and s["end_line"] >= s["start_line"]])
    pred_regs = regions(pred_spans)
    deltas = []
    for gs, ge in gold_regs:
        if not pred_regs:
            continue
        best = max(pred_regs, key=lambda r: max(0, min(ge, r[1]) - max(gs, r[0])))
        ov = max(0, min(ge, best[1]) - max(gs, best[0]))
        if ov <= 0:
            continue
        union = max(ge, best[1]) - min(gs, best[0])
        iou = ov / union if union else 0.0
        deltas.append((best[0] - gs, best[1] - ge, iou))  # Δstart, Δend (pred-gold)
    return tp, fp, fn, bool(gold), bool(pred), deltas


def score(data, pred_by_doc, ids):
    TP = FP = FN = 0
    dtp = dfp = dfn = dtn = 0   # detection confusion
    ious = []; dstart = []; dend = []
    perdoc = []  # (tp,fp,fn) per doc for bootstrap
    for doc_id in ids:
        d = data.docs[doc_id]
        tp, fp, fn, gh, ph, deltas = doc_metrics(d, pred_by_doc.get(doc_id, []))
        TP += tp; FP += fp; FN += fn
        dtp += gh and ph; dfp += (not gh) and ph; dfn += gh and (not ph); dtn += (not gh) and (not ph)
        for ds, de, iou in deltas:
            ious.append(iou); dstart.append(ds); dend.append(de)
        perdoc.append((tp, fp, fn))
    prec = TP / (TP + FP) if TP + FP else 1.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    # detection
    dp = dtp / (dtp + dfp) if dtp + dfp else 0.0
    dr = dtp / (dtp + dfn) if dtp + dfn else 0.0
    n = dtp + dfp + dfn + dtn
    po = (dtp + dtn) / n if n else 0.0
    pe = ((dtp + dfn) * (dtp + dfp) + (dfp + dtn) * (dfn + dtn)) / (n * n) if n else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 1.0
    prose_eat = sum(1 for a, b in zip(dstart, dend) if a < -3 or b > 3)
    return dict(line_prec=prec, line_rec=rec, fp_rate=1 - prec, n_removed=TP + FP,
                det_prec=dp, det_rec=dr, det_f1=2 * dp * dr / (dp + dr) if dp + dr else 0.0, kappa=kappa,
                med_iou=st.median(ious) if ious else 0.0,
                iou_ge80=sum(i >= 0.8 for i in ious) / len(ious) if ious else 0.0,
                med_dstart=st.median(dstart) if dstart else 0.0, med_dend=st.median(dend) if dend else 0.0,
                prose_eat_rate=prose_eat / len(ious) if ious else 0.0, n_matched=len(ious),
                perdoc=perdoc)


def paired_boot(A, B, n_boot=2000):
    A = np.array(A, float); B = np.array(B, float); n = len(A); seed = 999; deltas = []
    for b in range(n_boot):
        s = seed + b * 2654435761; idx = np.empty(n, int)
        for k in range(n):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF; idx[k] = s % n
        fa = fbeta(*A[idx].sum(0)); fb = fbeta(*B[idx].sum(0))
        deltas.append(fb - fa)
    deltas.sort()
    return deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot)], float(np.mean(deltas))


def build_arms(data):
    params = json.load(open(f"{HERE}/span_smooth_params.json"))
    arms = {}
    # Rust baseline
    rb = json.load(open(f"{HERE}/units/SPAN_rust_baseline.json"))
    arms["rust_header_eof"] = {doc: [tuple(x) for x in v] for doc, v in rb.items()}
    # interpretable line-LR + synergy (decode cached pline with tuned hysteresis)
    for tag, syn in (("line_lr", False), ("synergy", True)):
        pline = DS.get_pline(data, synergy=syn)
        arms[tag] = {doc: DS.decode_doc(d, pline[doc], params) for doc, d in data.docs.items()}
    # optional DL arm if its decoded spans were dumped
    dl = f"{HERE}/units/SPAN_pred_dl.json"
    if os.path.exists(dl):
        arms["dl_crf"] = {doc: [tuple(x) for x in v] for doc, v in json.load(open(dl)).items()}
    return arms, params


def main():
    data = D.load()
    arms, params = build_arms(data)
    test = [doc for doc, d in data.docs.items() if d["split"] == "test"]
    gp_oa = [doc for doc in test if data.docs[doc]["source"] in ("greek_phd", "openarchives")]
    print(f"smoother params: {params}")
    print(f"test docs: {len(test)} (greek_phd+openarchives {len(gp_oa)}, where the Rust baseline applies)\n")

    def row(name, m):
        return (f"  {name:<16} line P {m['line_prec']:.3f} R {m['line_rec']:.3f} | FP-rate {m['fp_rate']:.3f}"
                f" | det P{m['det_prec']:.2f} R{m['det_rec']:.2f} κ{m['kappa']:.2f}"
                f" | IoU {m['med_iou']:.2f} (≥.8 {m['iou_ge80']:.0%}) | Δs {m['med_dstart']:+.0f} Δe {m['med_dend']:+.0f}"
                f" | prose-eat {m['prose_eat_rate']:.0%}")

    res = {"params": params, "n_test": len(test), "n_gp_oa": len(gp_oa)}
    print("=== ALL TEST SOURCES (our models; Rust N/A on kallipos) ===")
    for name in ("line_lr", "synergy"):
        m = score(data, arms[name], test); res[name + "_all"] = {k: v for k, v in m.items() if k != "perdoc"}
        print(row(name, m))
    print("\n=== greek_phd + openarchives TEST (head-to-head vs Rust baseline) ===")
    scored = {}
    for name in ("rust_header_eof", "line_lr", "synergy") + (("dl_crf",) if "dl_crf" in arms else ()):
        m = score(data, arms[name], gp_oa); scored[name] = m
        res[name + "_gpoa"] = {k: v for k, v in m.items() if k != "perdoc"}
        print(row(name, m))

    print("\n=== paired bootstrap: precision-weighted Fβ0.5 vs rust_header_eof (gp+oa test docs) ===")
    base = scored["rust_header_eof"]["perdoc"]
    res["bootstrap_vs_rust"] = {}
    for name in ("line_lr", "synergy") + (("dl_crf",) if "dl_crf" in arms else ()):
        lo, hi, mn = paired_boot(base, scored[name]["perdoc"])
        sig = "REAL (CI excludes 0)" if (lo > 0 or hi < 0) else "noise (spans 0)"
        res["bootstrap_vs_rust"][name] = [lo, mn, hi]
        print(f"  {name:<16} ΔFβ0.5 {mn:+.3f}  [95% {lo:+.3f}, {hi:+.3f}]  → {sig}")

    # slices by metadata (line P/R) on all-test for the best interpretable arm
    print("\n=== line_lr sliced by metadata (all test sources) ===")
    best = arms["line_lr"]
    slices = collections.defaultdict(lambda: [0, 0, 0])  # key -> [tp,fp,fn]
    for doc_id in test:
        d = data.docs[doc_id]
        # assign each gold line the metadata of the span covering it (first match)
        line_meta = {}
        for s in d["spans"]:
            a, b = s.get("start_line"), s.get("end_line")
            if isinstance(a, int) and isinstance(b, int):
                for li in range(a, b + 1):
                    line_meta.setdefault(li, s)
        pres = present(d); gold = gold_lineset(d)
        pred = set()
        for s, e in best.get(doc_id, []):
            pred.update(range(s, e + 1))
        pred &= pres
        for axis in ("script", "noise_level", "kind"):
            for li in gold:
                key = (axis, line_meta.get(li, {}).get(axis, "?"))
                slices[key][2 if li not in pred else 0] += 1  # fn or tp
    sl = {}
    for (axis, val), (tp, fp, fn) in sorted(slices.items()):
        r = tp / (tp + fn) if tp + fn else 0
        sl[f"{axis}:{val}"] = round(r, 3)
        print(f"  {axis:<12} {val:<18} recall {r:.3f}  ({tp+fn} gold lines)")
    res["line_lr_slice_recall"] = sl

    json.dump(res, open(f"{HERE}/results_span_models.json", "w"), indent=1)
    print("\nwrote results_span_models.json")


if __name__ == "__main__":
    main()

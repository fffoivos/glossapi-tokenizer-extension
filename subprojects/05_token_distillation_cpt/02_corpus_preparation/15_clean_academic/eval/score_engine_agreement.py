#!/usr/bin/env python3
"""Bake-off: does gpt-5.5 reproduce Opus's ToC/bibliography annotations? Opus = reference.

Compares two annotation dirs over the SAME batch_*.json inputs (the 10 docs we already have Opus
annotations for). Three views, the same precision-first lens the cleaner uses (eating main text is the
costly error):
  1. PER-LINE agreement on {0 other, 1 bibliography, 2 table_of_contents} — overall accuracy, Cohen's κ,
     and per-class precision/recall with Opus as truth. HEADLINE = bib/toc PRECISION: of the lines gpt-5.5
     marks as scaffolding, what fraction Opus agrees are scaffolding (1 − precision = prose it would eat).
  2. SECTION detection P/R/F1 at ≥0.5 line-IoU (per kind) — did it find the same ToC/bibliography blocks.
  3. Signed boundary Δ on matched spans — Δstart<0 / Δend>0 are over-capture (eating prose); report
     median + %within ±3/±5 lines.

Local only — reads JSON, no model calls.
  Usage: score_engine_agreement.py [--opus-dir units/STRUCT2_FT] [--cand-dir units/BAKEOFF_gpt55]
"""
import argparse, glob, json, os, re, collections
HERE = os.path.dirname(os.path.abspath(__file__))
LINE_RE = re.compile(r"^L(\d+):\s?(.*)$")
LABEL = {"other": 0, "bibliography": 1, "table_of_contents": 2}
NAME = {0: "other", 1: "bibliography", 2: "table_of_contents"}


def present_lines(text):
    out = []
    for ln in text.split("\n"):
        g = LINE_RE.match(ln)
        if g and g.group(2).strip() and "elided" not in g.group(2):
            out.append(int(g.group(1)))
    return out


def label_map(ann):
    """abs line -> label (2 toc / 1 bib) from an annotation's sections."""
    lab = {}
    for s in ann.get("sections", []):
        v = LABEL.get(s.get("kind"), 0)
        if v == 0:
            continue
        for n in range(int(s["start_line"]), int(s["end_line"]) + 1):
            lab[n] = v
    return lab


def members(section, present_set):
    return {n for n in present_set if int(section["start_line"]) <= n <= int(section["end_line"])}


def iou(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def kappa(cm):
    N = sum(cm.values()) or 1
    po = sum(v for (i, j), v in cm.items() if i == j) / N
    rows = collections.Counter(); cols = collections.Counter()
    for (i, j), v in cm.items():
        rows[i] += v; cols[j] += v
    pe = sum((rows[k] / N) * (cols[k] / N) for k in set(rows) | set(cols))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opus-dir", default="units/STRUCT2_FT")
    ap.add_argument("--cand-dir", default="units/BAKEOFF_gpt55")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()
    opus_dir = a.opus_dir if os.path.isabs(a.opus_dir) else os.path.join(HERE, a.opus_dir)
    cand_dir = a.cand_dir if os.path.isabs(a.cand_dir) else os.path.join(HERE, a.cand_dir)

    cm = collections.Counter()                       # (opus_label, cand_label) -> count
    det = {1: [0, 0, 0], 2: [0, 0, 0]}               # kind -> [TP, FP, FN]
    deltas = []                                       # (kind, dstart, dend)
    n_docs = 0; counts = {"opus": collections.Counter(), "cand": collections.Counter()}
    missing = []

    for op in sorted(glob.glob(f"{opus_dir}/ann_*.json")):
        idx = re.search(r"ann_(\d+)\.json$", op).group(1)
        cp = f"{cand_dir}/ann_{idx}.json"
        bp = f"{opus_dir}/batch_{idx}.json"
        if not (os.path.exists(cp) and os.path.exists(bp)):
            missing.append(idx); continue
        O = json.load(open(op)); C = json.load(open(cp)); U = json.load(open(bp))[0]
        pres = present_lines(U["text_numbered"]); pset = set(pres)
        lo, lc = label_map(O), label_map(C)
        for n in pres:
            cm[(lo.get(n, 0), lc.get(n, 0))] += 1
        for kind in (1, 2):
            os_ = [s for s in O.get("sections", []) if LABEL.get(s.get("kind")) == kind]
            cs_ = [s for s in C.get("sections", []) if LABEL.get(s.get("kind")) == kind]
            counts["opus"][kind] += len(os_); counts["cand"][kind] += len(cs_)
            om = [members(s, pset) for s in os_]; cmem = [members(s, pset) for s in cs_]
            # greedy max-IoU matching
            pairs = sorted(((iou(om[i], cmem[k]), i, k) for i in range(len(os_)) for k in range(len(cs_))),
                           reverse=True)
            used_o, used_c = set(), set()
            for v, i, k in pairs:
                if v < a.iou or i in used_o or k in used_c:
                    continue
                used_o.add(i); used_c.add(k)
                det[kind][0] += 1
                deltas.append((kind,
                               int(cs_[k]["start_line"]) - int(os_[i]["start_line"]),
                               int(cs_[k]["end_line"]) - int(os_[i]["end_line"])))
            det[kind][1] += len(cs_) - len(used_c)   # FP (cand unmatched)
            det[kind][2] += len(os_) - len(used_o)   # FN (opus unmatched)
        n_docs += 1

    # ---- report ----
    print(f"BAKE-OFF  gpt-5.5 (cand) vs Opus (reference)   docs={n_docs}"
          + (f"  [missing cand for idx {missing}]" if missing else ""))
    k, po = kappa(cm)
    Ntot = sum(cm.values()) or 1
    print(f"\n1) PER-LINE  ({Ntot} present lines)   accuracy={po:.3f}   Cohen κ={k:.3f}")
    for cls in (2, 1):  # toc, bib
        tp = cm[(cls, cls)]
        cand_pos = sum(v for (i, j), v in cm.items() if j == cls)
        opus_pos = sum(v for (i, j), v in cm.items() if i == cls)
        prec = tp / cand_pos if cand_pos else float("nan")
        rec = tp / opus_pos if opus_pos else float("nan")
        eat = (1 - prec) if cand_pos else float("nan")
        print(f"   {NAME[cls]:<18} precision={prec:.3f}  recall={rec:.3f}   "
              f"(prose-eaten if marked {NAME[cls]} = {eat:.3f})   opus_lines={opus_pos} cand_lines={cand_pos}")

    print(f"\n2) SECTION detection (IoU≥{a.iou})")
    for kind in (2, 1):
        tp, fp, fn = det[kind]
        P = tp / (tp + fp) if tp + fp else float("nan")
        R = tp / (tp + fn) if tp + fn else float("nan")
        F = 2 * P * R / (P + R) if P + R else float("nan")
        print(f"   {NAME[kind]:<18} P={P:.3f} R={R:.3f} F1={F:.3f}   "
              f"(opus {counts['opus'][kind]} / gpt5.5 {counts['cand'][kind]} sections; TP={tp} FP={fp} FN={fn})")

    print(f"\n3) BOUNDARY Δ on matched spans (lines; <0 start / >0 end = over-capture / prose-eating)")
    for kind in (2, 1):
        ds = [d[1] for d in deltas if d[0] == kind]
        de = [d[2] for d in deltas if d[0] == kind]
        if not ds:
            print(f"   {NAME[kind]:<18} (no matched spans)"); continue
        def med(x): return sorted(x)[len(x) // 2]
        def within(x, t): return sum(1 for v in x if abs(v) <= t) / len(x)
        print(f"   {NAME[kind]:<18} Δstart med={med(ds):+d} (±3:{within(ds,3):.0%} ±5:{within(ds,5):.0%})   "
              f"Δend med={med(de):+d} (±3:{within(de,3):.0%} ±5:{within(de,5):.0%})   "
              f"eats-before={sum(1 for v in ds if v<0)}/{len(ds)} runs-past={sum(1 for v in de if v>0)}/{len(de)}")

    print("\nGO/NO-GO guide: proceed to the 2000 on gpt-5.5 if κ is high and bib/toc PRECISION is high "
          "(low prose-eaten) — recall/boundary slack is recoverable; eating main text is not.")
    out = os.path.join(cand_dir, "_bakeoff_report.json")
    json.dump({"n_docs": n_docs, "accuracy": po, "kappa": k,
               "confusion": {f"{i}->{j}": v for (i, j), v in cm.items()},
               "detection": {NAME[kk]: det[kk] for kk in det},
               "counts": {e: dict(counts[e]) for e in counts}},
              open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare Haiku vs Opus on the SAME bibliography-window annotations (Opus = reference).
Three independent views: (1) detection agreement per window (does it see a bibliography at all),
(2) span overlap = line-level IoU where both see one, (3) boundary Δ (Haiku region start/end − Opus's)."""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))


def lines(a):
    s = set()
    for sp in a.get("bib_lists", []):
        if isinstance(sp.get("start_line"), int) and isinstance(sp.get("end_line"), int) and sp["end_line"] >= sp["start_line"]:
            s |= set(range(sp["start_line"], sp["end_line"] + 1))
    return s


def main():
    opus = {a["unit_id"]: a for a in json.load(open(f"{HERE}/annotations_goal_windows/all.json"))["annotations"]}
    haiku = {a["unit_id"]: a for a in json.load(open(f"{HERE}/annotations_goal_windows_haiku/all.json"))["annotations"]}
    shared = sorted(set(opus) & set(haiku))
    print(f"windows: opus={len(opus)} haiku={len(haiku)} shared={len(shared)}")

    # (1) detection agreement
    tp = fp = fn = tn = 0
    ious = []; dstart = []; dend = []
    for u in shared:
        ol, hl = lines(opus[u]), lines(haiku[u])
        oh, hh = bool(ol), bool(hl)
        tp += oh and hh; fp += (not oh) and hh; fn += oh and (not hh); tn += (not oh) and (not hh)
        if ol or hl:
            ious.append(len(ol & hl) / len(ol | hl))
        if ol and hl:
            dstart.append(min(hl) - min(ol)); dend.append(max(hl) - max(ol))
    n = len(shared)
    po = (tp + tn) / n
    p_yes_o = (tp + fn) / n; p_yes_h = (tp + fp) / n
    pe = p_yes_o * p_yes_h + (1 - p_yes_o) * (1 - p_yes_h)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 1.0
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0

    print("\n=== (1) detection: does Haiku see a bibliography where Opus does? (Opus=truth) ===")
    print(f"  agree {po*100:.0f}% · Cohen's κ = {kappa:.2f}")
    print(f"  Haiku vs Opus  precision {prec:.2f}  recall {rec:.2f}  (TP {tp} FP {fp} FN {fn} TN {tn})")
    print("\n=== (2) span overlap where ≥1 sees a bib (line-IoU) ===")
    if ious:
        ious.sort()
        print(f"  median IoU {st.median(ious):.2f}  ·  ≥0.8: {sum(i>=0.8 for i in ious)}/{len(ious)} ({sum(i>=0.8 for i in ious)*100//len(ious)}%)"
              f"  ·  ≥0.5: {sum(i>=0.5 for i in ious)}/{len(ious)}")
    print("\n=== (3) boundary Δ (Haiku − Opus, lines; both saw a bib) ===")
    if dstart:
        def d(v, nm):
            v = sorted(v); print(f"  Δ{nm}: median {st.median(v):+.0f} · within ±3 lines {sum(abs(x)<=3 for x in v)}/{len(v)} ({sum(abs(x)<=3 for x in v)*100//len(v)}%) · range [{v[0]:+},{v[-1]:+}]")
        d(dstart, "start"); d(dend, "end")
    json.dump(dict(n=n, agree=po, kappa=kappa, precision=prec, recall=rec,
                   median_iou=st.median(ious) if ious else None,
                   median_dstart=st.median(dstart) if dstart else None,
                   median_dend=st.median(dend) if dend else None),
              open(f"{HERE}/results_haiku_vs_opus.json", "w"), indent=1)


if __name__ == "__main__":
    main()

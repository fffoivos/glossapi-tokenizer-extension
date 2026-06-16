#!/usr/bin/env python3
"""Per-window (capture, overrun) in CHARACTERS, scattered to reveal failure-mode clusters.
  capture = chars(detector ∩ true_bib) / chars(true_bib)   in [0,1]   (under-capture axis)
  overrun = chars(detector \ true_bib) / chars(true_bib)   >= 0       (over-reach axis)
Only windows where the detector flagged AND Opus marked bibliography become points.
Misses (det empty, bib present) and false-regions (det present, no bib) are reported as counts.
"""
import json, os, re, glob, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    man = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{HERE}/units/W_windows_manifest.jsonl") if l.strip()}
    ann = {a["unit_id"]: a for a in json.load(open(f"{HERE}/annotations_goal_windows/all.json"))["annotations"]}
    # char length per line, per window, from the unit text
    linelen = {}
    for p in glob.glob(f"{HERE}/units/W_windows/batch_*.json"):
        for u in json.load(open(p)):
            d = {}
            for ln in u["text_numbered"].split("\n"):
                m = re.match(r"L(\d+): ?(.*)", ln)
                if m:
                    d[int(m.group(1))] = len(m.group(2))
            linelen[u["unit_id"]] = d

    pts = []  # (capture, overrun, source, window)
    n_miss = n_false = n_match = 0
    for uid, a in ann.items():
        m = man.get(uid)
        if not m or uid not in linelen:
            continue
        LL = linelen[uid]
        det = set(m.get("detector_bib_lines_in_window", []))
        B = set()
        for s in a.get("bib_lists", []):
            if isinstance(s.get("start_line"), int) and isinstance(s.get("end_line"), int) and s["end_line"] >= s["start_line"]:
                B |= set(range(s["start_line"], s["end_line"] + 1))
        cB = sum(LL.get(l, 0) for l in B)
        cInter = sum(LL.get(l, 0) for l in (det & B))
        cOut = sum(LL.get(l, 0) for l in (det - B))
        if det and not B:
            n_false += 1; continue
        if B and not det:
            n_miss += 1; continue
        if not det or cB == 0:
            continue
        n_match += 1
        pts.append((cInter / cB, cOut / cB, m["source"], m["window"]))

    # scatter
    fig, ax = plt.subplots(figsize=(7.5, 6))
    CAP = 4.0  # clip overrun for display
    col = {"greek_phd": "#1f77b4", "openarchives": "#d62728"}
    mk = {"tail": "o", "body": "^"}
    for src in col:
        for win in mk:
            xs = [c for c, o, s, w in pts if s == src and w == win]
            ys = [min(o, CAP) for c, o, s, w in pts if s == src and w == win]
            clip = [o > CAP for c, o, s, w in pts if s == src and w == win]
            ax.scatter(xs, ys, c=col[src], marker=mk[win], s=46, alpha=0.6,
                       edgecolors=["k" if cl else "none" for cl in clip], linewidths=1.0,
                       label=f"{src} / {win}")
    ax.axhline(0.25, color="gray", ls=":", lw=0.8); ax.axvline(0.8, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("capture  =  chars(detector ∩ bib) / chars(bib)   →  under-capture ←")
    ax.set_ylabel("overrun  =  chars(detector \\ bib) / chars(bib)   (clipped at 4; black edge = clipped)")
    ax.set_title(f"β-boundary span quality per window  (n={n_match} matched; {n_miss} missed, {n_false} false-region)")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.15, CAP + 0.2)
    # quadrant labels
    ax.text(0.98, 0.05, "clean hit", ha="right", color="green", fontsize=9)
    ax.text(0.98, CAP - 0.2, "swallowed appendix", ha="right", va="top", color="darkred", fontsize=9)
    ax.text(0.02, 0.05, "short / partial", ha="left", color="orange", fontsize=9)
    ax.text(0.02, CAP - 0.2, "misaligned", ha="left", va="top", color="purple", fontsize=9)
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(f"{HERE}/plots", exist_ok=True)
    fig.savefig(f"{HERE}/plots/capture_overrun.png", dpi=130)

    # text summary of clusters
    def frac(pred):
        return sum(1 for c, o, s, w in pts if pred(c, o)) / max(1, len(pts))
    print(f"matched windows plotted: {n_match} | missed (det empty): {n_miss} | false-region (no bib): {n_false}")
    print(f"  clean hit   (capture≥0.8, overrun≤0.25): {frac(lambda c,o: c>=0.8 and o<=0.25)*100:.0f}%")
    print(f"  appendix    (capture≥0.8, overrun >0.25): {frac(lambda c,o: c>=0.8 and o>0.25)*100:.0f}%")
    print(f"  short/part  (capture<0.8, overrun≤0.25): {frac(lambda c,o: c<0.8 and o<=0.25)*100:.0f}%")
    print(f"  misaligned  (capture<0.8, overrun >0.25): {frac(lambda c,o: c<0.8 and o>0.25)*100:.0f}%")
    print(f"  median capture={sorted(c for c,_,_,_ in pts)[len(pts)//2]:.2f}  median overrun={sorted(o for _,o,_,_ in pts)[len(pts)//2]:.2f}")
    print(f"saved plots/capture_overrun.png")


if __name__ == "__main__":
    main()

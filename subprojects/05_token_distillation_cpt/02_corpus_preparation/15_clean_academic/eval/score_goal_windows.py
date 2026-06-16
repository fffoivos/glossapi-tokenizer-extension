#!/usr/bin/env python3
"""Score the goal-eval whole-doc windows: LINE-level total-bibliography recall/precision of the
end-matter detector, split by window type. Body-window recall exposes the end-of-chapter / mid-doc
bibliographies the end-matter-only detector structurally misses.
"""
import json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    man = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{HERE}/units/W_windows_manifest.jsonl") if l.strip()}
    raw = json.load(open(f"{HERE}/annotations_goal_windows/all.json"))
    ann = {a["unit_id"]: a for a in (raw["annotations"] if isinstance(raw, dict) else raw)}

    # aggregate line-level intersection by (source, window)
    agg = collections.defaultdict(lambda: dict(inter=0, opus=0, det=0))
    overall = collections.defaultdict(lambda: dict(inter=0, opus=0, det=0))
    for uid, m in man.items():
        a = ann.get(uid)
        if not a:
            continue
        det = set(m.get("detector_bib_lines_in_window", []))
        opus = set()
        for span in a.get("bib_lists", []):
            s, e = span.get("start_line"), span.get("end_line")
            if isinstance(s, int) and isinstance(e, int) and e >= s:
                opus |= set(range(s, e + 1))
        inter = len(opus & det)
        for key in [(m["source"], m["window"]), (m["source"], "ALL")]:
            agg[key]["inter"] += inter; agg[key]["opus"] += len(opus); agg[key]["det"] += len(det)
        overall[m["window"]]["inter"] += inter; overall[m["window"]]["opus"] += len(opus); overall[m["window"]]["det"] += len(det)

    def pr(d):
        rec = d["inter"] / d["opus"] if d["opus"] else float("nan")
        prec = d["inter"] / d["det"] if d["det"] else float("nan")
        return rec, prec

    L = ["# Goal-eval — whole-doc windows: LINE-level total bibliography recall/precision\n",
         "Recall = fraction of Opus-marked bibliography lines the detector flagged. "
         "Body windows reveal end-of-chapter / mid-doc bibliographies the end-matter detector misses.\n",
         "| source | window | bib-recall | precision | opus-bib-lines |", "|---|---|---:|---:|---:|"]
    for (src, win), d in sorted(agg.items()):
        rec, prec = pr(d)
        L.append(f"| {src} | {win} | {rec:.2f} | {prec:.2f} | {d['opus']} |")
    L.append("\n## By window type (both sources)\n| window | bib-recall | precision |\n|---|---:|---:|")
    for win, d in overall.items():
        rec, prec = pr(d); L.append(f"| {win} | {rec:.2f} | {prec:.2f} |")
    L.append("\n**Read:** high tail-recall = end-matter lists caught; low body-recall = end-of-chapter / "
             "mid-doc bibliography the detector structurally misses (the goal-vs-component gap).")
    open(f"{HERE}/CONFUSION_MATRIX_GOAL_WINDOWS.md", "w", encoding="utf-8").write("\n".join(L))
    json.dump({f"{k[0]}/{k[1]}": dict(zip(["recall", "precision"], pr(v))) for k, v in agg.items()},
              open(f"{HERE}/results_goal_windows.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()

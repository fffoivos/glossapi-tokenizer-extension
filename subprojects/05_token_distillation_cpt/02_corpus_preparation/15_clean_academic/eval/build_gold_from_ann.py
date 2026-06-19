#!/usr/bin/env python3
"""Bridge the front+tail annotation outputs → a per-line labeled gold set for retraining. For each doc,
every present (front+tail) line gets a label: 2=table_of_contents, 1=bibliography (incl. chapter/CV), 0=other
(main text). The elided middle is simply absent (unlabeled, by design — the model generalises from entry
features and infers on the full doc). Carries the doc-grouped split. Local only — no Opus.
  Usage: build_gold_from_ann.py units/STRUCT_RUN  ->  units/STRUCT_RUN_gold.jsonl"""
import json, os, re, glob, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
LINE_RE = re.compile(r"^L(\d+):\s?(.*)$")
LABEL = {"other": 0, "bibliography": 1, "table_of_contents": 2}


def present(text):
    out = []
    for ln in text.split("\n"):
        g = LINE_RE.match(ln)
        if g and g.group(2).strip() and "elided" not in g.group(2):
            out.append((int(g.group(1)), g.group(2)))
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "units/STRUCT_RUN"
    if not os.path.isabs(d):
        d = os.path.join(HERE, d)
    out_path = d.rstrip("/") + "_gold.jsonl"
    rows, counts, splits = [], collections.Counter(), collections.Counter()
    for ap in sorted(glob.glob(f"{d}/ann_*.json")):
        try:
            A = json.load(open(ap))
        except Exception:
            continue
        bp = ap.replace("ann_", "batch_")
        if not os.path.exists(bp):
            continue
        U = json.load(open(bp))[0]
        pl = present(U["text_numbered"])
        # label map per absolute line
        lab = {}
        for s in A.get("sections", []):
            v = LABEL.get(s["kind"], 0)
            for n in range(s["start_line"], s["end_line"] + 1):
                lab[n] = v
        lines = [[n, t, lab.get(n, 0)] for n, t in pl]
        split = U.get("split") or A.get("split") or "train"
        rows.append({"doc_id": A["doc_id"], "source": A.get("source", U.get("source")),
                     "split": split, "n_lines": U.get("n_lines") or A.get("n_lines"),
                     "mode": U.get("mode") or A.get("mode"), "lines": lines})
        splits[split] += 1
        for _, _, v in lines:
            counts[v] += 1
    open(out_path, "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    tot = sum(counts.values()) or 1
    print(f"{len(rows)} docs → {out_path}   (train {splits['train']} / test {splits['test']})")
    print(f"line labels: other {counts[0]} ({100*counts[0]//tot}%) · "
          f"bibliography {counts[1]} ({100*counts[1]//tot}%) · table_of_contents {counts[2]} ({100*counts[2]//tot}%)")
    print("retrain: line_lr.py adapts to multi-class {0,1,2}; decode_spans.py runs per class. See ANNOTATION_RUNBOOK.md")


if __name__ == "__main__":
    main()

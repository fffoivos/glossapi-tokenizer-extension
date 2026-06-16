#!/usr/bin/env python3
"""Build a stratified sample of REMOVED-but-not-gold lines (the model's 'false positives') WITH context,
for an independent Opus adjudication that validates the prose-protection claim and catches annotation
misses (an 'FP' that is actually bibliography the windowed Opus annotation missed → the model is more
right than measured). Includes ALL alleged running-prose removals + a sample of each citation type + a
control sample of correct bib removals. Writes units/FPVAL/batch_*.json + FPVAL_manifest.jsonl."""
import os, json, collections, random
import span_seq_data as D
import span_signals as S
import decode_spans as DS
import failure_analysis as FA
HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(0)
PER_TYPE = {"prose": 999, "footnote_prose": 999, "short/numeric": 30, "caption": 999, "toc": 999,
            "header": 25, "entry_like(non-target)": 45, "footnote_citation": 35, "table": 30, "web/url": 25,
            "entry_frontmatter": 999, "inline_cite_prose": 999, "_correct_bib": 25}


def ctx(lines, i, k=6):
    a, b = max(0, i - k), min(len(lines), i + k + 1)
    out = []
    for j in range(a, b):
        mark = "  >>> TARGET >>> " if j == i else "                 "
        out.append(f"{mark}L{lines[j][0]:05d}: {lines[j][1][:160]}")
    return "\n".join(out)


def main():
    data = D.load()
    pline = DS.get_pline(data)
    params = json.load(open(f"{HERE}/span_smooth_params.json"))
    pool = collections.defaultdict(list)
    for doc_id, d in data.docs.items():
        if d["split"] != "test":
            continue
        gold = set()
        for s in d["spans"]:
            a, b = s.get("start_line"), s.get("end_line")
            if isinstance(a, int) and isinstance(b, int) and b >= a:
                gold.update(range(a, b + 1))
        pred = set()
        for s, e in DS.decode_doc(d, pline[doc_id], params):
            pred.update(range(s, e + 1))
        lines = d["lines"]; idx = {a: i for i, (a, _) in enumerate(lines)}; N = d["N"] or 1
        for abs_idx in pred:
            if abs_idx not in idx:   # blank/absent line inside a span range
                continue
            i = idx[abs_idx]; t = lines[i][1]
            cat = "_correct_bib" if abs_idx in gold else FA.categorize_fp(t, S.line_signals(t), abs_idx / N)
            pool[cat].append((doc_id, abs_idx, ctx(lines, i)))
    units, man = [], []; ctr = 0
    for cat, items in pool.items():
        random.shuffle(items)
        for doc_id, abs_idx, context in items[:PER_TYPE.get(cat, 0)]:
            uid = f"V{ctr:04d}"; ctr += 1
            units.append({"unit_id": uid, "context": context})
            man.append({"unit_id": uid, "doc_id": doc_id, "abs_idx": abs_idx, "heuristic_cat": cat})
    random.shuffle(units)
    udir = f"{HERE}/units/FPVAL"; os.makedirs(udir, exist_ok=True)
    for f in os.listdir(udir):
        os.remove(os.path.join(udir, f))
    paths = []
    for bi in range(0, len(units), 25):
        p = f"{udir}/batch_{bi//25:03d}.json"; json.dump(units[bi:bi + 25], open(p, "w"), ensure_ascii=False); paths.append(p)
    open(f"{HERE}/units/FPVAL_manifest.jsonl", "w").write("\n".join(json.dumps(m, ensure_ascii=False) for m in man))
    json.dump(paths, open(f"{HERE}/units/FPVAL_batchpaths.json", "w"))
    print(f"{len(units)} removed lines sampled in {len(paths)} batches; by heuristic cat:",
          dict(collections.Counter(m["heuristic_cat"] for m in man)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assemble the bibliography-span dataset from the Opus annotations + report the distributions.
Each span becomes a labelled row {doc_id, source, window, start_line, end_line, + metadata}.
Writes eval/span_dataset.jsonl and prints kind/style/language/script/subject/noise/n_entries/has_header
distributions + spans-per-document multiplicity."""
import json, os, collections, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    man = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{HERE}/units/SPAN_manifest.jsonl") if l.strip()}
    raw = json.load(open(f"{HERE}/annotations_span/all.json"))
    ann = raw["annotations"] if isinstance(raw, dict) else raw

    rows = []
    for a in ann:
        m = man.get(a["unit_id"])
        if not m:
            continue
        for s in a.get("spans", []):
            if not (isinstance(s.get("start_line"), int) and isinstance(s.get("end_line"), int) and s["end_line"] >= s["start_line"]):
                continue
            rows.append(dict(doc_id=m["doc_id"], source=m["source"], window=m["window"],
                             start_line=s["start_line"], end_line=s["end_line"], **{k: s.get(k) for k in
                             ("kind", "citation_style", "language", "script", "subject_register", "noise_level", "n_entries", "has_header")}))
    with open(f"{HERE}/span_dataset.jsonl", "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"=== bibliography-span dataset: {n} spans from {len(set(r['doc_id'] for r in rows))} documents ===")
    print(f"by source: {dict(collections.Counter(r['source'] for r in rows))}")
    def dist(k):
        c = collections.Counter(r[k] for r in rows)
        print(f"  {k:<16}: " + " · ".join(f"{kk}={vv}" for kk, vv in c.most_common()))
    for k in ("kind", "citation_style", "language", "script", "subject_register", "noise_level", "has_header"):
        dist(k)
    ne = [r["n_entries"] for r in rows if isinstance(r["n_entries"], int)]
    if ne:
        print(f"  n_entries        : median {st.median(ne)} · mean {sum(ne)/len(ne):.0f} · max {max(ne)} · total ≈ {sum(ne)}")
    # spans per document (multiplicity, now full)
    perdoc = collections.Counter()
    bydoc = collections.defaultdict(list)
    for r in rows:
        bydoc[r["doc_id"]].append(r)
    for doc, rs in bydoc.items():
        perdoc[min(len(rs), 5)] += 1   # cap label at 5+
    print("\nspans per document (capped at 5+):")
    for k in sorted(perdoc):
        print(f"   {perdoc[k]:>4} docs -> {k}{'+' if k == 5 else ''} span(s)")
    multi = sum(v for k, v in perdoc.items() if k >= 2)
    print(f"→ docs with MULTIPLE bibliography spans: {multi}/{len(bydoc)} = {multi*100//len(bydoc)}%")
    json.dump({"n_spans": n, "n_docs": len(bydoc), "multi_pct": multi * 100 // len(bydoc)},
              open(f"{HERE}/results_span_dataset.json", "w"), indent=1)


if __name__ == "__main__":
    main()

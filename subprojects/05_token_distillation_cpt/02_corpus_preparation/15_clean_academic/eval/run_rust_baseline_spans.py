#!/usr/bin/env python3
"""Extract the current Rust detector's header→EOF bibliography boundary (`endmatter_bib`) for the
documents in the SPAN dataset, into the common span format, so it can be scored as Baseline (b) by
score_span_models.py. greek_phd + openarchives only (the whole-doc header→EOF detector's domain;
Kallipos is section-based and emits no endmatter_bib). Writes units/SPAN_rust_baseline.json
= {doc_id: [[start_line, end_line], ...]}."""
import json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    want = {json.loads(l)["doc_id"] for l in open(f"{HERE}/units/SPAN_manifest.jsonl") if l.strip()}
    out = collections.defaultdict(list)
    for src in ("greek_phd", "openarchives"):
        f = f"{ROOT}/out/{src}_full/refspans/{src}.spans.jsonl"
        if not os.path.exists(f):
            continue
        n = 0
        for line in open(f):
            if "endmatter_bib" not in line:          # cheap pre-filter before json.loads
                continue
            d = json.loads(line)
            if d.get("kind") == "endmatter_bib" and d["doc_id"] in want:
                out[d["doc_id"]].append([d["line_start"], d["line_end"]])
                n += 1
        print(f"{src}: {n} endmatter_bib for SPAN docs")
    json.dump(out, open(f"{HERE}/units/SPAN_rust_baseline.json", "w"))
    print(f"wrote units/SPAN_rust_baseline.json ({len(out)} docs with a detection)")


if __name__ == "__main__":
    main()

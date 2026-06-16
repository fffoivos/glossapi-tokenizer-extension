#!/usr/bin/env python3
"""Python↔Rust parity check for the line-LR span model. Dumps the present-line stream of N test docs
({present, abs_idx, n_total}), runs `reference_detect --mode score-lines`, and compares the Rust
per-line probabilities to the Python model's (the cached pline). Max |Δp| should be ~float-noise."""
import json, os, subprocess, sys
import numpy as np
import span_seq_data as D
import decode_spans as DS
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = f"{HERE}/../reference_detector/target/debug/reference_detect"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    data = D.load()
    pline = DS.get_pline(data)
    docs = [d for d in data.docs if data.docs[d]["split"] == "test"][:n]
    with open("/tmp/parity_in.jsonl", "w") as w:
        for doc in docs:
            d = data.docs[doc]
            w.write(json.dumps({"present": [t for _, t in d["lines"]],
                                "abs_idx": [a for a, _ in d["lines"]], "n_total": d["N"]}, ensure_ascii=False) + "\n")
    subprocess.run([BIN, "--mode", "score-lines", "--input", "/tmp/parity_in.jsonl",
                    "--out-spans", "/tmp/parity_spans.jsonl", "--out-counters", "/tmp/parity_out.jsonl"], check=True)
    rust = [json.loads(l)["p"] for l in open("/tmp/parity_out.jsonl") if l.strip()]
    maxdiff = 0.0; worst = None
    for doc, rp in zip(docs, rust):
        pp = pline[doc]
        assert len(pp) == len(rp), (doc, len(pp), len(rp))
        for i, (a, b) in enumerate(zip(pp, rp)):
            if abs(a - b) > maxdiff:
                maxdiff = abs(a - b); worst = (doc, i, float(a), float(b), data.docs[doc]["lines"][i][1][:80])
    print(f"compared {sum(len(p) for p in rust):,} lines across {len(docs)} docs")
    print(f"OVERALL max |Δp| = {maxdiff:.6f}")
    if worst:
        print(f"worst line: doc {worst[0][:10]} line {worst[1]}  py={worst[2]:.5f} rust={worst[3]:.5f}\n   text: {worst[4]}")
    print("PARITY OK (≤1e-3)" if maxdiff < 1e-3 else "PARITY FAIL — investigate the worst line above")

    # end-to-end: reconstruct each doc as a full text (blanks at the right absolute positions), run
    # `--mode spans`, compare the emitted bib_span line ranges to Python decode_spans.
    params = json.load(open(f"{HERE}/span_smooth_params.json"))
    with open("/tmp/parity_docs.jsonl", "w") as w:
        for doc in docs:
            d = data.docs[doc]
            buf = [""] * d["N"]
            for a, t in d["lines"]:
                buf[a] = t
            w.write(json.dumps({"id": doc, "text": "\n".join(buf)}, ensure_ascii=False) + "\n")
    subprocess.run([BIN, "--mode", "spans", "--input", "/tmp/parity_docs.jsonl", "--text-field", "text",
                    "--id-field", "id", "--out-spans", "/tmp/parity_docspans.jsonl", "--out-counters", "/tmp/pc2.jsonl"], check=True)
    rust_spans = {}
    for l in open("/tmp/parity_docspans.jsonl"):
        s = json.loads(l)
        rust_spans.setdefault(s["doc_id"], []).append((s["line_start"], s["line_end"]))
    mism = 0
    for doc in docs:
        d = data.docs[doc]
        py = [(s, e) for s, e in DS.decode_doc(d, pline[doc], params)]
        ru = sorted(rust_spans.get(doc, []))
        if sorted(py) != ru:
            mism += 1
            if mism <= 3:
                print(f"  span mismatch {doc[:10]}: py={sorted(py)} rust={ru}")
    print(f"span-level: {len(docs)-mism}/{len(docs)} docs IDENTICAL spans" +
          ("  ✓" if mism == 0 else f"  ({mism} differ)"))


if __name__ == "__main__":
    main()

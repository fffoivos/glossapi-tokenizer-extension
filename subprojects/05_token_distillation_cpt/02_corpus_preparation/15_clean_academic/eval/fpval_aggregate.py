#!/usr/bin/env python3
"""Aggregate the Opus FP-validation audit. Reweights the stratified sample back to the removed-line
POPULATION (using composition counts) to give independent population estimates of: genuine running-prose
removal rate (validates prose-protection), should-remove rate (effective precision when citation
reference-mass counts as a correct removal), and mislabeled-bibliography rate (annotation misses → the
model is more right than the silver gold says). Also reports heuristic-categorizer accuracy vs Opus.

Usage: fpval_aggregate.py <opus_result.json>"""
import os, sys, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    res = json.load(open(sys.argv[1]))
    cls = {c["unit_id"]: c for c in (res["result"]["classifications"] if "result" in res else res["classifications"])}
    man = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{HERE}/units/FPVAL_manifest.jsonl") if l.strip()}
    comp = json.load(open(f"{HERE}/results_composition.json"))["composition"]
    pop = {k.replace("bibliography_entry (correct)", "_correct_bib"): v for k, v in comp.items()}
    n_removed = json.load(open(f"{HERE}/results_composition.json"))["n_removed"]

    # per heuristic category: validated fractions
    by = collections.defaultdict(lambda: {"n": 0, "prose": 0, "remove": 0, "misbib": 0, "optype": collections.Counter()})
    for uid, m in man.items():
        c = cls.get(uid)
        if not c:
            continue
        b = by[m["heuristic_cat"]]
        b["n"] += 1
        b["prose"] += int(c["type"] == "running_prose" and not c["should_remove_for_bib_cleaning"])
        b["remove"] += int(c["should_remove_for_bib_cleaning"])
        b["misbib"] += int(c["is_mislabeled_bibliography"])
        b["optype"][c["type"]] += 1

    print(f"audited {sum(b['n'] for b in by.values())} lines across {len(by)} heuristic categories\n")
    print("per heuristic category — validated fractions (n audited):")
    print(f"  {'heuristic_cat':<26} {'pop':>6}  prose  should-remove  mislabeled-bib   (n)")
    pe_prose = pe_remove = 0.0
    fp_pop = fp_misbib = 0.0   # annotation-miss rate among NON-gold removed (the real 'FP' set)
    for cat, b in sorted(by.items(), key=lambda x: -pop.get(x[0], 0)):
        n = b["n"]; P = pop.get(cat, 0)
        fp, fr, fm = b["prose"] / n, b["remove"] / n, b["misbib"] / n
        print(f"  {cat:<26} {P:>6}  {fp:5.2f}  {fr:12.2f}  {fm:13.2f}   ({n})")
        pe_prose += P * fp; pe_remove += P * fr
        if cat != "_correct_bib":
            fp_pop += P; fp_misbib += P * fm
    print(f"\n=== POPULATION estimates over all {n_removed:,} removed lines (sample-reweighted) ===")
    print(f"  genuine running-prose removed : {pe_prose/n_removed*100:.2f}%   → PROSE-PROTECTION ≈ {1-pe_prose/n_removed:.4f}  (Opus-validated; deterministic est. was 0.998)")
    print(f"  should-remove (cleaner wants gone): {pe_remove/n_removed*100:.1f}%   → effective precision ≈ {pe_remove/n_removed:.3f}")
    print(f"  of the {fp_pop:,.0f} 'false positives' (non-gold removed), Opus says {fp_misbib/fp_pop*100:.0f}% are BIBLIOGRAPHY the windowed annotation MISSED")
    print(f"  → strict precision UNDER-states truth; true-bib precision ≈ {(n_removed-fp_pop+fp_misbib)/n_removed:.3f}")
    # categorizer accuracy: does heuristic prose match opus prose?
    json.dump({"pop_prose_pct": round(pe_prose / n_removed * 100, 3),
               "opus_prose_protection": round(1 - pe_prose / n_removed, 4),
               "effective_precision_should_remove": round(pe_remove / n_removed, 3),
               "fp_that_are_missed_bibliography_pct": round(fp_misbib / fp_pop * 100, 1),
               "true_bib_precision_est": round((n_removed - fp_pop + fp_misbib) / n_removed, 3)},
              open(f"{HERE}/results_fpval.json", "w"), indent=1)
    print("\nwrote results_fpval.json")


if __name__ == "__main__":
    main()

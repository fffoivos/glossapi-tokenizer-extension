#!/usr/bin/env python3
"""Score the goal-eval section sources: total-bibliography recall/precision of the STACKED pipeline
(section-classifier → β → gate), weighted to corpus. The headline gain over the β-only eval: false
negatives now include bibliographies the classifier mislabelled (non-β), which the β-gate never saw.
"""
import json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    man = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{HERE}/units/G_sections_manifest.jsonl")}
    raw = json.load(open(f"{HERE}/annotations_goal_sections/all.json"))
    ann = {a["unit_id"]: a for a in (raw["annotations"] if isinstance(raw, dict) else raw)}
    rows = []
    for uid, m in man.items():
        a = ann.get(uid)
        if not a:
            continue
        rows.append(dict(truth=bool(a["is_bibliographic_list"]), det=bool(m["detector_flagged_bib"]),
                         w=m["weight"], ps=m["predicted_section"], stratum=m["stratum"], kind=a.get("kind")))

    def wcm(rs):
        TP = sum(r["w"] for r in rs if r["det"] and r["truth"]); FP = sum(r["w"] for r in rs if r["det"] and not r["truth"])
        FN = sum(r["w"] for r in rs if not r["det"] and r["truth"]); TN = sum(r["w"] for r in rs if not r["det"] and not r["truth"])
        p = TP / (TP + FP) if TP + FP else 0; r = TP / (TP + FN) if TP + FN else 0
        return dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0)

    cm = wcm(rows)
    # where do the FN come from? (the classifier blind spot)
    fn = [r for r in rows if not r["det"] and r["truth"]]
    fn_by_ps = collections.Counter(r["ps"] for r in fn)
    fn_classifier_miss = sum(r["w"] for r in fn if r["ps"] != "β")   # bib the section classifier did NOT label β
    fn_gate_miss = sum(r["w"] for r in fn if r["ps"] == "β")          # bib labelled β but gate said kept
    fn_w = sum(r["w"] for r in fn)
    # raw counts of true bibs found in non-β sections (the headline blind-spot evidence)
    bib_in_nonbeta = [r for r in rows if r["truth"] and r["ps"] != "β"]

    L = [f"# Goal-eval — section sources (Kallipos/Pergamos): TOTAL bibliography recall\n",
         f"sampled {len(rows)} sections across all classes (β + high/low entry-density non-β).\n",
         "## Stacked-pipeline confusion (detector = section-classifier→β→gate; weighted to corpus)\n",
         "| | truth: bibliographic list | truth: not |", "|---|---:|---:|",
         f"| detector flags bib | {cm['TP']:.0f} | {cm['FP']:.0f} |",
         f"| detector keeps | {cm['FN']:.0f} | {cm['TN']:.0f} |", "",
         f"**precision = {cm['precision']:.3f}  recall = {cm['recall']:.3f}  F1 = {cm['f1']:.3f}**\n",
         "## Where the misses are (the blind spot the β-only eval could not see)\n",
         f"- FN total (weighted) = {fn_w:.0f}.  **Classifier miss** (true bib in a NON-β section the gate never saw) = "
         f"{fn_classifier_miss:.0f} ({100*fn_classifier_miss/max(1,fn_w):.0f}%).  Gate miss (β but kept) = {fn_gate_miss:.0f}.",
         f"- raw: {len(bib_in_nonbeta)} of the sampled true-bibliography sections were classified NON-β "
         f"(predicted_section ∈ {dict(collections.Counter(r['ps'] for r in bib_in_nonbeta))}) — invisible to the β-gate.",
         f"- FN by predicted_section: {dict(fn_by_ps)}"]
    open(f"{HERE}/CONFUSION_MATRIX_GOAL_SECTIONS.md", "w", encoding="utf-8").write("\n".join(L))
    json.dump(dict(cm=cm, fn_classifier_miss=fn_classifier_miss, fn_gate_miss=fn_gate_miss,
                   bib_in_nonbeta=len(bib_in_nonbeta)), open(f"{HERE}/results_goal_sections.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()

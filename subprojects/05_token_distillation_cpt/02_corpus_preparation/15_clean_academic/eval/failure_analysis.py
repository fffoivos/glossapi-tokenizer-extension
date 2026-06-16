#!/usr/bin/env python3
"""Failure analysis for the interpretable line classifier: WHICH non-bibliography lines look like a
bibliography (false positives = removed prose, the costly error) and WHICH bibliography lines get
missed (false negatives). Categorizes each error line by type (ToC / caption / footnote / table / web /
CV-or-frontmatter entry / inline-citation-prose / continuation / header / prose) and splits FPs into
boundary-adjacent (benign slop) vs interior (genuinely eating prose). Output drives whether auxiliary
line-type labels would help the model discriminate.

Run on the held-out TEST split using the line-only model + tuned smoother."""
import os, re, json, collections
import numpy as np
import span_seq_data as D
import span_signals as S
import decode_spans as DS
HERE = os.path.dirname(os.path.abspath(__file__))

DOTS = re.compile(r"\.{3,}|…|(?:[.·]\s){3,}")
PAGE_END = re.compile(r"\b\d{1,4}\s*$")
CAPTION = re.compile(r"^\s*(Εικόνα|Σχήμα|Σχ\.|Πίνακας|Πίν\.|Figure|Fig\.|Table|Διάγραμμα|Γράφημα|Φωτογραφ|Χάρτης|Εξίσωση|Equation|Plate)\s*\.?\s*\d", re.I)
FOOTNUM = re.compile(r"^\s*\[?\d{1,3}\]?[.)]\s+\S")
TABLEY = re.compile(r"\|.*\||\t")
SENT_END = re.compile(r"[.;··:]\s*$")
WORDS = re.compile(r"[Α-Ωα-ωΆ-ώA-Za-z]{2,}")


def categorize_fp(line, sig, pos):
    low = line.lower()
    if CAPTION.search(line):
        return "caption"
    if DOTS.search(line) and PAGE_END.search(line):
        return "toc"
    if sig["is_header"] or sig["is_bib_header"]:
        return "header"
    if TABLEY.search(line):
        return "table"
    if sig["url"]:
        return "web/url"
    if FOOTNUM.match(line):
        return "footnote_citation" if sig["is_entry"] else "footnote_prose"
    if sig["is_entry"]:
        return "entry_frontmatter" if pos < 0.30 else "entry_like(non-target)"
    if sig["year_bare"] and len(line) > 80 and SENT_END.search(line):
        return "inline_cite_prose"
    nwords = len(WORDS.findall(line))
    if len(line.strip()) < 12 or nwords <= 1:
        return "short/numeric"
    return "prose"


def categorize_fn(line, sig):
    if sig["is_bib_header"] or sig["is_header"]:
        return "header_line"
    if sig["url"]:
        return "web/archival_entry"
    if sig["is_entry"] and sig["latin_frac"] < 0.2:
        return "greek_entry"
    if sig["is_entry"]:
        return "entry(other)"
    nwords = len(WORDS.findall(line))
    if len(line.strip()) < 12 or nwords <= 1:
        return "short/fragment"
    # in a bib but not entry-shaped: usually a wrapped continuation (title/publisher overflow)
    return "continuation(no entry signal)"


def main():
    data = D.load()
    pline = DS.get_pline(data)
    params = json.load(open(f"{HERE}/span_smooth_params.json"))
    fp_cat = collections.Counter(); fp_zone = collections.Counter()
    fn_cat = collections.Counter()
    fp_ex = collections.defaultdict(list); fn_ex = collections.defaultdict(list)
    n_fp = n_fn = n_pred = n_gold = 0
    for doc_id, d in data.docs.items():
        if d["split"] != "test":
            continue
        gold = set()
        for s in d["spans"]:
            a, b = s.get("start_line"), s.get("end_line")
            if isinstance(a, int) and isinstance(b, int) and b >= a:
                gold.update(range(a, b + 1))
        spans = DS.decode_doc(d, pline[doc_id], params)
        pred = set()
        for s, e in spans:
            pred.update(range(s, e + 1))
        present = {a for a, _ in d["lines"]}
        gold &= present; pred &= present
        n_pred += len(pred); n_gold += len(gold)
        N = d["N"] or 1
        for abs_idx, text in d["lines"]:
            inp, ing = abs_idx in pred, abs_idx in gold
            if inp and not ing:               # FALSE POSITIVE (removed prose)
                n_fp += 1
                sig = S.line_signals(text)
                c = categorize_fp(text, sig, abs_idx / N)
                fp_cat[c] += 1
                near = any(abs(abs_idx - g) <= 5 for g in gold) if gold else False
                fp_zone["boundary-adjacent" if near else "interior(deep in prose)"] += 1
                if len(fp_ex[c]) < 4:
                    fp_ex[c].append(text.strip()[:96])
            elif ing and not inp:             # FALSE NEGATIVE (missed bib)
                n_fn += 1
                sig = S.line_signals(text)
                c = categorize_fn(text, sig)
                fn_cat[c] += 1
                if len(fn_ex[c]) < 4:
                    fn_ex[c].append(text.strip()[:96])

    print(f"TEST: removed(pred) {n_pred:,} lines, gold {n_gold:,} | FP {n_fp:,} ({100*n_fp/max(1,n_pred):.1f}% of removed) | FN {n_fn:,} ({100*n_fn/max(1,n_gold):.1f}% of gold)\n")
    print("=== FALSE POSITIVES (prose the model removed) — by zone ===")
    for z, n in fp_zone.most_common():
        print(f"   {z:<26} {n:,} ({100*n//max(1,n_fp)}%)")
    print("\n=== FALSE POSITIVES — by look-alike type (what fooled it) ===")
    for c, n in fp_cat.most_common():
        print(f"   {c:<24} {n:,} ({100*n//max(1,n_fp)}%)")
        for ex in fp_ex[c][:3]:
            print(f"        · {ex}")
    print("\n=== FALSE NEGATIVES (bibliography the model missed) — by type ===")
    for c, n in fn_cat.most_common():
        print(f"   {c:<28} {n:,} ({100*n//max(1,n_fn)}%)")
        for ex in fn_ex[c][:3]:
            print(f"        · {ex}")
    json.dump({"n_fp": n_fp, "n_fn": n_fn, "fp_by_type": dict(fp_cat), "fp_by_zone": dict(fp_zone),
               "fn_by_type": dict(fn_cat)}, open(f"{HERE}/results_failure_analysis.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

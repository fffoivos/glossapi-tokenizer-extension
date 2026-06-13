# Eval B — β-gate confusion matrix + tuning actions

Opus 4.8 annotated 64 β sections **blind to the gate**, stratified to oversample the decision
boundary. Inter-annotator agreement on the double-annotated subset: **Cohen's κ = 1.00** (9 units) —
the binary truth is reliable, so the matrix reflects detector error, not label noise.

## Two views of the matrix

**(a) On the balanced eval sample (16 per stratum):** precision 0.781 · recall 0.862 · F1 0.820.

**(b) Prevalence-reweighted to the full β population (94,282 sections):**

| | truth: bibliography | truth: NOT bib |
|---|---:|---:|
| gate: bib (flag) | TP ≈ 43,000 | FP ≈ 18,400 |
| gate: kept-non-bib | FN ≈ 2,400 | TN ≈ 30,500 |

**precision ≈ 0.70 · recall ≈ 0.95 · F1 ≈ 0.81.** *(Small-sample: 16/stratum → wide CIs, treat as ±0.1.)*

**Headline correction:** the investigation's *estimate* of β-precision (0.85–0.90) was **optimistic**.
Measured precision is **~0.70** — the gate **over-flags**: ~30 % of sections it calls bibliography are
not droppable reference lists. Recall is high (~0.95): few true bibliographies are missed. So the gate's
error budget is dominated by **false positives (over-removal), not misses** — the opposite of what the
fail-closed design assumed, and directly relevant to your drop-policy.

## What the errors are (the tuning to-do, from measured features)

**False negatives (gate kept, truly bibliography) — all 4 identical in profile:**
`chicago_turabian` style, **Greek script** (latin ≤ 0.08), no dash bullets, gate reason `unconfirmed`.
B0048/B0063 had year-density 1.75/1.37 yet were kept. The gate's bib rule
(`year>0 AND (latin>0.15 OR dash>0.3)`) structurally **misses Greek-script Chicago bibliographies**
(Greek author names → low Latin; hanging indent → no dash).
→ **Fix R1:** high year-density alone (e.g. ≥ ~1.0 with ≥3 distinct years) is a bibliography signal,
independent of Latin/dash. Low risk; catches the entire FN class.

**False positives (gate bib, truly not) — two modes:**
1. **Header-stem over-fire** — `πηγ` matched "Πηγές Ενέργειας" (energy *sources*, a topic) and an
   abbreviations table matched the stem. → **Fix P1:** require the `πηγ`/`αναφορ` stems to be
   corroborated by citation-shape (year present), not header alone; bare `πηγ` is too broad.
2. **CV publication lists** (ΔΗΜΟΣΙΕΥΣΕΙΣ, ΚΑΤΑΛΟΓΟΣ ΑΝΑΚΟΙΝΩΣΕΩΝ — the author's own works) flagged as
   bib. The CV deny exists but is position-gated (pos<0.4); these sat at pos 0.2–0.98.
   → **Fix P2:** extend the CV-deny family (ΔΗΜΟΣΙΕΥΣΕΙΣ/ΑΝΑΚΟΙΝΩΣΕΙΣ/ΚΑΤΑΛΟΓΟΣ…) and apply it
   **regardless of position** (an own-publications list is keep-worthy wherever it sits).
3. Residual: an activities list with dashes+year, a Latin charter, and a single-citation footnote
   tripped `citation_shape` — partly addressed by requiring ≥N entries, not fully fixable by rule.

## Net

R1 lifts recall further and, more importantly, P1+P2 attack the dominant FP mass (~18 k sections).
All three are deterministic, derived from labelled errors, and re-measurable by re-running this eval
after the change. The labelled set (`annotations/`, with style/language/script/subject/syntax per
section) is reusable as a regression set.

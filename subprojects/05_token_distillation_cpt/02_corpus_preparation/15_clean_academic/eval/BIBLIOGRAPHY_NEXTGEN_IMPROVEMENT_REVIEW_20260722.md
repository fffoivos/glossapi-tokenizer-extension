# Improvement review for `position_hist_component_scope` (2026-07-22)

## Status of this document

Diagnostic review of the frozen sealed-test result (line P 0.9680 / R 0.9171).
It performs **no selection, no calibration and no training**. All numbers below
are recomputed from the already-published frozen prediction arrays and the
sealed consensus-silver labels, using the immutable bundle
`bib_nextgen_099c6b1_fullruntime`.

The rules sized in §4 were **read off the sealed test set itself**. Their
projected metrics are an optimistic upper bound, not a validated result, and
must be re-derived on the 1,118-document development corpus before any
candidate is re-frozen. See §6.

## 1. Error decomposition (this is missing from the experiment log)

The log reports aggregate P/R but not where the errors live. They are not
spread evenly.

### False negatives — 1,704 lines

| Cause | Lines | Share |
|---|---:|---:|
| Markdown headings deleted by `emit_markdown_headings=False` | 360 | 21.1% |
| Whole true components vetoed by the scope layer (53 components) | 501 | 29.4% |
| Image markers inside gold blocks | 35 | 2.1% |
| Ordinary lines never proposed by line model + decoder | 808 | 47.4% |

**100% of the 360 gold Markdown headings are missed**, because the frozen
decoder config sets `emit_markdown_headings=False`, which zeroes every heading
line after decoding. That setting buys the "zero non-BIB heading crossings"
gate by discarding all true headings as well. It also *splits* components:
BIB subheaders that the decoder deliberately bridged across are then punched
out, fragmenting one bibliography into several components and degrading the
aggregate evidence the scope classifier sees.

### False positives — 623 lines / 74,636 chars

| Cause | Lines | Share |
|---|---:|---:|
| Fully spurious components that survived the veto (35 components) | 465 | 74.6% |
| Contamination inside otherwise-correct components (overrun / bridge) | 158 | 25.4% |

Of the FP lines, 60 are `<!-- image -->` markers and 563 are ordinary lines.
Only 44 FP lines (7%) are longer than 330 characters, but they carry
**37,758 chars — 50.6% of all false-positive character mass**.

Kept true components are clean: median gold fraction 1.00, p05 0.89, and only
one kept component is under 50% gold. The whole-component keep/veto decision is
structurally sound; the failure is *which* components, not *how much* of them.

### Per source

| Source | FP lines | FP chars | chars/FP line | scope layer removed |
|---|---:|---:|---:|---|
| greek_phd | 360 | 35,834 | 100 | 1,452 → 360 (−75%) |
| kallipos | 115 | 33,646 | **293** | 124 → 115 (−7%) |
| openarchives | 148 | 5,156 | 35 | 362 → 148 (−59%) |

Kallipos is the worst source (char P 0.9292) for a specific reason: its false
positives are **long prose lines swallowed into otherwise-correct components** —
25 lines over 330 chars carry 22,583 chars, 67% of Kallipos FP char mass. A
component-level accept/reject cannot touch this failure mode at all, which is
exactly what the numbers show (the scope layer removed 7% of Kallipos FP lines
versus 75% for greek_phd).

## 2. Why development precision (0.9974) did not survive to test (0.9680)

Two compounding causes, both fixable.

**Winner's curse at the gate boundary.** The reported 0.9974 is the maximum
over roughly 192 decoder configurations × 13 scope thresholds, selected on the
same OOF predictions that produced the estimate. When the selection rule is
"maximise recall subject to precision ≥ 0.98", the winning configuration sits
exactly on the constraint boundary and the constraint is satisfied partly by
noise. A 2.9-point optimism at that operating point is the expected magnitude.

**The binding gate carried no test signal.** The threshold 0.90 was chosen
because it is the *first* value at which
`spurious_blocks_per_zero_block_document` falls from 0.0376 to exactly 0.0000:

```
thr 0.85  lineP 0.9951  lineR 0.8058  spur/0doc 0.0376   <- fails the <=0.02 gate
thr 0.90  lineP 0.9974  lineR 0.7891  spur/0doc 0.0000   <- selected
```

On the sealed test, `spurious_blocks_per_zero_bib_document` is **0.0 for every
candidate, including the unscoped ablation**. The discrete, knife-edge metric
that drove the threshold choice turned out to be uninformative, while the
metric that actually failed (line precision) was read off an optimistically
biased estimate.

**Distribution mismatch in the scope training set.** Development produced 2,425
components of which 152 (6.3%) were spurious. The sealed test produced 479
components of which 129 (**26.9%**) were spurious — a 4× difference in the
negative-class prior. The scope classifier was trained on 152 negatives to
solve a problem that is four times denser at deployment. Its test ranking
quality reflects that: AUC 0.874 overall, and 0.771 on Kallipos.

## 3. What the surviving false positives actually are

The 35 kept spurious components, characterised:

| Kind | Example |
|---|---|
| Figure/caption lists interleaved with image markers | `7. Άγιος Τίτος, αξονομετρικό σχέδιο (Di Vita 2010)` + `<!-- image -->` ×N (93 lines, cp 0.986) |
| CV / résumé blocks | `CLINICAL / PROCEDURAL EXPERIENCE`, `Ιαν 2002 - Οκτ 2002`, institution names |
| Name/index registers | `Eliade M., 137, 190` triplicated across table columns |
| Dublin Core metadata dumps (Kallipos) | `\| dc.classificationURI \| ... \|` |
| Footnote runs with rule separators | `\_\_\_\_\_` then `54 Βλ . π . χ . ὁμιλία 52 (DW II, σ. 492)` |
| OCR garbage | 20 consecutive lines of `.` |
| Enumerated teaching/appendix lists | `- -ESMSC safety Triagle` |

Group statistics (component means):

| Group | n | has heading | BIB-lexicon heading | image frac | rule frac | year frac | alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kept true | 297 | 0.72 | **0.43** | 0.004 | 0.000 | **0.816** | 0.661 |
| Kept spurious | 35 | 0.49 | **0.03** | 0.041 | 0.017 | **0.240** | 0.567 |
| Vetoed true | 53 | 0.81 | 0.19 | 0.000 | 0.000 | 0.857 | 0.714 |
| Vetoed spurious | 94 | 0.43 | 0.01 | 0.001 | 0.004 | 0.297 | 0.575 |

The single most discriminative available signal is **the text of the Markdown
heading immediately above the component**. A lexicon match on
`βιβλιογραφ|αναφορ|πηγές|παραπομπ|δικτυογραφ|references|bibliograph|works cited`
fires on 139 of 350 true components and on **2 of 129 spurious ones** — rule
precision ≈ 98.6%. The scope model currently consumes only a *probability* from
a header model (`neighbor:before:bib_header:max10`), never the heading text
class, and it is not exploiting it: 34 of the 35 kept-spurious components have
no BIB heading above, while 10 of the 53 vetoed-true components do.

`year_frac` is similarly separated (0.816 vs 0.240) and the scope model *does*
have `signal:presence:year_count:*` — yet it kept 20 spurious components with
year fraction below 0.25. With 152 negatives and `LogisticRegression(C=0.1)`,
the model has neither the data nor the capacity to use what it already has.

## 4. Ranked improvements, with measured size

Sizes are measured on the sealed test (§Status caveat applies).

### R1 — Emit BIB Markdown headings under a gate instead of deleting all of them
`emit_markdown_headings=False` costs 360 gold lines. Replace it with: emit a
heading only when a kept component starts within *k* lines below it, no kept
line above it, and `probability:bib_header ≥ 0.5`.

| window | emitted | gold | non-BIB (gate breaker) |
|---:|---:|---:|---:|
| 1 | 88 | 87 | **0** |
| 2 | 100 | 99 | **0** |
| 3 | 104 | 100 | 3 |

At window 2 this recovers **+99 TP lines with zero non-BIB heading crossings** —
the hard gate survives. It should also be re-run with subheader emission
restored inside components, which un-fragments the components the scope model
scores. Cost: one decoder flag turned from a blanket mask into a gated rule.

### R2 — Heading-anchored rescue of vetoed components
Never let the scope layer veto a component whose governing Markdown heading
matches the BIB lexicon, unless the component score is very low.

| rescue floor | components | gold lines recovered | FP lines added |
|---:|---:|---:|---:|
| cp ≥ 0.5 | 10 | **+170** | **+1** |
| cp ≥ 0.7 | 9 | +153 | +1 |

170 gold lines for 1 false positive. This is the cheapest recall win available
and it directly addresses finding #1 of the worst-docs review ("whole valid
components are still vetoed" — e.g. the vetoed components headed
`ΕΛΛΗΝΟΓΛΩΣΣΗ ΒΙΒΛΙΟΓΡΑΦΙΑ`, `3.9 Βιβλιογραφία`, `Bibliography`,
`ΒΙΒΛΙΟΓΡΑΦΙΚΕΣ ΑΝΑΦΟΡΕΣ`).

### R3 — Structural component vetoes the model cannot currently see
`structure:image_marker` is **absent** from the scope model's `SIGNALS` tuple
(only `markdown_heading` and `table_row` are there). Add it, plus a
footnote-rule detector (`^[\s\\_—–\-\.]{4,}$`).

| rule | spurious killed | FP lines removed | true killed | gold lost |
|---|---:|---:|---:|---:|
| image fraction > 0.15 | 4 / 35 | 115 | **0 / 297** | **0** |
| rule-line fraction > 0.05 | 2 / 35 | 27 | **0 / 297** | **0** |

142 FP lines removed at zero recall cost.

### R4 — Asymmetric bar for components with no governing heading
`no heading above AND year fraction < 0.4` kills 13/35 spurious components
(197 FP lines) at a cost of 3/297 true components (73 gold lines) — net −124
error lines. Prefer implementing this as an *interaction feature* in the scope
model rather than a hard rule, so it is fitted rather than asserted.

### R5 — Conditioned length guard on bridged and expanded lines
`normal_seed_length_limit=330` currently governs **anchors only**
(`eligible_length` in `raw_anchor`). Gap bridging and `adjacent_expansion`
apply no length test, so long body paragraphs are absorbed into true
components. This is the Kallipos char-precision failure.

**Do not apply a naive cap.** 548 true BIB lines exceed 330 chars and carry
234,656 chars (7.6% of char TP) — a blunt cap would cost more recall than it
buys. The guard must be conditioned: block bridge/expansion onto a line only
when it is both long *and* carries no bibliographic evidence (no year, no page
range, no DOI/URL, no numbered-entry marker). Target: 37,758 FP chars, 50.6% of
all FP character mass.

### R6 — Rebuild the scope training set from a high-recall decoder sweep
The root cause of AUC 0.874 is 152 negatives. Decode the development corpus at
several deliberately loose configurations (anchor 0.60/0.75, `anchors_required`
2, larger bridge gaps) and pool all resulting components into the scope
training set. This multiplies the negative class by 5–10× and matches the
26.9% spurious prior seen at deployment instead of the 6.3% prior seen in
training. With a real negative class, `HistGradientBoosting` becomes viable
where `LogisticRegression(C=0.1)` currently is not.

### R7 — Fix the scope training target
`targets.append(bool(gold[start:end+1].any()))` — *any* overlap makes a
component positive. A component that is 5% bibliography and 95% body text
trains as a positive. Replace with a graded target: positive at gold fraction
≥ 0.5, negative at 0, and either drop or down-weight the middle band. Better
still, regress the gold fraction and threshold on expected purity, which also
gives the trimming signal R5 needs.

### R8 — Additional component features suggested by the confusion classes
- governing heading **text class**, bucketed keep/stop (`Παράρτημα`,
  `Ευρετήριο`, `Περιεχόμενα`, `Δημοσιεύσεις`, `Βιογραφικό`,
  `Κατάλογος πινάκων/σχημάτων/εικόνων`, `Appendix`, `Index`,
  `List of figures/tables`) — addresses worst-docs finding #3;
- alphabetic-character fraction (kills the OCR-garbage components);
- near-duplicate-line fraction (kills triplicated index-register table rows);
- date-range fraction (`2002 - 2005`) to separate CV/employment lists from
  citations;
- document-level structural prior: rank of the component's score within its
  document, distance to the best-scoring component, and whether the document
  already contains a heading-supported component. A document normally has one
  primary bibliography region; a second citation-shaped block far from it and
  without a heading is much more likely to be footnotes.

### R9 — Treat image markers as don't-care in scoring
126 gold lines and 60 FP lines are `<!-- image -->` placeholders carrying no
text. They inflate both error counts without representing missed or destroyed
citation content. Excluding them from line-level precision/recall accounting is
an evaluation-policy fix, not a model fix, and should be decided explicitly
(worst-docs finding #5).

## 5. Composite projection

Applying R1 (window 2), R2 (cp ≥ 0.6), R3, and optionally R4 to the frozen
prediction arrays:

| Variant | Line P | Line R | Char P | Char R | non-BIB MD headings | body char loss |
|---|---:|---:|---:|---:|---:|---:|
| Frozen `position_hist_component_scope` | 0.9680 | 0.9171 | 0.9763 | 0.9350 | 0 | 0.00220 |
| + R1 + R2 + R3 (conservative) | 0.9754 | **0.9304** | 0.9815 | **0.9456** | **0** | 0.00172 |
| + R4 (full) | **0.9833** | 0.9268 | **0.9853** | 0.9430 | **0** | 0.00136 |
| `incumbent_entry` (production) | 0.9998 | 0.6409 | 0.9998 | 0.6916 | 0 | 0.00002 |

Both variants improve precision **and** recall simultaneously over the frozen
candidate, and the full variant clears the 0.98 line and character precision
gate. R5 is not included (it needs the conditioned implementation) and would
add roughly 1.5 points of character precision on top.

**This projection is fitted on the sealed test and is an upper bound.** It is
evidence that the gate is reachable, not evidence that it has been reached.

## 6. Protocol recommendations

1. **Nested selection.** Split development into inner folds for decoder and
   threshold selection and outer folds for the estimate, and report the outer
   estimate against the 0.98 gate. Without this the reported operating point is
   biased upward by roughly the 3 points observed here, and the gate is being
   applied to a number that cannot support it.
2. **Retire `spurious_blocks_per_zero_block_document` as a hard gate.** It is
   discrete, knife-edge on development, and was 0.0 for every candidate on
   test. It cost recall (threshold 0.85 → 0.90) and bought nothing measurable.
3. **Reconsider what the gate measures.** The downstream cost is body text
   destroyed in a pre-training corpus. The frozen candidate destroys **0.22% of
   all non-bibliography characters** while the incumbent leaves **31% of
   bibliography characters in the corpus**. If the deployment criterion were
   stated as a body-damage rate rather than a precision floor, the ranking
   between the two candidates would likely invert. This is a decision for the
   corpus owner, and it should be made explicitly rather than inherited from
   the entry-decoder era.
4. **The 143-document cohort is now burned.** It has been opened for the
   one-shot report, the worst-docs review and this analysis. Any candidate
   built on R1–R9 must be validated on development OOF, re-frozen, and
   evaluated on a *new* sealed cohort — or the 143 documents must be explicitly
   redesignated as a development set with a fresh test cohort sealed behind it.
5. **Reconcile the label policy first** (worst-docs finding #5). Whether CV
   publication lists and scattered body footnotes should be removed by the
   production cleaner changes the sign of several of the errors counted above.
   R4 in particular is partly fighting the consensus labels rather than the
   model.

## 7. Reproduction

Diagnostics were run on a Clariden login node against the immutable bundle:

```
CR=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research
uenv run pytorch/v2.9.1:v2 --view=default -- \
  env PYTHONPATH="$CR/code_bundles/bib_nextgen_099c6b1_fullruntime:$CR/python_deps/sklearn-1.9.0-py312" \
  python3 <diagnostic script>
```

Inputs: `unseen_features_099c6b1_r1`, `unseen_predictions_099c6b1_r1`,
`scope_gen3_linear_balanced_4adbef2_r1/models/fold*.pkl`, and the sealed cohort
at `sealed_tests/bibliography_150_20260718/48_consensus_silver/run-4256753`.
Component probabilities were regenerated with the frozen fold bundles via
`build_component_table` + `predict_component_probability`, reproducing the
published component counts exactly (479 total, 332 kept at threshold 0.90).

# Cohort-2 one-shot, triage and same-cohort bake-off (2026-07-23)

## Decision

**`heading_lexgate` at scope threshold 0.98 is the recommended bibliography
cleaner and supersedes `incumbent_entry`.**

On the fresh 150-document cohort it removes **86.0% of bibliography characters
while destroying 0.258% of body characters**. The deployed incumbent removes
**53.9% while destroying 0.505%**. That is strict domination on both axes —
there is no trade-off to weigh between them.

This recommendation **does not clear the historical 0.98 line-precision gate**
(the candidate scores 0.9592 line precision). The argument for it is that the
gate is denominated in the wrong unit; see §4. Accepting this recommendation
means accepting that change of measure, which is an owner decision.

## 1. Evaluation cohort

`sealed_tests/bibliography_150_20260723_v2/30_consensus_silver`

- 150 documents, 210,704 lines, balanced 50/50/50 across greek_phd, kallipos
  and openarchives.
- Dual-`terra-high` consensus silver, binary A/B agreement 0.99733, 563
  UNKNOWN lines, **0 documents excluded**.
- Independently verified disjoint: **0 overlap** with the 143-document
  20260718 cohort and **0 overlap** with the 1,392-document development source.
- Bibliography is 8.93% of trusted text (3,837,035 of 42,978,881 characters).

This is a materially better cohort than 20260718, whose original 150-document
freeze failed its agreement gate and dropped 7 documents.

## 2. One-shot result (sealed)

The candidate was frozen before any label access
(`frozen_candidates_cohort2_v7.json`, status `frozen_before_test_open`),
features and predictions were computed label-blind, and the evaluation was
one-shot. This is the only genuinely sealed number in this document.

| Metric | Value |
|---|---:|
| Line precision / recall | 0.95921 / 0.83177 |
| Character precision / recall | 0.97033 / 0.85954 |
| TP / FP / FN lines | 19,708 / 838 / 3,986 |
| Body characters destroyed | 0.258% |

Against the same configuration on the 20260718 cohort (line P 0.98181 /
R 0.91347) this is a real generalisation gap: 2.3 points of precision and 8.2
points of recall. The earlier cohort was optimistic.

## 3. Error triage: annotation or model?

The question was whether the residual error is annotation quality. It is not.

**Annotation quality improved.** Lexicon-matched Markdown headings: 137, gold
137, **non-gold 0**. The label error class that blocked the entire heading
family on the previous cohort is absent here — the new pipeline's
Markdown-header repair fixed it. The lexicon-crossing exemption added in
commit `151b9dc1` is therefore inert on this data (harmless, but earning
nothing). The development-derived extra lexicon entries fired 6 times, all 6
gold: they neither helped nor hurt.

**The dominant causes are extraction quality and genuine model failure:**

- *Fragmented extraction.* The largest false-negative runs are bibliographies
  shattered to one token per line (`' Shelf'`, `' of'`, `' the'`) or headings
  destroyed by OCR (`## ΛΙΟΓ ;`, `## Βιβλιογραφία ANA ΚΕΦΑΛΑΙΟ`). No heading or
  lexicon work can recover these.
- *Straight model misses.* A 398-line bibliography in ordinary
  `- [1] Author, 'Title'` format missed entirely, nothing predicted nearby.
- *Domain shift.* One law thesis (`6cf60ed63b`) with citation-dense footnotes
  contributes **236 of 838 false-positive lines — 28% of all false positives
  from a single document**.
- *A genuine label-policy case, not an error.* Bibliography abbreviation lists
  (`AC = Archeologia Classica`, `ΑΔ = Αρχαιολογικό Δελτίο`) counted as 38
  false-positive lines; these arguably belong to the bibliography.

### Extraction-quality stratification

Computed from document text only — no labels, no model output. Reported as a
full curve so no cut is chosen post hoc.

| fragmentation cut | docs kept | Line P | Line R | Char P | Char R |
|---:|---:|---:|---:|---:|---:|
| none | 150 | 0.95921 | 0.83177 | 0.97033 | 0.85954 |
| 0.30 | 147 | 0.97755 | 0.84980 | 0.97670 | 0.85908 |
| 0.25 | 145 | 0.97758 | 0.85000 | 0.97665 | 0.85829 |
| 0.20 | 143 | 0.97732 | 0.84929 | 0.97639 | 0.85746 |

**Three documents out of 150 cost 1.8 points of line precision.** The result is
flat across cuts 0.30–0.20, so the conclusion is not knife-edge. Excluding them
recovers most of the *precision* gap versus the previous cohort but only ~1.8
of the 8.2 points of *recall* gap — the recall regression is genuinely the
model.

On the excluded documents the model **under-removes** (FN 599 vs FP 398), which
is the safe failure direction: those documents keep their bibliographies rather
than losing body text.

**The cohort quality gate does not measure fragmentation.** `glossapi-rs-noise`
and the fragment fraction are close to uncorrelated here:

| document | fragment fraction | median tokens/line | noise score |
|---|---:|---:|---:|
| `758646081e` | 0.667 | 1 | 18.7 |
| `7717ed7905` | 0.366 | 4 | 0.0000 |
| `ab7dbf682f` | 0.167 | 12 | 57.5 |

That is why these passed a screen which reviewed 36 candidates and rejected 8.
Adding a fragmentation statistic to the selection gate is cheap and would make
future cohorts match the pipeline's intended scope.

## 4. Same-cohort bake-off

Every candidate run on the same 150 documents, same features, same metric.
`orig_component_scope` was executed under its own code bundle because it uses
the older 81-feature component schema; everything else shares one code path.
No selection was made against cohort-2 labels — all five models were fixed
beforehand.

| candidate | char P | char R | bib removed | body destroyed | bib chars per body char |
|---|---:|---:|---:|---:|---:|
| `incumbent_entry` (deployed) | 0.9127 | 0.5388 | 53.9% | 0.505% | 10.5 |
| `position_hist_unscoped` | 0.9193 | 0.9254 | 92.5% | 0.796% | 11.4 |
| `orig_component_scope` @0.90 | 0.9454 | 0.8840 | 88.4% | 0.500% | 17.3 |
| `devfix_corrected` @0.85 | 0.9412 | 0.9033 | 90.3% | 0.553% | 16.0 |
| **`heading_lexgate` @0.98** | **0.9703** | 0.8595 | 86.0% | **0.258%** | **32.7** |

### Why the historical gate misleads

The incumbent's **line** precision on this cohort is 0.9788 — respectable. Its
**character** precision is 0.9127. Its false positives are long lines. For
corpus cleaning the loss is characters, not lines, so line precision flatters a
model that wrongly deletes a few long paragraphs and penalises one that wrongly
deletes many short citation fragments. The gate was measuring the wrong
quantity in the wrong unit.

### The marginal trade

Starting from `heading_lexgate` @0.98 and buying more recall:

| move | extra bibliography removed | extra body destroyed | marginal ratio |
|---|---:|---:|---:|
| → `devfix_corrected` @0.85 | +168k chars | +115k chars | 1.46 |
| → `position_hist_unscoped` | +253k chars | +211k chars | 1.20 |

Beyond this operating point roughly one character of body text is destroyed per
extra character of bibliography removed. For pretraining data that is a bad
exchange. `position_hist_unscoped` has the best character F1 (0.9224), but F1
assumes both error types cost the same, which here they do not.
`heading_lexgate` @0.98 has the best character F0.5 (0.9460) of any candidate
at any threshold.

### Honest margin

`heading_lexgate` and `devfix_corrected` are close and trade places along the
curve — at threshold 0.98 lexgate leads on F0.5 (0.9460 vs 0.9432), at 0.99
devfix leads (0.9457 vs 0.9446). A meaningful part of lexgate's advantage *at
its own operating point* comes from where the body-damage criterion placed its
threshold. The margin over `incumbent_entry` and `orig_component_scope` is
large and consistent; the margin over `devfix_corrected` is not.

## 5. Recommended configuration

| Component | Path |
|---|---|
| Feature table | `experiments/bib_nextgen_devfix_20260722/full_table_v3` |
| Line model | `experiments/bib_nextgen_devfix_20260722/line_hist_v3` |
| Decoder | `experiments/bib_nextgen_devfix_20260722/decode_v6_lextopo` |
| Component scope | `experiments/bib_nextgen_devfix_20260722/scope_linear_v7` |
| Scope threshold | **0.98** |
| Scope guards | heading rescue 0.6, image-fraction veto 0.15, rule-fraction veto 0.05 |

Receipt digests (SHA-256):

```
table         052bf69fff41ffb0b6d58c47ab78440cb265ca7706083035efed40b650940d20
line model    75d28d2be076b700593e5eb25f2cec83d99534f696feda2fa5b117ebbf36fefd
decoder       7c5fb2411a4019167685409654d6355ccb6f54aaf98cfd825d06acb9dde4f78d
scope         0a597bc779d6c8536f8a929848ce3c616f19ace31cead0d1e9d77bb3dbfac6d0
freeze        9c58d922763ca7490f56cf74794f07d926011ed6fa076f94bf41f15cc51966a7
one-shot      76816d339b43c20415794494e9469e3ada0ed16ff6b78328d08cd150da474eb6
```

Immutable bundle: `code_bundles/bib_nextgen_cohort2_ec1b5977`.
Branch `codex/bib-nextgen-lexicon-gate`, head `ec1b5977`.

Jobs: table 2869797, line model 2870073, decoder 2871021, scope 2871301,
unseen features 2876167, prediction 2876279, one-shot evaluation 2876294.

## 6. What the candidate is, and what is load-bearing

`heading_lexgate` is cumulative: it contains all of `devfix_corrected` (scope
guards, ≥50% component-purity target, conditioned expansion) plus:

1. lexicon-aware crossing gate, which unblocked gated heading emission;
2. heading normalisation for section numbers and brackets;
3. development-derived extra subheadings in the lexicon;
4. `lexicon_heading_topology` — an exact lexicon match substitutes for the
   heading model's probability;
5. the body-damage gate, which placed the operating point at 0.98.

Measured as contributing little or nothing: the retrained line model is inert
for accuracy (0.89113 original vs 0.89090 retrained on development) and is kept
only because the v3 feature schema requires it; the extra lexicon entries fired
6 times on this cohort; `lexicon_heading_topology` was worth +0.14pp on
development and was never separately ablated on either cohort. The load-bearing
parts are the heading-emission unblock and the operating-point move.

## 7. Conditions before corpus-scale use

1. **Corpus-scale dry run, no labels required.** Removal rate per document, its
   distribution, share of documents with zero removal, share with implausibly
   high removal. Catches distribution shift against a corpus far larger than
   150 documents.
2. **Decide the fragmentation policy** — scope those documents out under a
   stated rule, or accept that the cleaner under-removes on them. Either is
   defensible; leaving it undecided is not.
3. **Eyeball a sample of removed spans**, including the abbreviation-list case
   in §3.
4. **Hold the threshold at 0.98–0.99.** Looser settings buy recall at roughly
   1:1 body damage.

## 8. Known limitations

- **Kallipos recall is 0.677** at this operating point — 170 gold blocks
  against 91 predicted components, i.e. missed per-chapter bibliographies. One
  of three sources is materially under-served.
- **Variance at n=150 is real.** Three documents moved pooled precision by 1.8
  points; a different draw could differ materially.
- **Three sources only.** The production corpus may be broader.
- **Cohort 2 is now a development cohort.** Its labels were opened for the
  triage and bake-off. No selection was made against them, but a future
  unbiased read needs a fresh seal.
- The body-damage level `0.00025` was calibrated on the 20260718 cohort. It is
  out-of-sample for cohort 2, but it is still a back-calibrated number rather
  than one derived from a stated body-text-loss tolerance.

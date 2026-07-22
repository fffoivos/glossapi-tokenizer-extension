# Bibliography signal refinement run — 2026-07-14

## Decision

The edge-aware decoder is a useful research candidate. The component gate is
not yet an adequate solution for citation-dense prose, and neither arm is
approved for destructive corpus cleaning.

The strongest train-OOF edge policy uses every deterministic competing role
on the left fringe and only structural roles on the right fringe. It improves
line precision from 0.913932 to 0.920290 while line recall changes from
0.954082 to 0.952889. Token precision improves from 0.910663 to 0.918244 while
token recall changes from 0.981018 to 0.980238. This stays inside the frozen
0.25-point line-recall and 0.10-point token-recall budgets.

The conservative component gate improves line precision to 0.919827 and keeps
line recall at 0.953162, but it does not solve the dominant unseen
literature-review error. Its more aggressive operating points improve
precision substantially only by discarding too much bibliography material.

## What was implemented

The frozen D1 line classifier, signal TCN, and recall-first block configuration
were not changed.

### Edge experiment

`bibliography_signal_refinement.py` reconstructs the same block twice:

1. the frozen two-line-expansion prediction; and
2. the independently anchored core with expansion set to zero.

Only the fringe outside that core may be removed. A deterministic role inside
the core is immutable. Left and right role sets are selected independently on
grouped train OOF, allowing a strict left policy without imposing the same
rule on the right.

The selected train-only policy removed 1,156 proposed lines: 994 silver
non-BIB and 162 silver BIB. It does not change the number of proposed blocks;
it only tightens their edges.

### Component experiment

Two proposal views were cross-fitted over document folds:

- complete frozen blocks; and
- blocks split at structural Markdown headings, assigning the heading to the
  following component.

The logistic and monotonic-HGB arms see only frozen signal/entry probability
summaries, deterministic role fractions/continuity, and exact-header state.
They never receive text, source, document identity, raw length, or document
position. Candidate purity at 80%/20% supplies positive/negative supervision;
mixed candidates are masked.

The selected conservative component is `heading_split` logistic L2 at a 0.10
base and 0.10 structural-heading threshold. Raising only the structural-
heading threshold to 0.20 would reject more narrative sections, but train-OOF
token recall falls to 0.978713, beyond the predeclared 0.980018 floor. It was
therefore rejected.

## Train-OOF result

| Arm | Line precision | Line recall | Token precision | Token recall | Line FP | Line FN |
|---|---:|---:|---:|---:|---:|---:|
| Frozen recall-first baseline | 0.913932 | 0.954082 | 0.910663 | 0.981018 | 12,208 | 6,239 |
| Selected asymmetric edge | 0.920290 | 0.952889 | 0.918244 | 0.980238 | 11,214 | 6,401 |
| Selected conservative component | 0.919827 | 0.953162 | 0.916667 | 0.980476 | 11,288 | 6,364 |

The edge and component numbers are independent ablations over the same frozen
baseline. The run did not select a combined train operating point.

## Development review result

Foivos's 99 corrections were not used as fit labels or threshold inputs. The
first experiment definition was review-blind. After its failure analysis, the
second definition added the missing asymmetric edge grid and conditional
structural-heading test. Consequently, the second result is a development-set
diagnostic and is not an independent held-out estimate.

| Frozen v2 development diagnostic | Marked FP removed | Boundary FP removed | Whole-block FP removed | Unmarked removals requiring review |
|---|---:|---:|---:|---:|
| Edge only | 26/99 | 20/41 | 6/58 | 47 |
| Component only | 21/99 | 11/41 | 10/58 | 71 |
| Intersection/combined | 37/99 | 25/41 | 12/58 | 105 |

The weird Greek-PhD document accounts for 75 of the 142 combined removals.
On the 29 ordinary documents, the combined arm removes 67 predictions: 34
marked wrong and 33 unmarked. These unmarked removals are risk, not established
false negatives, but they prevent sign-off without another review.

The main whole-block failures remain:

- `701d19f7754e...`: none of the 22 marked literature-review lines is removed;
  its heading-split component scores 0.263, above the conservative 0.10 gate;
- `dcf172164a6e...`: the combined arm removes 12 of 39 marked lines, leaving
  citation-dense legal/historical prose accepted; and
- the component arm succeeds on the six-line method-section split in
  `f6667f451521...`, showing that heading splitting works for some, but not all,
  structural tails.

## Generalization controls and limits

- Every fit score is document-grouped OOF over 1,113 extraction-qualified
  train documents.
- The line classifier and signal TCN are unchanged.
- Feature inputs exclude raw text, source, document identity, length, and
  position.
- Both refinements are reject-only and cannot create a new deletion line.
- Recall budgets were frozen before the run.
- The 99 corrections influenced the v2 experiment design, so a fresh unseen
  source-balanced review is mandatory before any production decision.

The component model's standardized logistic signs are stable enough to show
that this is not a simple fold accident, but several signs are contrary to the
initial monotonic story. In particular, dense per-line entry positives can be
negative after contextual probability and run continuity are included, and a
structural heading is not universally negative because genuine bibliography
subsections share that shape. This is another reason not to convert the model
into hand-written weights.

## Artifacts

Local code and tests:

```text
subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval/sequence_models/bibliography_signal_refinement.py
subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval/sequence_models/bibliography_signal_refinement_unseen.py
subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval/sequence_models/tests/test_bibliography_signal_refinement.py
```

Clariden train artifact, job `2759089`, commit
`fbff76013a85dcf9bfe6e0531ed914f14208aaf1`:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/signal_refinement_r4
```

Clariden development diagnostic, job `2759096`:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/signal_refinement_unseen_r2/diagnostic.json
```

## Next gate

1. Freeze the asymmetric edge candidate without further tuning.
2. Draw a new source-balanced canonical sample absent from both training and
   the 30-document development packet.
3. Review every line removed by the edge arm and a sample it retains at both
   boundaries; conduct an explicit false-negative pass.
4. Do not advance the current component gate. Build a separate component-level
   annotation packet for citation-dense narrative prose and true bibliography
   blocks from all three sources, then test a narrative-discussion auxiliary
   role or a small contextual model on document-grouped folds.
5. Recombine arms only after each passes independently on the fresh packet.

# Bibliography entry-to-block ladder run — 2026-07-13

## Scope and status

This run implements `BIB_LINE_TO_BLOCK_CLASSIFIER_PLAN.md`.  It is a
bibliography-only comparison against GPT-generated LLM-silver labels.  It is
not human gold and does not authorize corpus deletion.  The historical
608-document test split is physically absent.  The 274-document retrospective
validation split remains unopened until the train-only configuration is
frozen.

Current execution state:

- the entry label contract, grouped folds, L0–L4/D1 OOF ladder, and B0/H0 are
  complete;
- corrected B1 completed on Clariden as job `2754368` with all 20 independent
  fold/arm/ablation fits in one CPU wave;
- B2 job `2754369` recorded the predeclared skip decision, freeze job `2754370`
  completed, retrospective validation job `2754371` completed, and review-site
  job `2754381` completed;
- all outputs remain `production_eligible: false`.

## Immutable locations

Clariden experiment root:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z
```

Runtime code bundle:

```text
/capstor/scratch/cscs/fffoivos/classifier_research/code_bundles/bib_entry_e1463db_runtime
```

The bundle is read-only.  Its `inventory.sha256` hash is
`0044d89799187e71bc40ccb670d9f418fe5990e95c36e4c3aefbb111c9a8cd57`.
It contains the `sequence_models` package and the three legacy feature-module
siblings required at import time.  A CPU-node smoke test imported the B1,
validation, and review modules successfully before job submission.

The review-only bundle is:

```text
/capstor/scratch/cscs/fffoivos/classifier_research/code_bundles/bib_entry_a169a6b_runtime
inventory.sha256 838e7e953db5777ca00cd8fa084b78fd59ee9179922d6e2672b8a9f5a908e717
```

Pinned Python dependencies:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/python_deps/sklearn-1.9.0-py312
```

Pinned input:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/struct2k_sources/struct2k_joint_20260712b/struct2k.LLM_silver.jsonl
sha256 f18ef6bf3061d932ae0aaeb2349392a2e590f2778e3205cf7cbcb5c79dffa7c0
```

## Completed train-only evidence

The feature table contains 1,118 train documents / 1,100 canonical works and
939,014 emitted lines.  There are 138,447 entry positives, 799,771 negatives,
and 796 exact bibliography headings/subheadings masked from entry-model loss.
All splits are grouped by `work_id`.

| Arm | Inputs/model | OOF line PR-AUC | B0/H0 line precision | B0/H0 line recall | Token precision | Token recall | Spurious blocks / silver-zero doc |
|---|---|---:|---:|---:|---:|---:|---:|
| L0 | equal-weight binary presence | 0.765926 | 0.933653 | 0.798999 | 0.908634 | 0.848181 | 0.789474 |
| L1 | binary logistic | 0.874916 | 0.971149 | 0.772369 | 0.963564 | 0.819556 | 0.473684 |
| L2 | raw-count logistic | 0.862986 | 0.975553 | 0.718765 | 0.972143 | 0.773977 | 0.308271 |
| L3 | log-count logistic | 0.871641 | 0.973696 | 0.748339 | 0.968548 | 0.801236 | 0.390977 |
| L4 | presence + log-count elastic net | 0.888210 | 0.973967 | 0.775953 | 0.968462 | 0.823642 | 0.443609 |
| D1 | depth-limited boosted trees | 0.891568 | 0.979407 | 0.775328 | 0.976128 | 0.821358 | 0.375940 |

Learned combination is materially better than equal feature weighting.  D1
and L4 were retained for B1.  Neither B0/H0 arm meets the predeclared safety
gate of at least 0.99 line precision and at most 0.02 spurious blocks per
silver-zero document, so no train-only B0 result is production eligible.

H0 attached 396/678 eligible exact structural lines for D1 with 15 false
attachments (precision 0.9635, recall 0.5841).  H0 cannot initiate a block.

## B1 and train-only freeze

| Variant | Line precision | Line recall | Token precision | Token recall | Spurious blocks / silver-zero doc | IoU>=0.5 block precision | IoU>=0.5 block recall | Split + merge errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1, no header feature | 0.992141 | 0.772498 | 0.993010 | 0.814345 | 0.180451 | 0.758755 | 0.597426 | 125 |
| D1, header feature | 0.990593 | 0.797835 | 0.991682 | 0.838249 | 0.218045 | 0.762985 | 0.639093 | 121 |
| L4, no header feature | 0.990036 | 0.759938 | 0.990479 | 0.809803 | 0.285714 | 0.727794 | 0.622549 | 130 |
| L4, header feature | 0.990630 | 0.759248 | 0.990975 | 0.809600 | 0.278195 | 0.723359 | 0.628064 | 131 |

The explicit header feature improved D1 recall but reduced precision, so the
predeclared ablation rule retained D1/no-header.  H0 remains a post-block stage.
B1 reduced D1 split/merge errors from 399 to 125.  B2 was skipped because the
remaining error count was far below 90% of B0/H0's count, as required by the
predeclared escalation rule.

Freeze job `2754370` selected B1/D1/no-header, but recorded
`research_only_no_candidate_met_safety_gate`: line precision passed 0.99, while
0.180451 spurious blocks per silver-zero document remained above the 0.02 gate.
The frozen train-only configuration therefore cannot be promoted by a good
retrospective-validation score.

## Single retrospective validation

Validation job `2754371` opened the 274-document retrospective split only
after freeze.  It completed in 4m30s.  Its receipt SHA-256 is
`2030d194096060c78f99d5fccd8f6753b06d24ca12f1501264a41a54d7e10c3e`.

| Metric | Frozen B1/D1/no-header |
|---|---:|
| line precision | 0.997976 |
| line recall | 0.629953 |
| token precision | 0.998341 |
| token recall | 0.825239 |
| token F0.5 | 0.958145 |
| predicted / gold blocks | 305 / 365 |
| exact-block precision / recall | 0.432787 / 0.361644 |
| IoU>=0.5 block precision / recall | 0.790164 / 0.660274 |
| spurious blocks / silver-zero document | 0.000000 |
| documents with any false deletion | 0.043796 |
| false-positive lines | 58 |
| false-positive long lines | 4 |

Per-source line precision is 0.997685 (`greek_phd`), 0.997896 (`kallipos`),
and 0.998990 (`openarchives`).  Line recall is lower and source-dependent:
0.593794, 0.824450, and 0.692505 respectively.  The model is therefore
precision-first, and the next improvement target is recall/boundaries without
reintroducing zero-document false blocks.

Validation passed the frozen operating metrics but does not override the
train-only `research_only` decision.  The comparison is against LLM-silver,
not human gold, and this version remains forbidden from corpus deletion.

## Prediction-blind extraction-quality qualification

The unexpectedly low micro-averaged recall was concentrated in a small number
of extraction failures.  A separate audit was therefore run without reading
the bibliography predictions or labels.  It used the canonical GlossAPI Rust
noise scorer from GlossAPI commit `6f29a2825559c540ab342fc77ae4457cf3556f2a`
plus conservative text-only signals for extreme line fragmentation,
character-spaced OCR, and unresolved glyph placeholders.

Audit job `2755017` screened all 274 validation documents and produced nine
review candidates.  Prediction-blind decisions were locked in commit
`e3bbade`: six documents were excluded as unusable extractions and three were
retained after the flag was found to be mathematical notation or localized
encoding noise.  Job `2755022` applied those locked decisions and only then
recalculated the 268-document metrics.

A subsequent outcome-directed check ranked the 50 documents with the most
missed silver-BIB tokens.  For every document, the beginning, quartiles,
middle, end, and first/middle/last missed-BIB regions were inspected.  Text
usability—not classifier agreement—determined the decision.  Of those 50, 44
were usable; five confirmed earlier exclusions; and one additional document,
`eaf30b21c052…`, was excluded because its central and bibliography regions are
dominated by reversed or otherwise garbled encoded text.  Commit `360aab2`
records the exclusion and `d7dab98` records that its selection was
outcome-directed.

| Metric | All 274 | Blind-qualified 268 | Follow-up-qualified 267 |
|---|---:|---:|---:|
| line precision | 0.997976 | 0.997976 | 0.997976 |
| line recall | 0.629953 | 0.821625 | 0.841896 |
| token precision | 0.998341 | 0.998340 | 0.998340 |
| token recall | 0.825239 | 0.856019 | 0.866984 |
| false-positive lines | 58 | 58 | 58 |
| false-negative lines | 16,802 | 6,208 | 5,370 |

The unchanged false-positive count is important: qualification removed
unreadable documents, not classifier-positive lines.  The original six
exclusions were three documents shattered into mostly one-word lines, two
dominated by 17,349 and 19,473 unresolved `GLYPH` placeholders, and one
character-corrupted OCR document with canonical Rust badness `68.10 > 60`.
The seventh is the garbled document found during the worst-50 inspection.

The 267-document figures are a diagnostic sensitivity analysis, not an
unbiased held-out estimate, because the follow-up documents were selected by
model error.  The original frozen 274-document result and the independently
qualified 268-document result remain the appropriate leakage-safe reports.

Clariden artifact:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/quality_audit_r4
job 2755069; quality_audit sha256 7ca0d9da8d9a70b2cd65b984d50606d5bb23903727efaba6515369a30599eb86
```

The document-level decisions and review provenance are in
`bibliography_validation_quality_decisions_20260714.json`.  The receipt records
that the automatic quality screen itself was prediction-blind; the decisions
file separately marks the later top-50 follow-up as outcome-directed.  Duplicate
submission `2755070` failed closed with exit 93 because the completed immutable
`quality_audit_r4` output already existed; it produced no replacement artifact.

## Train-only block-recall research — 2026-07-14

The retrospective validation results above were not used to fit or choose the
next model.  All experiments in this section use grouped out-of-fold predictions
for 1,113 readable training documents; validation remains closed until one
complete configuration is frozen.

The proposal ceiling established that candidate generation is not the limiting
step.  With line length removed from B1 emissions, the candidate union can reach
99.80% of silver-BIB lines and 99.96% of silver-BIB tokens.  Removing both line
length and document position raises the ceiling to 99.99% of lines and 99.99%
of tokens.  The problem is selecting true bibliography regions without deleting
citation-like prose, tables, captions, footnotes, or auxiliary lists.

Manual inspection of high-scoring false train-OOF components found seven common
failure shapes: merged running prose after a bibliography mention, lists of
figures/tables, abbreviation or source lists, footnotes interleaved with prose,
figure captions, tables of article metadata, and related-material/archive lists.
The inspection packet is:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/component_diagnostics_r2/false_component_contexts.json
job 2756398
```

The deterministic line system was then reused only to assign competing document
roles—not to duplicate bibliography features.  Figure captions and footnotes
were strongly negative individually, but the combined role detector also marked
30.4% of silver-BIB lines, so it is unsafe as a hard veto.  The component gate
therefore receives only the fraction of lines with an explicit competing role.
Exact negative-section headings were also tested, but their coefficient reversed
direction in one work-level fold and the precision/recall frontier did not
improve; they remain diagnostic metadata rather than a selected feature.

The final stable component schema (`v5`) has five plain-language inputs:

1. extent that rises to 32 lines and then saturates;
2. median frozen entry-line probability;
3. longest uninterrupted weak-line run as a fraction of the component;
4. exact bibliography header at or immediately before the start; and
5. fraction of lines assigned an explicit competing deterministic role.

Every expected direction held in all five grouped OOF folds.  Clariden job
`2757450` wrote the immutable `component_gate_r7` artifact.  Its selected safety
point is the `no_length` logistic gate at threshold `0.995`:

| Metric | Train OOF |
|---|---:|
| line precision | 0.992836 |
| line recall | 0.583454 |
| token precision | 0.996294 |
| token recall | 0.626076 |
| spurious blocks / zero-BIB document | 0.007519 |

The same gate can reach 92.55% line recall / 96.52% token recall above 95% line
precision, demonstrating that the remaining trade-off is component calibration,
not line-feature reach.  The exact feature meanings, rejected additions, and
non-overlap rules are in `BIBLIOGRAPHY_BLOCK_FEATURE_REFERENCE.md`.

The next registered experiment permits a high-confidence component to establish
a block and lets lower-confidence proposal spans expand it only through an
overlap-connected chain.  A weak or disconnected proposal can never create a
deletion.  This directly tests the intended rule that long/weak lines may be
kept inside a bibliography block but must be rejected in isolation.

Clariden job `2757451` completed that experiment as `component_expansion_r1`.
It rejected the overlap-chain design.  For the stable no-length logistic gate,
even an equal `0.995` core/expansion threshold joined overlapping alternative
proposals: 2,329 lines were added, line recall moved only from 0.583454 to
0.587509, and line precision fell from 0.992836 to 0.971403.  Lower expansion
thresholds caused larger false tails.  Only two expanded candidates passed the
safety rule, and both had substantially lower recall than the unexpanded stable
gate.  This result is retained as negative evidence and is not eligible for
validation.

The next train-only candidate moves the already-materialized perpendicular
deterministic roles into B1 as one-hot sequence observations.  A figure caption,
footnote, table/equation, exact negative scope, generic heading, prose,
legal/procedural line, or other explicit negative role is learned in context;
none is a hard veto.  This lets the CRF use repeated competing structure to stop
a region while still retaining an occasional marked line inside a coherent
bibliography.  The positive citation summary remains the frozen entry
probability, so no author/date/page feature is duplicated.

Clariden job `2757460` completed `role_sequence_r1`.  No candidate passed the
full safety gate, chiefly because isolated proposals remained in silver-zero
documents.  Its best point above 95% line precision was the no-length role CRF
with no extra anchor/count filter: line precision 0.960729, line recall
0.931406, token precision 0.961731, and token recall 0.968379.  The spurious
zero-document rate was 0.308271, so this is a proposal model only.

Six role effects were negative in every fold: figure caption, table/equation,
exact negative-scope heading, generic heading, footnote, and running/enumerated
prose.  Legal/procedural evidence reversed weakly in one fold, and the catch-all
other role stayed effectively zero with sign changes.  The next experiment
therefore composes the improved role-aware proposals with the already-stable
five-feature component gate; it does not weaken the safety thresholds or open
validation.

Clariden job `2757761` completed that composition as
`role_component_gate_r1`.  The strict safe point improved only modestly, while
the 0.95 threshold exposed a much more useful diagnostic: all 23 spurious
blocks came from one silver-zero OpenArchives document containing repeated
source lists under exact related-material/archive headings.  This was a
section-scope failure, not a generic citation-feature failure.

Jobs `2757865` through `2758154` narrowed and audited an exact deterministic
scope veto.  The retained `auxiliary_scope_veto_r6` recognizes only the
pre-existing exact auxiliary headings, exact Greek/English `WHY`/`EXAMPLES`
forms, and the exact selected-variants archive prefix.  Generic headings and
fuzzy matches are forbidden.  It may only reject an already-proposed component.
The scope covers 12,292 train lines and overlaps zero silver-BIB lines.  Its
selected 0.90 point is:

| Metric | Train OOF |
|---|---:|
| line precision | 0.991728 |
| line recall | 0.804743 |
| token precision | 0.991565 |
| token recall | 0.854008 |
| spurious blocks / zero-BIB document | 0.015038 |

At 0.95 the safe role/component model had left 23 spurious blocks; after the
exact veto, only two allowed silver-zero patterns remain at the selected point:
an author's publications/presentations section and footnotes.  Neither was
converted into a new one-off lexical exception.

Job `2758164` tested `rich_component_gate_r1`: q10/q90 entry probability,
minimum boundary probability, and eight separate role fractions were added to
a monotonic shallow tree and a logistic comparison.  It was rejected.  The
safe monotonic candidate regressed to 0.992897 precision / 0.563795 recall.
The logistic comparison reached 0.950717 precision / 0.891920 recall, but q10
and minimum boundary probability repeatedly took the wrong sign; negative
heading roles also reversed in one fold.  Validation remained unopened.

Commit `bd1ccc3` adds the next genuinely different candidate: a 32-hidden-unit
residual TCN over a 31-line neighbourhood.  It sees only the frozen entry
probability, the eight mutually exclusive deterministic roles, and the exact
header flag.  Raw text, line length, document position, source identity, and
validation rows are absent.  Five document-held-out models produce OOF scores;
the audited exact-scope veto and unchanged safety gate apply downstream.
Clariden job `2758175` runs this experiment as `signal_tcn_r1`.

Job `2758175` completed in 2m47s.  The contextual scores are useful, but plain
line thresholding is the wrong decoder: threshold 0.40 reached 0.950512 line
precision / 0.931134 line recall and 0.949836 token precision / 0.960967 token
recall, while leaving 0.436090 spurious blocks per silver-zero document.  The
only strict-safe threshold was 0.999, which emitted no lines.  That empty point
is not treated as a model win.

Commit `244aac9` ports these frozen OOF scores to the intended block level.
Two or three high-score anchors in a bounded window must first establish a
region.  Weaker or long lines can then be included only between anchors or
directly beside the region; an isolated high-score line cannot start deletion.
Length is not an input.  The exact-scope veto and H0 remain downstream.  Job
`2758177` evaluates the predeclared anchored grid as `signal_blocks_r1`, with
validation still closed.

Job `2758177` completed the 384-configuration anchored grid in 1m37s.  The best
strict-safe nonempty configuration reached 0.997122 line precision but only
0.193771 line recall.  The useful high-recall point reached 0.981289 precision /
0.864608 recall and 0.984414 token precision / 0.910997 token recall, with
0.045113 raw-silver spurious blocks per zero-BIB document.  It does not replace
`auxiliary_scope_veto_r6`.

Job `2758181` then materialized complete contexts for the 50 apparent false
components with the most disagreeing tokens.  The Codex first pass found 12
clear silver omissions, 21 mostly-correct blocks with boundary overruns, 9
genuine non-bibliography blocks, 7 policy-sensitive publication/endnote/reading
lists, and one extraction/lineage issue.  Five of the six apparent blocks in
silver-zero documents are genuine numbered annotated bibliographies in
Kallipos document `802cdb75649e...`; the sixth is genuine prose.  Full evidence
and the case register are in `BIB_SIGNAL_FALSE50_REVIEW_20260714.md`.

Commit `c414769` tested split barriers from stable-negative headings, figure
captions, footnotes, and sustained very-low contextual probability.  Job
`2758188` completed 480 configurations in 1m44s.  No candidate met the strict
safety gate.  The useful frontier was effectively unchanged at 0.981560 line
precision / 0.864608 recall; the best point above 0.99 precision reached
0.990178 / 0.785776.  Barriers also split the mislabeled Kallipos bibliographies
into more raw-silver spurious blocks.  This arm is rejected.  Further model
tuning is paused at the label-completeness and policy boundary; validation is
still unopened.

## Recall-first block continuation — 2026-07-14

The later recall-first work did not relax the requirement that an isolated
citation-looking line cannot start deletion.  It reused the frozen signal TCN
and searched the predeclared train-OOF anchored grid for the highest line
recall at a minimum 0.90 raw-silver line precision.  Two or more anchors within
a bounded window must still establish a block.

The first validation attempt, job `2758212`, failed closed before computing
metrics.  The exact heading `ΣΥΝΤΟΜΟΓΡΑΦΙΕΣ` enclosed 971 silver-BIB lines in
one document.  Abbreviation lists can expand citation keys into complete
entries, so abbreviation headings remain a competing structural line role but
no longer establish a hard negative section scope.  This is a semantic rule,
not a document exception.  Corrected train job `2758219` and validation job
`2758220` then completed.

Failure review of that validation exposed a second generic decoder bug.  A
valid bibliography immediately before a `List of Figures`/`List of Tables`
region and a publication list immediately after it were both discarded when
boundary expansion touched the intervening negative scope.  Commit `8656787`
makes exact negative scope a wall: anchors, weak-line bridging, and header
attachment operate independently on each side; scope lines cannot be emitted;
and a neighbouring coherent block survives.  Sixteen focused tests passed in
the immutable Clariden runtime before the revised metric run.

Train-only job `2758230` selected the same recall-first configuration family:
anchor probability 0.30, two anchors within 16 lines, bridge gap 8, inside
probability 0.05, and two-line boundary expansion.  Against the previous
scope implementation, train-OOF line recall rose from 0.952948 to 0.954082 and
token recall from 0.979891 to 0.981018.  Line precision was 0.913932 and token
precision 0.910663.  No validation-derived threshold or model weight changed.

Validation job `2758231` evaluated the frozen candidate.  Three populations
must be kept distinct:

| Reporting population | Documents | Line precision | Line recall | Token precision | Token recall |
|---|---:|---:|---:|---:|---:|
| all LLM-silver validation rows, including unusable extractions | 274 | 0.941128 | 0.750975 | 0.941929 | 0.961685 |
| prediction-blind extraction-qualified | 268 | 0.944981 | 0.967273 | 0.951407 | 0.980736 |
| outcome-directed worst-50 diagnostic-qualified | 267 | 0.944981 | 0.991138 | 0.951407 | 0.993298 |

The 267-document number is the requested readability sensitivity analysis,
not a replacement for the raw 274-document result or an untouched estimate.
The worst-50 text audit inspected the beginning, quartiles, end, and three
missed-BIB regions of every selected document.  It retained 44/50 and excluded
six unusable extractions; a seventh prediction-blind exclusion lies outside
the worst 50.  The one extra outcome-directed exclusion is a garbled encoded
document in which the classifier emitted no lines, which is why precision is
identical in the 267- and 268-document rows.

On the 267 readable documents, line recall is independently high for every
source: 0.988376 for Greek PhD, 0.994786 for Kallipos, and 0.998216 for
OpenArchives.  Their token recalls are 0.990995, 0.995790, and 0.999373.

The largest remaining valid missed runs were inspected line by line.  Every
one lacks two signal scores at or above the frozen 0.30 anchor threshold within
16 lines; none is caused by the corrected scope barrier.  The dominant cases
are low-confidence but coherent legal-memorandum, archival-source,
ancient-author, and author-publication sub-blocks next to detected material.
Several much smaller misses are isolated two-line reading lists inside
textbook chapters, which the large-block policy intentionally does not target.
Lowering the anchor threshold after seeing these validation examples would be
validation tuning and was not attempted.

Immutable Clariden artifacts:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/signal_recall_blocks_r3
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/signal_validation_r3
```

### Whole-source robustness audit

Commit `0f7a34e` adds a selection-ineligible robustness path.  It preserves all
frozen line features and labels but assigns Greek PhD, Kallipos, and
OpenArchives to three complete source folds.  Job `2758234` materialized that
fold view.  Job `2758235` fitted three otherwise identical signal TCNs, each
without any document from its held-out source.  Job `2758239` applied the
already-selected recall-first decoder from job `2758230`; no threshold was
searched on the source-held-out predictions.

Overall source-held-out signal-TCN performance was 0.920972 line precision /
0.954707 line recall and 0.914799 token precision / 0.980904 token recall.
Line recall is therefore essentially unchanged from the ordinary grouped
train-OOF result of 0.954082.  Held-out-source results were:

| Entire TCN source held out | Documents | Line precision | Line recall | Token precision | Token recall |
|---|---:|---:|---:|---:|---:|
| Greek PhD | 370 | 0.928390 | 0.954084 | 0.922336 | 0.986576 |
| Kallipos | 369 | 0.919225 | 0.972396 | 0.910617 | 0.973817 |
| OpenArchives | 374 | 0.898386 | 0.947274 | 0.890540 | 0.964593 |

This is a robustness check for the contextual block model, not a full
unseen-source estimate.  Its frozen entry-probability input is document-OOF
but was not retrained with the complete source removed.  The result nevertheless
shows that the contextual recall gain does not depend on access to other
documents from the held-out source.

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/source_holdout_table_r1
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/signal_tcn_source_holdout_r1
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/signal_source_holdout_eval_r1
```

## High-risk joint-review site

Review job `2754381` selected 120 proposed blocks: exactly 40 each from
`greek_phd`, `kallipos`, and `openarchives`.  Multiple cases can belong to the
same document, so the packet contains 70 complete emitted documents / 65,178
lines rather than snippets.  It contains:

- all 35 ownership-resolved feature spans and entry probabilities;
- proposed BIB backgrounds, LLM-silver BIB underlines, and exact case borders;
- explicit long-line inclusion/rejection reasons;
- the four labelled arrow decisions, resume at first undecided, separate
  Foivos/Codex records, an agreement display, and combined JSON export.

The selected risk set contains 11 blocks with one or more silver-non-BIB lines,
43 boundary-disagreement blocks, and three false-positive long lines.  The
validation run proposed no blocks in silver-zero documents, so that desired
risk category was genuinely unavailable rather than omitted.

Clariden artifact:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/review_r4
packet sha256 42c05ccdb51c9e30fedf61eb85029720af3d490744924e507a034e41b0086b65
receipt sha256 241ff4e40cb833ded233de59fb2061ed666aea3bfe115bbe0f2be8819da6ac78
```

Local presentation and compact report archive:

```text
/Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-entry-review-20260713
```

It is currently served at `http://127.0.0.1:8771/`.  To restart it:

```bash
cd /Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-entry-review-20260713
python3 -m http.server 8771 --bind 127.0.0.1
```

The site and packet passed SHA-256 parity after transfer and HTTP 200/content
length checks.  In-app visual automation was unavailable in this session; its
own troubleshooting resource was also missing.  Python tests and JavaScript
syntax checks passed, but Foivos's first visual review remains the final UI
acceptance check.

## Implementation commits

- `e07ec42` — verify the conservative exact-header mask policy;
- `9ce273d` — materialize entry tables, grouped folds, L0–L4/D1, and B0/H0;
- `62303c5` — pin the Clariden scikit-learn runtime;
- `7aa9f1d` — add constrained B1 and conditional B2;
- `15fefc1` — freeze train-only selection before one-time validation and apply
  H0 after sequence decoding;
- `1a9980a` — add the source-balanced, full-context joint-review site;
- `0331e40` — parallelize independent B0 arms for future runs.
- `e1463db` — schedule all 20 independent B1 fits in one CPU wave.
- `a169a6b` — compare Foivos/Codex decisions and export both review records.
- `4081d1d` — add prediction-blind validation extraction-quality auditing.
- `2e3d6cb` — distinguish TeX variables from character-spaced OCR.
- `e3bbade` — lock six excludes and three keeps before recomputing metrics.
- `12e1290` — simplify the qualified failure reader and make line comparisons
  explicit.
- `360aab2` — exclude the garbled document found in the worst-50 review.
- `d7dab98` — record the outcome-directed provenance and metric caveat.
- `c00cdbe` — add grouped train-OOF component-structure diagnostics.
- `0dd445f` — replace unbounded component size with saturated extent evidence.
- `595be17` — test and reject proposal-boundary probability contrast.
- `1f5f712` — materialize complete contexts for high-scoring false components.
- `0365969` — materialize perpendicular deterministic competing-line roles.
- `3c93403` — measure those roles inside true and false proposed components.
- `f344db3` — add the graded explicit-negative-role fraction to the gate.
- `b5a2f15` — separate exact negative scope from generic document headings.
- `fc444e5` — retain only five fold-stable block features.
- `5d74c6d` — add core-only connected expansion for weaker neighbouring spans.
- `e7ffff3` — train grouped OOF B1 models with deterministic competing roles.
- `8656787` — treat exact negative scope as a hard barrier rather than a
  poison pill for neighbouring blocks.
- `0f7a34e` — add the selection-ineligible whole-source contextual-model
  robustness audit.

Superseded job `2754325` failed in ten seconds before model fitting because an
initial bundle omitted the legacy `line_lr.py` runtime sibling.  It produced no
experiment output.  The runtime bundle was corrected and import-smoked on a
Clariden CPU node.  Job `2754336` then ran successfully for three minutes but
was intentionally cancelled before completing its first wave when telemetry
showed that 20 fits could safely use 20 rather than 10 workers.  Its empty
`b1_r2` directory is not an input to the live chain.

## Remaining automatic gates

1. B1 compares no-header and explicit-header-feature variants on D1 and L4,
   using only line-model OOF probabilities.  H0 is applied only after decoding.
2. B2 runs only if B1's predeclared split/merge error condition fires.
3. The freeze job selects among B0/H0, B1, and an accepted B2 using train-only
   evidence.  If nothing meets the safety gate, it freezes a research-only
   winner rather than silently weakening the gate.
4. Only then is retrospective validation materialized and evaluated once.
5. The final site selects 120 source-balanced highest-risk proposed blocks and
   shows whole documents, feature spans, probabilities, silver/proposed blocks,
   and long-line decisions.  Foivos and Codex decisions are separate browser
   records and are forbidden as tuning data for this version.

The implementation and execution chain are complete.  The remaining work is
the separate Foivos/Codex review itself and the resulting retain/revise/shadow
decision; review decisions must not tune this frozen version.

## Recall-failure presentation — 2026-07-14

The ten readability-qualified validation documents contributing the most
missed bibliography tokens are presented as complete document readers.  The
seven extraction-quality exclusions cannot enter this selection.  The first
reader is deliberately a readable partial-recall case rather than the largest
raw numerical failure.  Final frozen B1 decisions are compared line by line
with the LLM-silver BIB region:

- green means both mark the line as BIB;
- red means silver BIB missed by the classifier;
- blue means classifier-only BIB; and
- neutral means both classify the line as non-BIB.

Every classifier-positive line also carries the ownership-resolved character
feature boxes.  Each line has explicit MODEL and SILVER decisions on the left,
and agreement/disagreement is bounded around the complete line.  Peripheral
statistics, probabilities, character counts, and feature totals were removed.
The interface retains all-lines, decision-context, disagreement-context,
previous/next-error, and first-block navigation.

Clariden job `2755066` rebuilt the qualified ten-document packet after the
worst-50 review in 13 seconds:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/failure_review_qualified_v4
packet sha256 2a58279f7e5ed98de66832a2fac1b762c6d0f26c5b578a1c74b1c2369bc6b3da
receipt sha256 c5e4e6283e8d396cd9c5831edf356f6ee5f8961d248caeb2d7927c4f46ea756c
```

The local presentation is:

```text
/Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-recall-failures-20260714
http://127.0.0.1:8772/
```

Restart command:

```bash
cd /Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-recall-failures-20260714
python3 -m http.server 8772 --bind 127.0.0.1
```

The separate nine-candidate automatic extraction-quality reader, including the
original six excludes and three explicit keeps, is archived and served at:

```text
/Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-validation-quality-20260714
http://127.0.0.1:8773/
```

Both local transfers match the Clariden SHA-256 receipts and passed HTTP 200,
content-length, packet-inventory, Python, and JavaScript syntax checks.
Playwright was available but its Chromium executable was not installed, so no
automated browser screenshot is claimed.

Superseded job `2754894` produced a valid one-document packet because Slurm
interpreted commas in the exported ID list as environment-variable separators.
That artifact is incomplete and is not presented.  Commit `8b88a29` replaced
the interface with a colon-delimited list before v2.  Job `2754908` and its v2
site are also superseded: the raw numerical top ten included three documents
now excluded as unusable extraction artifacts, and its interface exposed
irrelevant statistics rather than direct model/silver decisions.

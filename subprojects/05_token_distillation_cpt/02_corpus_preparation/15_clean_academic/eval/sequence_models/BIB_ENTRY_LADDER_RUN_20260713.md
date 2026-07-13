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

The ten validation documents contributing the most missed bibliography tokens
are presented as complete document readers.  Final frozen B1 decisions are
compared line by line with the LLM-silver BIB region:

- green means both mark the line as BIB;
- red means silver BIB missed by the classifier;
- blue means classifier-only BIB; and
- neutral means both classify the line as non-BIB.

Every classifier-positive line also carries the ownership-resolved feature
bounding boxes and probability.  The interface includes all-lines,
decision-context, and disagreement-context modes plus previous/next error and
first-block navigation.

Clariden job `2754908` completed the ten-document packet in 14 seconds:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/failure_review_top10_v2
packet sha256 01fca9350006409d3815cfd4d63e86e274ef132d3cf4ab08362ee18a3452a70f
receipt sha256 300ecff943be85664b264cda4ae6f7b4bf6369194598c7ebd62b185cc8c75994
```

It contains 29,118 emitted lines and 202,121 missed BIB tokens.  The local
presentation is:

```text
/Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-recall-failures-20260714
http://127.0.0.1:8772/
```

Restart command:

```bash
cd /Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-recall-failures-20260714
python3 -m http.server 8772 --bind 127.0.0.1
```

Superseded job `2754894` produced a valid one-document packet because Slurm
interpreted commas in the exported ID list as environment-variable separators.
That artifact is incomplete and is not presented.  Commit `8b88a29` replaced
the interface with a colon-delimited list before the successful v2 job.

# Bibliography role pipeline implementation — 2026-07-15

## Purpose

This strand keeps the selected P0D bibliography-entry classifier frozen and
learns the surrounding roles needed to construct accurate bibliography blocks.
It does not alter the entry detector and it does not authorize corpus deletion.

The operational roles are:

- `ENTRY`: independent citation evidence and the only block seed;
- `CONTINUATION`: broken citation content that may attach but not seed;
- `FILLER`: layout/extraction material that may bridge supported regions;
- `BIB_HEADER`: included in the block and a hard upward boundary;
- `BIB_SUBHEADER`: included and allowed to connect supported regions;
- `NON_BIB_HEADER`: excluded and a hard boundary;
- `OTHER`: excluded ordinary material; and
- `UNKNOWN`: unresolved supervision, always masked from losses.

The machine-readable contract is `bibliography_role_contract_v2.json`.

## Implemented stages

### 1. Typed heading inventory and review

`bibliography_heading_review.py` first applies a document-wide, high-recall
heading predicate, then retains heading-shaped lines within 30 physical lines
of frozen OOF P0D evidence. This is the region in which a heading can affect the
block decoder; document-wide unrelated chapter headings are not a useful review
target. Every existing trusted heading is included as a recall backstop. The
stage emits a prediction-blinded packet with five raw context lines on each
side, and keeps source identity and P0D probability only in a separate
provenance file.

The complete contextual inventory is not sent blindly to Codex. A second CPU
gate keeps every trusted or deterministic heading candidate, then fills a
source quota by round-robin sampling across work folds, heading-shape families,
and P0D-probability bins. The initial quota is 500 per source; sources with
fewer candidates are reviewed completely. This gives the heading expert broad
negative coverage without paying to dual-review tens of thousands of likely
non-headings. High-scoring unreviewed candidates can be audited in a later
active-learning round.

Two independent `gpt-5.6-luna` Codex passes see neither each other nor the old
labels. Exact agreement is trusted. Disagreements become `UNKNOWN`; there is no
silent majority rule or deterministic adjudication. An agreed `NOT_HEADER`
decision preserves an existing trusted non-heading role instead of erasing it.

### 2. Heading expert

`bibliography_role_experts.py` fits a grouped out-of-fold hierarchy:

1. heading versus not-heading; and
2. conditional `BIB_HEADER` versus `BIB_SUBHEADER` versus `NON_BIB_HEADER`.

The model sees Unicode character 2–5-grams, word 1–2-grams, language-agnostic
line-shape features, blank-line context, document position, and frozen OOF P0D
evidence within 30 lines. Nested grouped tuning selects regularization without
opening validation data.

### 3. Continuation/filler expert

`bibliography_role_features.py` and `bibliography_role_tables.py` create
candidate windows of ±30 physical lines around frozen entry evidence or heading
candidates. The expert sees:

- binary presence and log-counts of the 35 deterministic citation features;
- Unicode line shape, punctuation, script, whitespace, length, and indentation;
- lossless positional non-match summaries;
- entry evidence above and below at radii 1, 3, 5, 10, and 30;
- nearest anchors and position within the local candidate window;
- previous/current and current/next shape transitions;
- the gain from rescoring a joined neighbouring line with the fold-specific
  frozen P0D model; and
- OOF typed-heading probabilities.

It fits three binary heads: connector (`CONTINUATION|FILLER`) versus non-
connector, `CONTINUATION` versus `FILLER` conditional on connector, and `OTHER`
versus the remaining reviewed roles. Elastic-net logistic and shallow histogram
gradient-boosting arms are compared in nested grouped OOF evaluation. If two
independent role reviewers disagreed only between continuation and filler, the
example still supervises the connector head but is masked from the subtype
head.

### 4. Structured block decoder

`bibliography_role_block.py` implements a cost-sensitive linear semi-Markov
decoder over the seven OOF role probabilities. It requires two short P0D entry
seeds, proposes spans only within 30 lines of supported clusters, includes a
main bibliography heading only at the upper edge, permits subheaders inside,
and never crosses a predicted non-bibliography heading. Heading barriers are
decided inside the heading head and cannot be defeated by a high entry score on
the same line.

The structured-margin fit tunes false-positive and spurious-fragment costs by
grouped inner folds. Seed-unreachable gold sequences are measured in the
coverage ceiling and evaluation, but excluded from parameter fitting because
their target cannot be emitted by the conservative decoder. Reports include
line and character precision/recall, exact-document rate, mean IoU, source
slices, zero-bibliography spurious blocks, hard-stop crossings, and seed
coverage. The initial gate requires at least 0.99 line and character precision,
0.95 line and character recall, zero trusted hard-stop crossings, and no more
than 0.02 spurious blocks per reviewed zero-bibliography sequence.

This is deliberately supervised structured prediction, not reinforcement
learning. RL remains unnecessary unless the interpretable cost-sensitive model
shows a specific, reproducible limitation.

## Leakage and provenance controls

- The selected P0D classifier and its grouped OOF probabilities are immutable.
- Every joined-line score uses the model that excluded that line's fold.
- Heading and connector probabilities consumed by the decoder are OOF.
- Document/work folds are inherited from the frozen entry table.
- Validation remains unopened throughout model selection.
- Outputs are create-only and bind their input manifests/hashes, exact code
  commit, and Slurm job ID.
- The historical regional labels remain in `original_region_label`; the v1 to
  v2 migration is additive and lossless.

## Clariden execution order

All table materialization and fitting uses CPU Slurm jobs. The MacBook is used
only for code, review-call coordination, and small review artifacts.

1. `clariden/inventory_bibliography_headings.sbatch`
2. inspect the candidate count and run
   `clariden/select_bibliography_heading_review.sbatch`;
3. inspect the selected source/shape counts and estimated dual-pass call budget;
4. run independent heading review passes A and B;
5. adjudicate the typed heading overlay;
6. `clariden/materialize_bibliography_role_expert_table.sbatch` with
   `COMMAND=heading`;
7. `clariden/train_bibliography_role_expert.sbatch` with `KIND=heading`;
8. materialize the connector table with the same table launcher and
   `COMMAND=connector`;
9. train the connector expert with `KIND=connector`;
10. `clariden/materialize_bibliography_role_block_table.sbatch`; and
11. `clariden/train_bibliography_role_block.sbatch`.

Each stage is a gate: downstream work is not submitted if class coverage,
candidate-window coverage, provenance, or grouped OOF completeness fails.

## Verification before Clariden submission

- 20 focused unit tests pass for role migration, heading adjudication,
  Unicode/positional features, supervision masking, and hard block boundaries.
- Ruff and `git diff --check` pass.
- Every Slurm launcher passes `bash -n`.
- Heading and connector OOF fitting completed on synthetic five-fold data under
  the pinned scikit-learn 1.9.0 runtime, with future deprecations treated as
  errors.

## Current run state

The code is implemented locally. The next irreversible-cost gate is the
Clariden heading inventory. No dual Codex review should be launched until its
candidate count has been inspected, because the inventory deliberately favors
recall and may be substantially larger than the trusted role overlay.

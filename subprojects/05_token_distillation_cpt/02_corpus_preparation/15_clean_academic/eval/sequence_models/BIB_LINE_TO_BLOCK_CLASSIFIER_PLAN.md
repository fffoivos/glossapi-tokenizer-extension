# Bibliography line-to-block classifier plan

## Objective

Build an interpretable bibliography-entry line classifier from the 35 current
deterministic feature families, then use its out-of-fold scores in a separate
block model. Line length is deliberately excluded from the line classifier.
The approximately three-rendered-line length observation is used only by the
block stage: a long line cannot establish a bibliography block, but it can be
absorbed by a block established by surrounding entry evidence.

This remains a comparison against GPT-generated LLM-silver labels. It is not a
human-gold result or authorization to remove corpus text.

## Evidence already available

- The physically test-stripped STRUCT-2K materialization contains 1,392
  documents: 1,118 `train` and 274 `validation`.
- Sources are balanced: 466 `greek_phd`, 461 `kallipos`, and 465
  `openarchives` documents.
- The historical 608-document split named `test` is physically absent and
  remains sealed.
- The corpus has 160 zero-BIB-block, 973 one-BIB-block, and 259
  multiple-BIB-block documents. The one-block reader is a review surface, not
  the training subset.
- Frozen deterministic v2 local evidence reached validation precision 0.9361
  and recall 0.4080. Its conservative coherent decoder reached precision
  0.9919 and recall 0.2169. This shows that local evidence is useful and that
  coherence improves precision, but the existing rules discard too much
  recall.
- The completed joint sequence ladder, Clariden job `2738674`, found its
  character-ngram feature CRF stronger than the other joint ToC+BIB arms on
  retrospective LLM-silver replay. Those results are useful architecture
  evidence but are not directly comparable with the entry-only task defined
  here.

## 1. Freeze two related targets

The current `BIB` annotation describes a bibliography region. The local model
should instead detect bibliography entries. We therefore materialize two
views from the same documents:

### Entry-line target

- `ENTRY = 1`: an emitted line labelled `BIB` that is not in the audited
  header-candidate mask described below.
- `ENTRY = 0`: emitted `O` and `TOC` lines.
- `MASK`: `UNKNOWN`, coverage seams, and audited header candidates.

Headers are masked, never made negative. A false header decision can therefore
remove a training example but cannot teach the classifier that a real entry is
prose. The raw silver `BIB` region target is never modified.

Do not let a regex decide the mask. Build a deliberately high-recall candidate
set from all silver `BIB` lines using only nomination cues:

- known multilingual bibliography-heading vocabulary;
- short, citation-sparse lines at a silver block boundary;
- short internal lines that could be language/source subheadings such as
  `Ελληνόγλωσση βιβλιογραφία`, `Ξενόγλωσση βιβλιογραφία`, or `Πηγές`;
- typography and capitalization cues, without treating them as proof.

Adjudicate every nominated line with document context, including neighbouring
lines and its position inside the silver block. The contextual labels are
`ENTRY`, `BIB_HEADER`, `BIB_SUBHEADER`, `OTHER_STRUCTURE`, and `UNCERTAIN`.
Only `BIB_HEADER` and `BIB_SUBHEADER` are supplied to the separate block model
as positive boundary cues. All non-`ENTRY` candidate outcomes are masked from
the entry-classifier loss; none becomes an `O` negative.

Use two independent Codex judgments for candidate adjudication. Agreement on
`ENTRY` returns the line to the positive set; agreement on header/subheader
masks it and enables the boundary cue; disagreement or uncertainty remains
masked without becoming a cue. Then Foivos and Codex jointly inspect at least
100 source- and position-balanced candidates, with special attention to short
real citations and internal bibliography subdivisions. Freeze the mask only
if this audit finds no real entry incorrectly promoted to a header boundary
cue. Otherwise revise the nomination/adjudication contract and repeat with a
fresh audit sample.

These contextual labels are still LLM-assisted research labels, not human
gold. Their receipt must preserve the prompt, model identity, responses,
candidate hashes, agreement status, and final masking rule.

If contextual adjudication is unavailable, the fallback is conservative:
mask only exact high-confidence heading vocabulary and leave every other
silver `BIB` line positive. Do not infer a broad header mask from line length,
capitalization, or lack of citation features alone.

The resulting high-precision header/subheader cue is retained separately as
block-boundary evidence. The entry classifier itself is neither rewarded nor
penalized for recognizing it.

### Region/block target

Preserve the original continuous silver `BIB` runs, including their headings,
as the target spans for the second stage. Missing blank physical lines do not
split a block. A physical-coordinate gap over 64 lines is a hard boundary, as
in the existing block audit.

This separation lets the local model specialize in entries while the block
model can recover the whole removable bibliography region.

## 2. Dataset use and leakage controls

Use all 1,392 available documents:

- zero-block documents teach the decoder not to hallucinate a bibliography;
- one-block documents provide the cleanest boundary and long-line cases;
- multi-block documents ensure that the decoder can stop and restart rather
  than merging every citation-like region in a document.

Never split lines randomly. Preserve the canonical `work_id` grouping and
source stratification.

1. Within the 1,118 `train` documents, create deterministic five-fold grouped
   cross-validation splits stratified by source and bibliography-block count.
2. Produce out-of-fold line scores for every train line. These are the only
   line-model scores the learned block models may see during fitting.
3. Fit a final line model on all 1,118 train documents and predict the 274
   validation documents only after the line and block choices are frozen.
4. Do not open or predict the historical 608-document partition.

The 274-document validation split has already been used by earlier research,
so report it as retrospective LLM-silver validation, not as an untouched test.
Production confidence requires the separate source-matched review described
below.

## 3. Materialize the line feature table

For every emitted line, preserve:

- `document_id`, `work_id`, source, canonical split, coverage mode, and
  absolute physical line index;
- the 35 resolved deterministic feature counts and their exact character
  spans;
- a binary-presence version of each count;
- the entry target, header mask, original region label, and block identifier;
- line character length for the block stage only.

Do not give line length, token length, match density, matched-character
coverage, document position, neighbouring labels, or neighbouring feature
values to the line classifiers in this experiment. This isolates the question
of whether learned weights and a bias can combine the explicit citation
features better than equal weighting.

Counts are stored losslessly. Transformations are applied by each model arm,
not baked into the shared table.

## 4. Line-classifier arms

All learned arms estimate

`P(ENTRY) = sigmoid(bias + sum(weight[j] * feature[j]))`.

Use the same folds, labels, regularization search, and evaluation code for every
arm.

| ID | Inputs | Model | Purpose |
|---|---|---|---|
| L0 | Binary presence | Equal weight sum plus threshold | Current inspectable no-training baseline |
| L1 | Binary presence | L2 logistic regression | Learn which feature kinds matter and which are counter-evidence |
| L2 | Raw counts | L2 logistic regression | Test the user's literal count-weighting proposal |
| L3 | `log1p(count)` | L2 logistic regression | Reduce domination by very long author lists while retaining frequency |
| L4 | Binary presence + `log1p(count)` | Elastic-net logistic regression | Let existence and repetition have different effects; expected primary linear candidate |
| D1 | Same inputs as L4 | Small depth-limited boosted-tree model | Nonlinear diagnostic ceiling, not the preferred deployable model |

The tree arm must not receive text, length, density, or document context. It
answers only whether strong nonlinear interactions remain after the explicit
feature design. If its gain is negligible, retain the simpler linear model.

For the linear arms:

- search a small predeclared regularization grid inside train folds;
- retain the natural class prevalence during fitting and calibrate thresholds
  separately rather than hiding class imbalance in arbitrary resampling;
- standardize count columns for optimization, then export coefficients and the
  bias back into original feature units for inspectability;
- allow negative coefficients for prose-lead and other counter-signals;
- archive the complete coefficient table, fold predictions, calibration curve,
  and receipt for every arm.

Do not choose a final line winner from line F1 alone. Carry at least the best
binary arm, best count arm, L4, and L0 into the block comparison because a
slightly weaker isolated-line model may yield better blocks.

## 5. Line-level evaluation

Report on out-of-fold train predictions before touching validation:

- precision-recall curve, PR-AUC, precision, recall, and F1;
- calibration/Brier score;
- per-source and per-coverage results;
- false-positive documents and longest consecutive false-positive run;
- results by count of deterministic features;
- results by length bands (`<=110`, `111–220`, `221–330`, and `>330`
  characters), for analysis only;
- separate errors for `O`, `TOC`, masked headers, and entries.

The masked headers are diagnostic rows, not scored negatives. We should still
inspect how often each model fires on them so the block stage knows whether a
separate heading cue is needed.

## 6. Port line outputs into block models

Every block model consumes out-of-fold line logits/probabilities during
training. It never receives in-sample scores from a line model fitted on the
same document.

### B0: explicit anchored decoder

This is the first and most inspectable implementation.

1. A normal-length line above a high probability becomes an anchor.
2. A candidate block requires a local cluster, initially two anchors within
   three emitted lines or three anchors within five.
3. Small gaps may be bridged only between established anchors.
4. Once a block is established, weaker and long lines may be included between
   or immediately adjacent to anchors.
5. Long lines may never create or extend an unconfirmed block on their own.
6. Header cues may extend a confirmed block boundary but may not establish a
   block without entry anchors.
7. Coverage seams and strong prose/barrier runs terminate blocks.

The presentation-derived initial seed-length limit is approximately 330
normalized characters: about 110 characters per rendered line times three.
Freeze a small grid such as `{280, 330, 380}` rather than treating the display
measurement as ground truth.

Tune only a bounded set of parameters on out-of-fold train predictions:

- anchor probability;
- seed-length limit;
- anchors required and window width;
- maximum bridge gap;
- lower inside-block probability;
- adjacent expansion distance;
- number/strength of barrier lines needed to terminate.

### B1: constrained linear-chain CRF

Train a small CRF over states such as `O`, `BIB_START`, `BIB_INSIDE`, and
`BIB_GAP`. Observations are the frozen line-model logit, header cue, barrier
cue, character length, and local score aggregates. Add a hard transition mask
so a line over the seed-length limit cannot transition from `O` into
`BIB_START`; it may transition within an already established block.

This tests learned coherence and transition weights without abandoning the
long-line safety rule.

### B2: filtered semi-Markov segment model

Attempt this only if B1 still fragments or merges blocks badly. Generate
candidate spans from the same safe anchors and score whole spans using:

- anchor count and density;
- mean, minimum, maximum, and quantiles of entry logits;
- number and total length of bridged gaps;
- header/boundary evidence;
- long-line count inside the proposed span;
- barrier evidence at and within the boundaries.

The semi-Markov model is the most direct learned block model, but it is a
second escalation rather than the first experiment.

## 7. Block-level comparison

Cross every retained line arm with B0. Train B1 for the two strongest and most
interpretable line arms. Run B2 only if the error analysis justifies it.

Report:

- line and token precision/recall after block decoding;
- exact-span and IoU>=0.5 block precision/recall;
- boundary error in physical lines;
- spurious blocks per zero-block document;
- split/merge errors on multi-block documents;
- fraction of documents with any false deletion;
- longest consecutive false-positive token run;
- per-source and per-coverage metrics;
- long lines rejected outside blocks, recovered inside true blocks, and
  incorrectly absorbed inside false blocks.

The main selection criterion is block-level safety and coverage, not isolated
line F1. Prefer the simplest model within uncertainty of the best result.

## 8. Freeze, retrospective validation, and joint review

1. Freeze the line arm, all coefficients, probability calibration, block arm,
   and block thresholds using only grouped train-fold evidence.
2. Run once on the 274-document validation split and archive the immutable
   predictions and receipt.
3. Build a review site showing complete document context, line probabilities,
   feature spans, proposed blocks, silver blocks, and explicit reasons why a
   long line was included or rejected.
4. Foivos and Codex independently review a source-balanced set of at least 100
   highest-risk proposed removals, emphasizing isolated long lines, block
   boundaries, and blocks proposed in silver-zero documents.
5. Record disagreements separately from LLM-silver replay metrics. Do not tune
   on this review and then report it as held out.

If further tuning is required, start a new version and obtain a fresh review
sample.

## 9. Execution and artifacts

Run feature-table materialization, model fitting, out-of-fold prediction, and
block evaluation on Clariden CPU nodes under account `a0140`. The MacBook is
limited to code changes, status checks, small synthetic tests, receipts, and
serving the review site.

Each immutable experiment directory must contain:

- exact input hashes and document-selection receipt;
- header-mask audit and label-view receipt;
- fold manifest grouped by `work_id`;
- feature schema and extractor version;
- model coefficients/artifact and calibration parameters;
- out-of-fold and validation predictions;
- line and block reports with per-source errors;
- runtime/resource receipt;
- a no-production-change decision field.

The final inference contract is:

`document lines -> deterministic counts/spans -> frozen line weights+bias -> calibrated entry probabilities -> frozen block decoder -> proposed bibliography spans`.

Only after the independent review gate passes should the selected proposal be
ported into the corpus-cleaning pipeline, initially in audit/no-delete mode.

## Planned order

1. Freeze and audit the entry/header/region label contract.
2. Materialize the receipt-bound feature table and five grouped folds.
3. Fit L0–L4 and D1; emit out-of-fold line predictions.
4. Compare all retained line arms through B0.
5. Train B1 on the two strongest interpretable arms.
6. Decide from error analysis whether B2 is warranted.
7. Freeze one complete line-to-block configuration.
8. Run the single retrospective validation and build the joint-review site.
9. Review at least 100 high-risk cases and decide whether to revise, retain as
   audit-only, or promote to cleaning-pipeline shadow mode.

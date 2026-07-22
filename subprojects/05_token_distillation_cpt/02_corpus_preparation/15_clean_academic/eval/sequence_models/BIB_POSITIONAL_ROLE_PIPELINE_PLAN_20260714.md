# Bibliography positional-entry and role-aware block plan

Date: 2026-07-14

Status: implementation plan; no corpus-removal authorization

## Decision

We will stop treating every non-header line inside a silver `BIB` region as a
bibliography entry. The replacement has two deliberately separate layers:

1. a line-local `ENTRY_ANCHOR` model whose only positives are structurally
   recognizable bibliography entries and whose negatives have separately
   adjudicated non-anchor roles; this tests whether the character locations of
   the existing deterministic feature matches add useful information; and
2. a role-aware document decoder that uses separate evidence for entries,
   continuations, filler, headers, and stopping boundaries to reconstruct the
   complete removable bibliography region.

The order matters. The positional model must not be trained on the current
conflated target, or it will learn the locations typical of headers,
continuations, and layout filler while being told that all of them are entries.

## What the current implementation already gives us

- `bibliography_v2.py` already returns exact, ownership-resolved match spans in
  NFKC-normalized character coordinates. The feature explorer already displays
  and serializes those offsets. We do not need another regex implementation.
- The current feature table keeps only 35 feature counts. It discards the spans
  before model fitting.
- The current table has 939,014 lines from 1,118 training documents / 1,100
  grouped works. It masks 796 exact bibliography headers but labels all other
  silver `BIB` lines as `ENTRY`, yielding 138,447 nominal positives.
- This target is especially weak for short lines: the present D1 count model's
  `<=110`-character recall is about 0.556, while its `111-220` recall is about
  0.935. Some of that apparent error is likely role-label error rather than an
  inability to recognize complete entries.
- `deterministic_structure.py` already contains useful role concepts and
  conservative header/entry/continuation rules.
- `bibliography_entry_blocks.py` already supplies an inspectable anchor,
  bridge, and header-attachment baseline. It should remain frozen for paired
  comparison rather than being silently rewritten.
- `bibliography_signal_tcn.py` is useful as a deterministic training scaffold,
  but its current target is regional `BIB` membership. It is not an entry-role
  model and must not be reported as one.

## 1. Freeze the new label contract

Every reviewed line receives a role plus independent metadata. The roles are:

| Role | Meaning | Can seed a block? | Included after block confirmation? |
|---|---|---:|---:|
| `ENTRY_ANCHOR` | A recognizable bibliography entry, complete enough to supply independent citation evidence | yes | yes |
| `CONTINUATION` | Citation content broken across lines or too incomplete to establish an entry alone | no | yes, when attached to an entry region |
| `FILLER` | A separator, Markdown/table rule, OCR/layout fragment, or subdivision material that belongs inside the bibliography region but is not citation content | no | yes, only in context |
| `HEADER` | Overall bibliography/references heading | no | yes, only when attached to a confirmed block |
| `SUBHEADER` | Internal source/language/type subdivision inside a bibliography | no | yes, only when attached to a confirmed block |
| `NON_BIB` | Body prose or other document material | no | no |
| `UNKNOWN` | Insufficient evidence or annotator disagreement | no | never used in a supervised loss |

In addition, retain a boundary flag for `SOFT_STOP` and `HARD_STOP`. Chapter
headings, sustained prose, coverage gaps, and other clear post-bibliography
material can then stop expansion without pretending to be another kind of
bibliography line.

The materialized table keeps these concepts separate:

- original silver region membership and block ID;
- corrected `line_role`;
- whether a line is entry-seed eligible;
- whether it is attachable after a block exists;
- whether it should be removed with a confirmed block;
- label origin, confidence, reviewer/model identity, prompt/schema hash, and
  adjudication status.

### Loss masks for the individual models

The primary line-local entry model estimates anchor eligibility and uses:

- positive: reviewed `ENTRY_ANCHOR` only;
- negative: trusted `NON_BIB` plus adjudicated `HEADER`, `SUBHEADER`,
  `CONTINUATION`, and `FILLER`;
- masked: unreviewed regional `BIB` lines, disagreements, and `UNKNOWN`.

Thus only good entries are positive and the model explicitly learns that the
other reviewed roles cannot seed a block. A predeclared ablation will mask the
adjudicated in-block non-entry roles and use only trusted `NON_BIB` negatives;
this checks whether one-vs-rest training unfairly suppresses weak but valid
entry styles. A separate header model sees headers as positives. A contextual
attachment model sees continuations/filler as positives only in candidate
regions where those roles are operationally meaningful.

## 2. Correct and bootstrap the role dataset

### 2.1 Profile the existing silver blocks

Run one prediction-blind CPU profiling pass over every training line. For each
silver block, record:

- exact header/subheader matches;
- line length, feature count, matched-character coverage, and unmatched runs;
- block-relative line position;
- short lines (`<=110`), long lines (`>330`), zero/low-match lines, and
  contiguous runs of them;
- strong-entry evidence on either side;
- the first and last lines plus several `NON_BIB` lines outside each boundary;
- obvious separators, Markdown rules, table fragments, and extraction noise.

This pass nominates review cases; it does not assign truth from model score.

### 2.2 Create the first role-review packet

Start with 60 complete blocks, 20 from each of Greek PhD, Kallipos, and
OpenArchives. Stratify the sample so it contains roughly equal coverage of:

- dense, conventional references that should become clean `ENTRY_ANCHOR`
  examples;
- short/low-match/zero-match internal lines;
- long wrapped entries and continuations;
- exact and non-exact headers/subheaders;
- block boundaries, next-chapter headings, and citation-like body prose.

Show the whole block plus at least five lines before and after it. Very large
blocks may be split into overlapping chunks, but each chunk must preserve the
block coordinates and enough surrounding anchors to classify filler and
continuations contextually.

One structured Codex call should label a block or chunk, not one line per call.
During the first judgment, hide detector highlights, model scores, nomination
strata, and expected labels; reveal them only after the decision so annotation
does not merely reproduce the feature system being evaluated. Use a second
independent pass for disagreements and all `UNKNOWN` cases. Store the raw
responses and receipts. Foivos and Codex then jointly audit at least 30 blocks,
10 per source, with an interface that can correct every role and both
boundaries.

If `CONTINUATION` versus `FILLER` has poor reviewer agreement, preserve both raw
votes but collapse them to the operational superclass `ATTACHABLE` for the
first decoder. Do not force a distinction that the evidence cannot support.

### 2.3 Expand through active learning, not circular pseudo-truth

After the pilot taxonomy is stable:

1. label a source-balanced development set of blocks and boundary negatives;
2. train role models with work-grouped out-of-fold predictions;
3. send disagreements, low-confidence rows, unexpected short/long rows, and
   boundary crossings back to contextual review;
4. add only adjudicated labels to the trusted role table;
5. retain high-confidence model-only labels as `PROVISIONAL`, never as sealed
   evaluation truth.

The initial scaling target is 300 reviewed blocks, 100 per source, but the
review yield decides whether this is sufficient. We care about the number and
variety of each role, not an arbitrary total line count.

### 2.4 Build a genuinely independent role audit

The existing 274-document validation split is retrospective development
evidence because it has already informed prior work. Before comparing final
architectures, select fresh source-matched works and fully role-label the
chosen blocks plus their boundaries.

Exclude all STRUCT-2K works, the prior 30-document review, and the complete
recent 90-document candidate pool by canonical work identity and exact/near
duplicate checks. Use the verified source-specific identities:

- Greek PhD: `doc_id -> source_doc_id`;
- Kallipos: `doc_id -> filename/work_key`;
- OpenArchives: base `source_doc_id` or current `work_id`, not the composite
  current `source_doc_id`.

Select random full documents before looking at model predictions, not only
model-proposed error cases. Seal this audit before model selection.
Pseudo-labels and reviewed development packets must never enter it.

## 3. Preserve and encode character locations

### 3.1 Feature-table v2

Keep the current `counts [N,35]` and add a sparse span store:

```text
match_ptr      [N+1] uint64
match_feature  [M]   uint8
match_start    [M]   uint32
match_end      [M]   uint32
nfkc_length    [N]   uint32
```

Assert per-line/per-feature count-to-span parity. Normalize offsets by the
NFKC string length, not the pre-normalization raw character count.

For the meaning of "non-matches", form the union of semantic citation spans.
Exclude generic `punctuation_count`, `prose_lead_count`, and full-line
`table_row_count` from that union; otherwise those broad detectors would erase
the area whose structure we want to measure.

Represent the complement with:

- unmatched prefix and suffix fractions;
- unmatched total fraction;
- longest unmatched-run fraction and centre;
- unmatched-run count and mean run length;
- per-position character-shape channels for letters, digits, whitespace,
  punctuation/symbols, and other/OCR characters.

This answers both sides of the positional question: where the citation matches
occur and what kind of material occupies the gaps between them. Raw text,
document position, neighbouring lines, source identity, and absolute length
stay out of the primary entry model.

### 3.2 Entry-model experiment ladder

All arms use identical corrected labels, work-grouped folds, and tuning budget.

| Arm | Representation | Model | Question |
|---|---|---|---|
| `P0` | Binary + `log1p` counts | Existing L4 elastic-net logistic baseline | What does the corrected target alone fix? |
| `P0D` | Same counts | Existing depth-limited D1 | Nonlinear count-only ceiling |
| `P1` | Counts + normalized first, last, mean centre, and coverage for every feature | Elastic-net logistic | Do simple location summaries help? |
| `P1G` | P1 + unmatched-gap summaries | Elastic-net logistic | Do the non-matches explain additional errors? |
| `P2` | 8/16-bin feature coverage map + unmatched character maps + scalar counts/gaps | Sparse elastic-net logistic | Can an interpretable coarse spatial grid capture feature order? |
| `P3` | 64-bin spatial map + scalar counts/gaps | Small 1-D residual CNN | Is a specialized spatial model materially better? |

At 16 bins, P2 has 35 detector channels plus five unmatched-character channels
(`40 x 16`) and 77 count/gap scalars. Rasterize it to sparse CSR form.

P3 is attempted only if P2 demonstrates a real positional gain. It uses three
small convolutional/residual layers, explicit normalized coordinate channels,
four-region plus global mean/max pooling, and a small classification head. The
coordinate channels and regional pooling are important: global pooling alone
would largely discard the absolute beginning/middle/end information we are
trying to test. Build 64-bin tensors per batch from the sparse spans instead of
materializing the approximately two-gigabyte dense table.

Line length remains block-stage metadata in the primary comparison. A single
predeclared diagnostic arm may add `log1p(nfkc_length)` to determine whether it
helps, but it must be reported separately rather than confounded with the
location experiment.

### 3.3 Required positional ablations

- counts only;
- counts plus matched-span locations;
- add scalar unmatched gaps;
- add unmatched character-shape maps;
- remove all nonmatch channels;
- cyclically shift the complete position map within each line, preserving
  relative feature geometry but destroying absolute position;
- independently shift feature channels, preserving counts/span sizes but
  destroying cross-feature ordering;
- compare 8, 16, and 32 bins before permitting the 64-bin CNN.

The shuffled controls are essential. If they retain the gain, the gain did not
come from the location information we intended to test.

## 4. Break the final detector into explicit stages

### Stage E: detect independent entry evidence

Use the selected P0-P3 model to assign `P(ENTRY_ANCHOR)` to each line. A high
score is evidence for a citation entry, not permission to remove a line by
itself.

### Stage S: form entry-seeded islands

Create candidate islands only from clusters of independent entry evidence.
The existing normal-length anchor constraint remains a block-stage safeguard:
a very long line may be attached later but cannot establish a block alone.

### Stage C: attach continuations and weak entries

Train a contextual candidate model on reviewed `CONTINUATION` lines versus
boundary `NON_BIB` controls. Its inputs may include local match geometry,
neighbouring entry probabilities, side/distance to the nearest seed, and hard
barrier signals. It cannot create a block.

### Stage F: bridge filler

Use exact deterministic rules for unambiguous Markdown/table separators and a
contextual `ATTACHABLE` model for ambiguous filler. A filler line may bridge
two independently anchored regions or fill a small internal gap. It may not
extend indefinitely from only one weak edge and may never seed a bibliography.

### Stage H: attach headers and subheaders

Run the current exact multilingual header detector first. Train a separate
header fallback only on adjudicated misses and hard generic-heading negatives.
An overall header attaches backwards within a bounded window of a confirmed
entry island; internal subheaders attach between confirmed bibliography
evidence. A header alone never creates a block.

### Stage B: stop at real boundaries

Use hard deterministic barriers plus a contextual boundary model trained on
the lines immediately outside reviewed blocks. Prefer trimming an uncertain
edge to swallowing a new chapter. Sustained prose, chapter headings, large
physical gaps, and extraction seams terminate growth.

### Stage D: emit the removable region

The decoder emits both:

- per-line roles and probabilities, for inspection; and
- final bibliography spans containing attached `HEADER`, `SUBHEADER`,
  `ENTRY_ANCHOR`, `CONTINUATION`, and `FILLER` lines.

This makes errors attributable. We can tell whether a false positive came from
entry evidence, an over-permissive bridge, a mistaken header, or a failed stop
instead of adding more opaque block rules.

## 5. Evaluation and overfitting controls

### Entry model

- paired five-fold out-of-fold predictions grouped by canonical work;
- PR-AUC, Brier score, and recall at matched high precision;
- source, length, coverage, feature-count, and role strata;
- paired work-cluster bootstrap confidence intervals;
- a nested grouped selection loop or a predeclared small hyperparameter grid,
  so larger positional arms do not receive more tuning opportunities.

Advance a positional arm only if its lower 95% bound for recall gain at matched
precision is positive, no source materially regresses, and shuffled-location
controls lose the gain.

### Role and block pipeline

Report separately:

- entry precision/recall;
- continuation and filler attachment precision/recall;
- header/subheader attachment and false-attachment rates;
- boundary overshoot and undershoot in lines/tokens;
- exact-block and block-IoU metrics;
- line/token precision and recall for the final removable region;
- chapter-heading crossings and spurious blocks in zero-bibliography works;
- per-source and extraction-quality results.

Compare every new decoder with the frozen current count-only/B0/H0 and signal
TCN baselines using identical documents. A research model does not become a
cleaning model merely by improving F1: before corpus removal, it needs a
predeclared high-precision deployment gate (provisionally 0.99) and no severe
source-specific failure.

Keep raw-silver region metrics and reviewed-role/block metrics as separate
evidence tiers. An apparent gain against a known silver omission is not counted
as a reviewed false positive, and reviewed corrections do not silently rewrite
the immutable STRUCT-2K labels.

### Leakage controls

- split by canonical work, never by line or document fragment;
- use out-of-fold entry scores when fitting attachment/block models;
- apply the five frozen prediction-blind extraction-quality exclusions to
  fitting only, without redefining validation;
- use only out-of-fold predictions when bootstrapping development labels;
- freeze the role contract, features, thresholds, and decoder before opening
  the independent audit;
- keep every deterministic, Codex-adjudicated, human-corrected, and model-only
  label distinguishable in receipts.

## 6. Implementation order

### Phase 0 — contracts and frozen baselines

1. Freeze this role contract and the current count-only/B0/H0 outputs.
2. Define table-v2 and review JSON schemas.
3. Predeclare folds, hyperparameter budgets, metrics, and advancement gates.

### Phase 1 — dataset correction

1. Run the all-line/block profile on Clariden CPU nodes.
2. Build and review the 60-block pilot.
3. Resolve taxonomy disagreements and freeze role-contract v1.
4. Build the trusted development role table and independent audit inventory.

### Phase 2 — positional entry experiment

1. Add sparse spans and nonmatch encodings to table v2.
2. Refit corrected P0/P0D baselines.
3. Run P1/P1G, then P2 and its ablations.
4. Run P3 only if P2 passes its advancement gate.
5. Freeze one entry encoder and threshold family from train OOF evidence.

### Phase 3 — separate role evidence

1. Audit and extend deterministic header/subheader coverage.
2. Train the contextual continuation/attachable candidate model.
3. Add exact filler rules and train the ambiguous filler bridge arm.
4. Train/calibrate the boundary-stop model on block edges.

### Phase 4 — role-aware decoder

1. Implement E -> S -> C/F -> H -> B -> D as inspectable stages.
2. Run paired work-grouped block evaluation against frozen baselines.
3. Build a review surface that displays every stage's contribution and permits
   role and boundary correction.

### Phase 5 — bootstrap and freeze

1. Review uncertainty/disagreement cases from fresh works.
2. Retrain only after adding adjudicated labels with full provenance.
3. Freeze one candidate and run the sealed source-matched role/block audit.
4. Produce a deployment receipt or explicitly retain research-only status.

Phases 1 and the table-v2 implementation can overlap, but no positional model
result is valid until it uses the corrected role target. Header/filler rule
work can proceed in parallel with the entry ladder. Final decoder selection is
sequential after all three evidence streams are frozen.

Operational profiling, table construction, cross-validation, and model fitting
run on Clariden CPU nodes. The MacBook is limited to code/report editing,
coordination, and the local review interface.

## 7. Planned artifacts

Preserve current modules as frozen baselines. Add new, versioned artifacts
rather than changing their semantics in place:

- `bibliography_role_contract_v1.json`
- `bibliography_role_dataset.py`
- `bibliography_positional_features.py`
- `bibliography_positional_entry_models.py`
- `bibliography_role_review.py`
- `bibliography_attachment_models.py`
- `bibliography_role_block_decode.py`
- Clariden CPU launchers and immutable result receipts for each phase

Reuse:

- feature spans from `bibliography_v2.py`;
- grouped-fold and table conventions from `bibliography_entry_dataset.py`;
- fitting/metric conventions from `bibliography_entry_models.py`;
- deterministic roles from `deterministic_structure.py`;
- B0/H0 baseline interfaces from `bibliography_entry_blocks.py`;
- CPU/OOF scaffolding from `bibliography_signal_tcn.py`; and
- fresh-work selection from `source_matched_holdout.py`.

## Definition of done

This strand is ready for a cleaning decision only when:

1. the entry target contains only adjudicated, structurally recognizable
   entries and its other in-block roles are separately preserved;
2. the positional gain, if any, survives location-shuffle ablations,
   work-grouped OOF comparison, and all three sources;
3. headers cannot seed blocks, filler cannot seed or unilaterally extend them,
   and boundary stops prevent the documented next-chapter overrun pattern;
4. the full role-aware decoder improves the final region metrics and passes the
   frozen high-precision/source-safety gates on fresh works; and
5. every training and evaluation label has auditable provenance, with no
   model-only pseudo-label counted as independent test truth.

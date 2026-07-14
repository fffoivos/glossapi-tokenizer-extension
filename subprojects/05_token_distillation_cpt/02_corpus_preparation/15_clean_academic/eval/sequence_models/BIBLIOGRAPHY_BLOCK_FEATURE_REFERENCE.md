# Bibliography block feature reference

This reference describes block-level evidence in plain language.  It is
separate from the 35 deterministic line detectors documented by the feature
explorer.  A line detector asks, “does this line contain a bibliographic
shape?”  A block feature asks, “do several such lines form the kind of coherent
region that can safely be treated as a bibliography?”

The design rule is to give each item one job.  Line-level bibliographic shapes
remain ownership-resolved and perpendicular.  Block-level controls may reuse
the line model's combined probability, but they must not introduce a second
regex for the same citation shape.

## B1 proposal control

### Deletion bias

- **Stage:** B1 sequence proposal.
- **Question:** How much extra evidence must B1 see before proposing a line as
  part of a bibliography?
- **Expected behaviour:** A larger value produces fewer, more conservative
  lines.  A smaller value lets B1 propose a more complete region.
- **Not evidence:** This is a calibration control, not a bibliography feature.
  It must be selected on grouped train OOF predictions, never on validation.
- **Reason for the 2026-07-14 change:** The previous value `2.0` optimized
  isolated-line precision and removed whole blocks even when the frozen line
  model contained many strong citation lines.  The new experiment permits a
  lower-bias proposal only when the independent component gates below pass.

### Character length observation

- **Stage:** B1 emission score in the original model.
- **Question:** Is this line unusually long?
- **Original implementation:** Both `log1p_char_length` and a binary
  `over_seed_length_limit` value were supplied to every BIB state.  The learned
  weights strongly penalized long lines even after a block had started.
- **Problem:** This duplicates the start-safety rule and violates the intended
  contract.  A long bibliography entry should not start a block, but it should
  be recoverable inside one.
- **Ablation:** `no_length` removes both length observations from CRF emissions
  while retaining the hard Viterbi rule that a line over the seed limit cannot
  enter `B-BIB` or `S-BIB`.
- **Perpendicularity:** Length is structural safety evidence, not a citation
  detector.  In the corrected variant it has only the one start-gating job.

### Physical document position

- **Stage:** B1 emission score.
- **Question:** How far through the physical document does this line occur?
- **Expected use:** End matter is more likely to contain a bibliography.
- **Risk:** Chapter bibliographies, publication lists, and reference sections
  can appear much earlier.  Position can therefore suppress a coherent block
  whose lines otherwise look bibliographic.
- **Ablation:** `no_position` removes only this prior.  The combined
  `no_length_or_position` variant tests whether its effect is independent of
  the length correction.
- **Perpendicularity:** Position is a document-layout prior.  It does not
  overlap any author, date, title, identifier, or publication-coordinate
  detector.

## Competing-role sequence observations

The role-aware B1 experiment gives the sequence model one-hot observations for
explicit non-bibliography line roles.  These are not hard vetoes: the CRF can
retain a marked line inside a strongly supported bibliography sequence.  They
are also not citation detectors and do not add a second author/date/page rule.

Each line can have at most one of these eight observations:

- **Figure caption:** explicitly begins as a numbered figure or image caption.
- **Table or equation:** explicitly has table-row or equation structure.
- **Exact negative-scope heading:** names a known non-bibliography section such
  as notes, abbreviations, figure/table lists, or related material.
- **Generic Markdown heading:** is a heading whose scope is unknown; kept
  separate because a bibliography can contain source/language subheadings.
- **Footnote:** has an explicit footnote shape and is not also a valid numbered
  bibliography entry.
- **Running or enumerated prose:** has an explicit narrative, inline-citation,
  or long enumerated-prose shape.
- **Legal or procedural line:** explicitly begins as legal/procedural body text
  without a legal bibliography-citation form.
- **Other explicit negative role:** one of the remaining deterministic
  non-bibliography shapes not owned by the seven specific roles above.

The two calibration controls remain **deletion bias** and the independent
**minimum anchors / minimum component lines** gate.  They are swept on grouped
train OOF predictions.  The experiment compares the same no-length and
no-length-plus-no-position B1 variants, so the new evidence is isolated to
these role observations.

### Train-OOF result

The role effects were strongly fold-stable for captions, tables/equations,
exact negative headings, generic headings, footnotes, and running/enumerated
prose: every fold learned a negative BIB emission contrast.  The legal role was
weak and reversed in one fold; the catch-all other role was effectively zero
and changed sign.  Those two cues require an ablation before they can be
trusted as final features.

The role-aware CRF improved the permissive frontier to 93.14% line recall and
96.84% token recall at 96.07% line precision.  It did not meet the frozen
safety rule because it still created 0.308 spurious blocks per silver-zero
document.  It is therefore a higher-quality proposal model, not an accepted
removal policy.  The registered next test applies the stable five-feature
component gate to these proposals.

## Anchored-component gates

### Strong anchor

- **Stage:** Proposed-component certification.
- **Question:** Is this a normal-length line that the frozen entry model regards
  as strong bibliography evidence?
- **Definition:** Entry probability at least `0.70` and normalized character
  length at most the already-frozen seed limit (`380`).
- **Expected behaviour:** Author/date/publication-shaped lines can certify a
  region.  Long lines may be retained inside a certified block but cannot
  certify one.
- **Perpendicularity:** This does not add another author, date, DOI, or page
  detector.  It consumes the single combined line-model output.

### Minimum anchor count

- **Stage:** Proposed-component certification.
- **Question:** Does the proposed component contain several independently
  strong entry lines?
- **Expected behaviour:** A real bibliography block normally contains repeated
  entries.  A lone citation in prose, a footnote, or a coincidental DOI does not
  satisfy the gate.
- **Failure it addresses:** Lowering B1's deletion bias can recover weak and
  long lines, but without this gate it could also create isolated false
  removals.
- **Non-overlap:** It counts strong anchors; it does not measure component size.

### Minimum component lines

- **Stage:** Proposed-component certification.
- **Question:** Is the proposed component large enough to represent a document
  region rather than a single citation-like line?
- **Expected behaviour:** Small isolated fragments are rejected even when one
  line is a strong anchor.
- **Failure it addresses:** Citation-shaped examples occur in body prose,
  captions, publication lists, and footnotes.  The cleaning task targets
  organized bibliography regions rather than every such line.
- **Non-overlap:** It measures structural extent only; it does not inspect
  citation contents or anchor strength.

## Header attachment

### Exact header candidate (H0)

- **Stage:** After a block has already been certified.
- **Question:** Is an immediately preceding line an exact multilingual
  bibliography heading or subheading?
- **Expected behaviour:** Attach the heading to the removable region.
- **Safety rule:** A header can extend a certified block but cannot create one.
- **Perpendicularity:** Header vocabulary is structural evidence, not entry
  evidence; it is masked from entry-line training.

## Learned component gate

The fixed count gate did not pass the train-OOF safety rule: permissive B1
proposals reached roughly 95% token recall but also retained too many false
regions.  The next experiment therefore learns how to accept a complete
proposed component.  It compares an L2 logistic model with a shallow
monotonic-gradient-tree model.  Models and decisions are grouped by the
existing work-level folds, and the five unusable training extractions are
excluded by the independent text-quality review.

### Saturated minimum extent

- **Question:** Is the proposed contiguous region large enough not to be an
  isolated citation?
- **Representation:** Component line count divided by 32 and capped at `1.0`.
  A 10-line component receives `0.3125`, a 32-line component receives `1.0`,
  and a 400-line component still receives only `1.0`.
- **Expected direction:** Positive.
- **Reason for the 32-line saturation:** In the train-OOF diagnostic, accepted
  true components had median extent 34 lines, while accepted false components
  had median extent 10 and 75% were at or below roughly 21 lines.  Thirty-two
  is a round structural scale between those groups.  It is not a hard cutoff:
  short real bibliographies retain partial positive evidence.
- **Reason for saturation instead of raw size:** Permissive proposals can merge
  long stretches of citation-rich prose.  Such false components were often
  larger than true bibliographies, so “ever larger is safer” is not a valid
  monotonic rule.  Saturation rewards reaching a substantial region but gives
  no extra credit to pathological expansion.
- **Non-overlap:** This measures extent only; it does not inspect line content.

### Rejected exploratory additions (2026-07-14)

These measurements were evaluated on train OOF components but deliberately
not added to the gate:

- **Table-row fraction:** true and false accepted components had nearly the
  same mean, and both had median zero.  It does not explain the general error.
- **Prose-lead fraction:** almost always zero in both groups, so it adds no
  useful block discrimination.
- **Proposal-island count:** both true and false components usually belonged
  to one or two permissive proposal islands.  It did not identify scattered
  false regions reliably.
- **Outside-boundary probability:** the permissive decoder already stops where
  frozen line probabilities fall, so both true and false chosen components had
  nearly identical low probability immediately outside their boundaries.
  Adding this would mostly restate how the proposal was generated rather than
  provide independent evidence.
- **Author-start and publication-tail fractions:** these separate the broad
  candidate classes, but repeat evidence already consumed by the frozen line
  classifier.  Adding them at block level would violate the one-job ownership
  rule unless a future ablation proves independent structural information.

### Strong anchor fraction (removed from the learned gate)

- **Question:** What share of the component consists of independently strong,
  normal-length entry lines?
- **Representation:** Strong-anchor count divided by component line count,
  using the frozen `0.70` probability and `380`-character start limits.
- **Historical expected direction:** Positive.
- **Reason for density instead of count:** A 300-line prose region can contain
  more citation-like lines than a short true bibliography.  Density asks
  whether bibliographic support repeats throughout the proposed region.
- **Why removed:** It overlapped median entry probability.  After adding the
  independent deterministic-role fraction, one of five work-level folds gave
  anchor fraction a small negative coefficient while median probability
  remained strongly positive.  The gate now keeps the continuous median and
  removes the redundant thresholded summary.  Strong anchors still retain
  their separate hard job in sequence proposal: a long line cannot start a
  block.

### Median entry probability

- **Question:** How bibliography-like is the typical line in the component?
- **Representation:** Median frozen line-model probability.
- **Expected direction:** Positive.
- **Non-overlap:** This is the gate's only aggregate positive line-score
  summary.  It measures typical internal evidence rather than extent,
  continuity, or section scope.

### Longest weak run fraction

- **Question:** Does the proposed component contain a long uninterrupted hole
  of prose-like lines?
- **Representation:** Longest consecutive run below the frozen `0.25` inside
  probability, divided by component line count.
- **Expected direction:** Negative.
- **Non-overlap:** This measures internal continuity, not total size, positive
  evidence, or any citation token.

### Exact header at or before the component start

- **Question:** Is an exact multilingual bibliography heading on the first
  component line or within the two physical lines immediately before it?
- **Representation:** One binary value.
- **Expected direction:** Positive.
- **Safety rule:** It is supporting evidence in a multifeature component model;
  H0 still cannot create a deletion without an entry-line proposal.
- **Non-overlap:** This is section structure and uses no author, date,
  identifier, publication-coordinate, or page detector.

### Explicit negative-role fraction

- **Question:** How much of the proposed region is explicitly recognizable as
  a different document role rather than a bibliography entry?
- **Representation:** Share of component lines assigned exactly one frozen
  deterministic negative role: figure caption, table/equation, negative
  section heading, footnote, running/enumerated prose, or legal procedure.
- **Expected direction:** Negative.
- **Why it is not a hard veto:** The train role audit marked 30.4% of silver
  bibliography lines, mostly because real bibliography regions can contain
  tables, prose-like entries, and structural lines.  The learned gate therefore
  uses the *fraction* as graded contradictory evidence rather than deleting any
  component containing one marked line.
- **Train-OOF justification:** At the 0.5 component operating point, accepted
  true components had median negative-role fraction 0.215, versus 0.667 for
  accepted false components.  This separation appeared for both the logistic
  and shallow monotonic-tree gates.
- **Non-overlap:** This measures the presence of an explicit competing line
  role.  It is not the absence of positive citation evidence, component size,
  internal weak-run continuity, or heading support.

### Exact negative scope at or before the component start (not retained)

- **Question:** Does an explicit non-bibliography section begin where the
  proposed component begins?
- **Representation:** One binary value for an exact negative-scope heading on
  the first candidate line or within the preceding two physical lines.  Scope
  includes notes, CV/publications, ordinary body chapters, abbreviations,
  figure/table lists, and related-material sections.
- **Expected direction:** Negative.
- **Generic-heading safeguard:** An unknown Markdown heading does not count.
  This matters because genuine bibliography regions contain unenumerated
  subheadings such as source types and language divisions.
- **Train-OOF justification:** The cue occurred at 0.96% of accepted true
  component starts, 23.0% of accepted false starts, and 58.5% of false starts
  in silver-zero documents.
- **Non-overlap:** Negative-role fraction describes the composition of the
  whole component.  This feature describes the section boundary and scope.
- **Why rejected:** Despite strong aggregate separation, its learned
  coefficient reversed sign in one of five held-out work folds and the OOF
  precision/recall frontier did not improve.  Exact/generic heading roles stay
  materialized for diagnostics, but this cue is not an input to the selected
  gate.

### Component supervision purity

- **Question:** Is accepting this particular candidate mostly a correct BIB
  removal, regardless of how large the complete gold block is?
- **Positive:** At least 80% of the candidate's lines are silver BIB.
- **Negative:** At most 20% are silver BIB.
- **Masked:** Mixed 20%–80% boundary candidates are scored during OOF decoding
  but are not forced into either fitting class.
- **Reason:** Span IoU was the wrong target.  A 20-line component entirely
  inside a 200-line bibliography has low IoU with the whole block, despite
  every proposed removal line being correct.  Purity directly represents the
  component acceptance decision.
- **Not a feature:** Gold purity is training supervision only and is never
  available to the fitted classifier.

### Proposal thresholds and model threshold

- **Permissive deletion biases:** Calibration controls used to generate
  candidate spans.  They are not features and are not shown to the component
  classifier.
- **Component probability threshold:** A calibration control swept only on
  grouped training OOF scores.  It is not evidence and cannot change the five
  feature meanings.

## Core-only block expansion

The learned component gate is deliberately conservative: at the frozen safety
threshold it may recognize only the strongest portion of a long bibliography.
The core-expansion experiment tests whether weaker neighbouring proposals can
recover the rest without permitting new isolated deletions.

### Core threshold

- **Question:** Is this proposed component strong enough to establish that a
  bibliography block exists here?
- **Representation:** A high threshold on the grouped train-OOF component-gate
  score.
- **Safety job:** Only a component above this threshold may start a deletion.
- **Not evidence:** This is a calibration control over the five component
  features above; it adds no line detector or document cue.

### Expansion threshold

- **Question:** Once a strong core exists, which weaker proposed components are
  connected closely enough to belong to the same region?
- **Representation:** A lower threshold on the same component-gate score.
- **Coherence rule:** A lower-scored proposal is retained only when an
  overlap-connected chain joins it to a core.  A disconnected proposal can
  never start a block, regardless of its line features.
- **Why this matches the target:** Long and weak bibliography lines can be kept
  inside a region established by repeated strong entries, while an isolated
  citation-like line in prose remains outside.
- **Non-overlap:** This is a graph-connectivity decision over already-scored
  spans.  It does not count authors, dates, pages, identifiers, length, or
  headings again.

### Train-OOF result

The first overlap-chain implementation was rejected.  At equal core and
expansion thresholds, overlapping alternate proposals could already extend a
selected core; for the stable logistic gate this added 2,329 lines while
raising line recall only from 58.35% to 58.75% and reducing precision from
99.28% to 97.14%.  Lower thresholds caused larger spillover.  No expanded
candidate improved on the stable gate while satisfying the frozen safety rule.
The failed experiment remains archived as `component_expansion_r1`; it is not
a candidate for validation or corpus cleaning.

## Current experiment boundary

The anchored-coherence, learned-component, and core-expansion experiments
change no deterministic line regex and add no raw text, token length, source
identity, or validation-derived feature.  They reuse frozen OOF entry
probabilities and B1 checkpoints.  Candidate selection is based on grouped
train documents only.  Validation remains closed until a complete safe
configuration is frozen.

## Exact auxiliary-scope veto

The false-component review found that all 23 spurious blocks at the permissive
0.95 role-gate threshold came from one silver-zero OpenArchives document.  They
were repeated source lists under exact `ΣΧΕΤΙΖΟΜΕΝΑ ΧΝΑΡΙΑ`, `WHY`,
`EXAMPLES`, or selected-variant archive headings, rather than ordinary prose.

The retained deterministic rule is deliberately narrow:

- it recognizes only the pre-existing exact auxiliary headings, exact
  Greek/English `WHY` and `EXAMPLES` forms (including the observed OCR delta),
  and the exact selected-variants archive prefix;
- numbered heading prefixes are normalized, but generic or fuzzy headings are
  never promoted;
- ordinary ATX scope ends at the next ATX heading; a selected-variants archive
  persists only through exact AT/ATU type subheadings; and
- it may veto a proposed component that contains the scope or begins within
  the preceding two physical lines.  It cannot create or expand a deletion.

This scope covers 12,292 training lines and overlaps zero silver-BIB lines.
After the veto, the safe role-aware gate at threshold 0.90 reaches 99.17% line
precision / 80.47% line recall and 99.16% token precision / 85.40% token
recall, with 0.0150 spurious blocks per silver-zero document.  This is the
current retained train-only point in `auxiliary_scope_veto_r6`.

### Abbreviations are not a section veto

The first signal validation attempt stopped before metric evaluation because
one document placed 971 silver bibliography lines under the exact heading
`ΣΥΝΤΟΜΟΓΡΑΦΙΕΣ`.  Those lines expand short citation keys into complete
references.  This exposes a semantic mistake in the original scope rule:
“abbreviations” describes the **format** of the following items, not whether
the items are bibliographic.

The corrected ownership is perpendicular:

- the abbreviations heading itself remains an explicit structural/non-entry
  line role;
- `abbreviations`, `list of abbreviations`, `συντομογραφίες`, and `κατάλογος
  συντομογραφιών` can no longer establish a hard negative section scope; and
- the block model must decide from repeated citation evidence below the
  heading.  No document identity, source identity, position, or fuzzy heading
  match is used.

This is a generic rule change rather than an exception for the observed
document.  The grouped train-OOF decoder must be rerun before the corrected
veto is evaluated again.

### Exact negative scope is a wall, not a poison pill

The next validation diagnostic exposed a separate block-decoder error.  A
high-confidence bibliography ended immediately before an exact `List of
Figures`/`List of Tables` region, and another publication list began
immediately after it.  Boundary expansion touched one or two scoped lines.
The old all-or-nothing veto then discarded both otherwise coherent
bibliography blocks.

The corrected block rule is:

- an exact negative scope is a **hard wall**;
- the anchor decoder runs independently on each contiguous non-scope segment,
  so anchors and weak-line bridges can never cross the wall;
- scope lines themselves can never be emitted, including by downstream header
  attachment; and
- a coherent block on either side survives.  Merely touching a negative scope
  no longer poisons the entire block.

This changes no line feature or threshold.  It makes the deterministic scope
cue perpendicular to bibliography evidence: the cue owns only its explicitly
scoped region, while repeated citation evidence owns neighbouring regions.
The grouped train-OOF grid must again be rerun before validation.

### Frozen anchors and low-confidence islands

The recall-first decoder chosen on grouped train OOF uses a contextual score
of 0.30 as an anchor.  At least two anchors must occur within 16 emitted lines
before a block can exist.  Once they do, weak lines can be bridged over an
eight-line gap and two weak boundary lines can be attached when their score is
at least 0.05.  These numbers are decoder settings selected on train OOF, not
new line features.

After the scope-wall correction, every large remaining validation miss had
fewer than two 0.30 anchors in the required window.  Examples include long
lists of legal memoranda, ancient authors and works, archival sources, and
author publications.  That diagnosis does **not** justify lowering 0.30 on
validation: doing so could admit isolated citation-like prose and would tune
against the held-out result.  A future change must be proposed and selected on
train/source-held-out evidence, then evaluated on fresh documents.  Sparse
two-line reading lists embedded in textbook chapters remain outside the
large-block target even when each line looks bibliographic.

## Rich component gate (rejected)

A shallow monotonic tree and an unconstrained logistic model tested ten
additional component summaries: entry-probability q10/q90, minimum boundary
probability, and the eight individual competing-role fractions.  The safe
monotonic-tree point regressed to 99.29% line precision / 56.38% line recall.
The logistic arm could reach 95.07% precision / 89.19% recall, but violated the
predeclared direction contract in every fold: q10 and minimum boundary
probability repeatedly took negative weights, and two negative heading roles
also reversed in one fold.  The rich gate is therefore archived as
`rich_component_gate_r1` and rejected; it is not eligible for validation.

## Signal-only line TCN (active experiment)

The next architecture learns block membership directly over a 31-line
neighbourhood.  Its input per line is intentionally restricted to:

1. the frozen out-of-fold entry probability;
2. the eight mutually exclusive deterministic competing roles; and
3. the exact bibliography-header flag.

It receives no text, line length, document position, source identity, or
validation data.  Five document-grouped models make out-of-fold predictions,
so a model never scores a document on which it fitted.  A small residual TCN
(32 hidden units, dilations 1/2/4/8, dropout 0.10) predicts entry membership;
the exact auxiliary-scope veto then removes scoped components and H0 may attach
an exact header only after a component already exists.  The same 99% line
precision and 0.02 spurious-block safety gate applies before this candidate can
be retained.

The first plain per-line threshold evaluation is not retained.  At threshold
0.40 it reached 95.05% line precision / 93.11% line recall and 94.98% token
precision / 96.10% token recall, proving that the contextual ranking is useful.
But independent line thresholding fragmented true regions and left 0.436
spurious blocks per silver-zero document.  The only configuration satisfying
the strict safety gate was threshold 0.999, which emitted no lines.

The registered second-level decoder therefore requires two or three high-score
anchors in a bounded window before a block can exist.  Weak or long lines may
then be filled only between those anchors or immediately beside the established
region.  Line length remains absent, an isolated high-scoring line cannot start
a deletion, exact scope can only veto, and H0 remains downstream.  This is the
active `signal_blocks_r1` experiment.

The anchored grid did not beat the retained exact-scope candidate.  Its best
strict-safe nonempty point reached 99.71% line precision but only 19.38% line
recall.  Its useful diagnostic point reached 98.13% precision / 86.46% recall
and 98.44% token precision / 91.10% token recall, but raw silver counted 0.0451
spurious blocks per zero-BIB document.

A subsequent barrier experiment tested exact/generic headings, figure captions,
footnotes, and sustained runs of very-low TCN probability as split points.
Stable-negative barriers could not create or expand a deletion.  They did not
improve the high-recall frontier and produced no strict-safe candidate.  The
best line-precision-above-0.99 point was 99.02% precision / 78.58% recall, while
the best high-recall point remained effectively unchanged at 98.16% / 86.46%.
The barrier arm is rejected.

The separately registered high-recall anchor grid then tested whether the
same block rule had simply been anchored too conservatively.  This changed no
line feature and did not inspect validation.  Its best train-OOF point above
90% line precision uses the following plain-language rule:

- **Block evidence:** at least two lines with contextual probability 0.30 or
  greater must occur within a 16-line physical neighbourhood.  One line can
  never establish a block.
- **Interior continuity:** after those anchors establish the region, lines
  scoring at least 0.10 may bridge a gap of no more than eight physical lines.
- **Boundary allowance:** at most two adjacent lines may be attached at each
  edge.  This is a boundary repair, not independent line evidence.
- **Exact scope veto and header attachment:** the audited non-bibliography
  scope can only remove a proposed block, and an exact bibliography header is
  attached only after the block exists.

This point reaches 91.75% line precision / 94.29% line recall and 91.43%
token precision / 96.86% token recall.  The more conservative train-OOF point
above 95% line precision reaches 95.04% precision / 92.78% recall and 94.89%
token precision / 95.86% recall.  Neither satisfies the original 99% raw-silver
safety gate; both remain diagnostic candidates.  Their complete 288-setting
grid is archived as `signal_recall_blocks_r1`.  The thresholds and decoder
settings were frozen before signal-TCN validation.

Manual inspection explains the apparent safety failure.  Of the 50 components
with the most silver-non-BIB tokens, 12 are clear silver omissions, 21 are real
bibliography blocks with small boundary overruns, 9 are genuine whole-block
errors, 7 are policy-sensitive structured lists, and 1 has extraction/lineage
corruption.  In particular, five of six apparent blocks in silver-zero
documents are genuine numbered annotated bibliographies in one Kallipos book.
The case register and immutable packet are documented in
`BIB_SIGNAL_FALSE50_REVIEW_20260714.md`.  Raw-silver metrics remain necessary
for comparability, but further tuning against them alone would teach the model
to suppress correct bibliography detections.

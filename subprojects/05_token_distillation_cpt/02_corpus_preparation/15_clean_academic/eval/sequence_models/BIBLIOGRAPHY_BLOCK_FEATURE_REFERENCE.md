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

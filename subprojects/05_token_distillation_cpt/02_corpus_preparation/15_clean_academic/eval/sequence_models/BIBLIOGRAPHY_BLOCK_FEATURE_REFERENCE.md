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

### Component line count

- **Question:** How large is the proposed contiguous region?
- **Representation:** `log1p` of emitted line count.
- **Expected direction:** Positive.  A large region is more consistent with
  the cleaning target than one isolated citation.
- **Non-overlap:** This measures extent only; it does not inspect line content.

### Strong anchor count

- **Question:** How many independently strong, normal-length entry lines occur
  inside the component?
- **Representation:** `log1p` of the count, using the frozen `0.70` probability
  and `380`-character start limits.
- **Expected direction:** Positive.
- **Non-overlap:** Unlike component line count, this measures repeated
  bibliographic support rather than size.

### Median entry probability

- **Question:** How bibliography-like is the typical line in the component?
- **Representation:** Median frozen line-model probability.
- **Expected direction:** Positive.
- **Non-overlap:** A strong-anchor count measures how much strong evidence is
  present; the median measures whether the component as a whole is supported
  rather than being carried by a few outliers.

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

## Current experiment boundary

The anchored-coherence and learned-component experiments change no
deterministic line regex and add no raw text, token length, source identity, or
validation-derived feature.  They reuse frozen OOF entry probabilities and B1
checkpoints.  Candidate selection is based on grouped train documents only.
Validation remains closed until a complete safe configuration is frozen.

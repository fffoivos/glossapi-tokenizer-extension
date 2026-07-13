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

## Current experiment boundary

The anchored-coherence sweep changes no deterministic line regex and adds no
text, token length, or validation-derived feature.  It reuses frozen OOF entry
probabilities and existing B1 checkpoints, varies proposal bias, and applies
only the two distinct component gates above.  The sequence-ablation sweep then
re-trains B1 with the length and position removals described above.  Candidate
selection is based on the 1,118 grouped train documents.  Validation remains
closed until a complete configuration is frozen.

# Deferred review: post-mask deduplication

Status: **deferred by the user on 2026-08-09**.

No current training job, Slurm allocation, dataset artifact, packing schedule,
optimizer state, or learning-rate schedule is authorized to change because of
this note.

## Issue to revisit

The anonymized full-8B dataset pipeline applied a second global exact-content
deduplication after PII masking. This was outside the requested anonymization
scope. The completed receipt reports:

- 2,378,595 documents dropped as `postmask_exact_duplicate`;
- 8,081 documents dropped for validation-content collisions;
- 2,386,676 total dropped documents.

The current evidence does not separate duplicates newly created by PII
masking from duplicates that already existed in the frozen v2 row multiset.

## Candidate correction for later review

When this issue is resumed, evaluate restoring the rows marked only
`postmask_exact_duplicate` while:

- retaining their masked text and original row identities/multiplicity;
- continuing to exclude validation-content collisions;
- preserving the approved GreekMMLU decontamination and explicit eligibility
  exclusions;
- rebuilding token counts, packed data and the D0 schedule from receipts;
- keeping the proven 16-node DP32 execution geometry and approved WSD-10
  learning-rate policy;
- never splicing restored rows into the middle or tail of an existing
  trajectory;
- requiring explicit user approval before any rebuild, cancellation, restart,
  allocation change, or production submission.

This is a record of an open decision, not an implementation plan or authority
to act.

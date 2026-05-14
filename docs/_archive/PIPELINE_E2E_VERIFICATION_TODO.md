# Pipeline E2E Verification TODO

## Phase 1: Canonical Stage Chain

- [x] Freeze the exact repo-owned command or wrapper for each live stage:
  - dedup completion
  - overlay publish
  - tokenizer mix build
  - tokenizer training
  - uploader handoff prep
  - uploader launch or uploader-ready handoff
- [x] Record the exact required input artifact for each stage.
- [x] Record the exact completion artifact for each stage.
- [x] Record the exact first-progress signal for each stage.
- [x] Record the stall threshold for each stage.
- [x] Record the restart rule for each stage.
- [x] Replace any stale or dead script paths in the live chain.

## Phase 2: True End-to-End Worker Validation

- [x] Start from a dedup-complete state on the real worker.
- [x] Run the real overlay publish step.
- [x] Verify the overlay process exists and the overlay artifact appears.
- [x] Run the real tokenizer mix build step.
- [x] Verify the mix process exists and the first mix artifact appears.
- [x] Verify mix completion summary exists.
- [x] Run the real tokenizer training step.
- [x] Verify tokenizer training starts from the produced mix artifact.
- [x] Verify tokenizer training completion summary exists.
- [x] Run the real uploader handoff prep step.
- [x] Verify handoff manifest exists.
- [x] Run uploader launch or verify uploader-ready handoff state.
- [x] Produce a single worker-side run report with timestamps and artifact paths.

## Phase 3: Remaining Stage Parallelism Audit

- [x] Review overlay publish code for long serial sections.
- [x] Review tokenizer mix build code for long serial sections.
- [x] Review tokenizer training code for underused parallelism or oversized finalization.
- [x] Review uploader handoff prep for unnecessary serialization.
- [x] Review uploader launch for single-process bottlenecks.
- [x] Check configured worker count versus actual live worker usage for each stage.
- [x] Check whether any stage has a long serial prelude before workers start.
- [x] Check whether any stage has a large hidden finalization tail.
- [x] Write a short finding for each stage:
  - acceptable
  - needs instrumentation
  - needs parallelization
  - needs redesign

## Phase 4: Progress Transparency Audit

- [x] For overlay publish, record:
  - progress file
  - trace file
  - source of truth
  - completion marker
- [x] For tokenizer mix build, record:
  - progress file
  - trace file
  - source of truth
  - completion marker
- [x] For tokenizer training, record:
  - progress file
  - trace file
  - source of truth
  - completion marker
- [x] For uploader handoff prep, record:
  - progress file
  - trace file
  - source of truth
  - completion marker
- [x] For uploader launch, record:
  - progress file
  - trace file
  - source of truth
  - completion marker
- [x] Verify each stage emits progress early enough to detect real work.
- [x] Verify each stage exposes enough information to distinguish:
  - running
  - stalled
  - completed
- [x] Verify reported progress matches the true source of truth.

## Phase 5: Verification Output

- [x] Write the canonical stage-chain table.
- [x] Write the worker-side end-to-end verification report.
- [x] Write the remaining-stage serial/parallel findings.
- [x] Write the remaining-stage progress-transparency findings.
- [x] Turn the findings into a prioritized improvement list.

## Guardrail

- [x] Do not call the pipeline “end-to-end verified” again unless the real worker-side chain has been exercised from dedup completion through tokenizer launch with real artifacts at every handoff.

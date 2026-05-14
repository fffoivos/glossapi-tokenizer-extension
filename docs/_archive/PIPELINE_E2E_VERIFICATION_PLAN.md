# Pipeline E2E Verification Plan

## Goal

Prevent a repeat of the recent failure mode where:
- individual stages or contracts looked healthy in isolation
- but the real worker-side orchestration path was not actually validated end to end
- so the live chain stopped after dedup and tokenizer work never started

This plan defines how the pipeline will be checked for:
- true end-to-end operability
- remaining slow or non-parallelized stage sections
- transparent and trustworthy progress reporting at every remaining stage

## Scope

The verification scope starts after dedup input preparation and covers the real worker-side chain:

1. dedup completion
2. dedup overlay publish
3. tokenizer mix build
4. tokenizer training
5. uploader handoff prep
6. uploader launch or uploader-ready handoff

## Rule For “Verified”

Nothing counts as verified unless it has been checked on the real worker path with:
- the real repo-owned command or wrapper
- the real expected input artifact from the previous stage
- the real expected output artifact for the next stage
- a live progress signal that advances during execution

Component tests and contract tests are necessary, but they do not count as a true end-to-end verification by themselves.

## Part A: True End-to-End Validation

### Objective

Prove that the actual chained orchestration path works from dedup completion through tokenizer launch, without manual intervention or dead waiting shells.

### Method

1. Define one canonical stage chain.
2. For each stage, record:
- exact command or wrapper script
- required input artifact
- first durable progress signal
- completion artifact
- timeout or stall threshold
- restart rule
3. Run the real chain on the worker.
4. Verify every handoff with artifacts, not assumptions.
5. Produce one run report with timestamps, commands, and outputs.

### Required Per-Stage Proof

Every stage must have all three:
- running proof: process exists and log or trace is advancing
- first durable progress proof: first real output file appears
- completion proof: expected summary, manifest, or final artifact exists and is readable

### Canonical Handoff Checks

The verification must prove these transitions:

1. dedup success marker appears
2. overlay publish starts from the repo-owned script path
3. overlay artifact appears where mix build expects it
4. mix build starts and emits its first mix artifact
5. mix completion artifact appears
6. tokenizer training starts from the mix output
7. tokenizer training completion summary appears
8. uploader handoff starts from the real training and release inputs
9. uploader handoff manifest appears
10. uploader launch starts or explicit uploader-ready handoff is materialized

### Pass Criteria

The chain passes only if:
- no stage is blocked behind a dead script path
- no stage is blocked forever by a missing artifact that should already exist
- no stage relies on a stale detached shell that is waiting on the wrong file
- every transition is backed by a real artifact and a real process

## Part B: Remaining Serial / Non-Parallelized Stage Review

### Objective

Identify every remaining stage section that is:
- unexpectedly serial
- weakly parallelized
- apparently parallel but actually bottlenecked on one hot parent
- dominated by an oversized finalization tail

### Method

Use both static review and live runtime review.

### Static Review

For each remaining stage, inspect the code for:
- long serial preludes before worker pools start
- full-table or full-file materialization
- large in-memory regrouping before chunk registration
- single-process final combine steps
- final export steps that write huge files with no intermediate checkpoint
- scheduler paths that underfeed workers

### Live Runtime Review

For each stage, check:
- configured worker count
- actual active worker count
- parent CPU vs worker CPU
- worker I/O wait state
- chunk completion rate
- first-checkpoint latency
- whether progress stalls while CPU or I/O remains active

### Remaining Stages To Review

1. dedup overlay publish
2. tokenizer mix build
3. tokenizer training
4. uploader handoff prep
5. uploader launch

### Pass Criteria

A stage passes the parallelism review only if:
- configured parallelism roughly matches real live usage, or the stage is intentionally serial for a documented reason
- there is no hidden long serial prelude without explicit progress reporting
- there is no hidden large finalization tail without explicit explanation and observability
- the stage can justify any one-core-hot parent bottleneck

## Part C: Progress Transparency Review

### Objective

Make every remaining stage externally readable enough that a watcher can answer:
- is it running
- is it progressing
- is it stalled
- is it complete

without guessing from stale files or process names alone.

### Method

For each stage, define and verify:
- progress file path
- trace file path
- source of truth for progress
- first visible progress signal
- stall threshold
- completion marker

### Progress Requirements

Each remaining stage should expose:
- a progress file early in stage startup
- a `phase` field if there is a prelude
- a trustworthy `total_units` and `completed_units` when chunked
- a trace log for milestone transitions
- final artifact paths in the completion payload

### Required Checks

For each stage:
- confirm the progress file appears near stage start
- confirm the progress file updates before stage end
- confirm the trace log explains major phase transitions
- confirm the reported progress matches the true source of truth
- confirm stall detection is possible from artifacts or DB state

### Pass Criteria

A stage fails transparency review if:
- progress is invisible until the very end
- progress files are stale while the DB or artifacts tell a different story
- there is no way to distinguish active work from a dead wait
- there is no explicit marker for stage completion

## Deliverables

This plan is complete only when it produces:

1. a canonical stage-chain table with exact commands and expected artifacts
2. an end-to-end run report from the real worker path
3. a remaining-stage serial/parallel review sheet
4. a remaining-stage progress-contract review sheet
5. a prioritized fix list for whatever fails the checks

## Exit Criteria

The verification effort is complete only when:
- the real chain runs end to end from dedup completion through tokenizer launch
- every remaining stage has been reviewed for hidden serial bottlenecks
- every remaining stage has explicit and trustworthy progress reporting
- the repo has a single canonical checklist for repeating the verification later

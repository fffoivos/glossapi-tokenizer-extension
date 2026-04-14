# Near Dedup Redesign Plan

## Purpose

This file is the concrete plan for redesigning the near-dedup path.

It exists because the current near-candidate implementation is not scaling well enough on the live corpus.

This plan must be enriched as decisions are made.

## Non-Negotiable Constraints

### 1. Functional Semantics

We must preserve the intended dedup behavior, except for explicit semantic changes that are deliberately chosen and tested.

Currently agreed semantic change:
- remove the hard near-dedup `length_ratio` admission gate

Everything else should be treated as preserved unless explicitly changed.

### 2. Efficiency

The redesigned near-dedup path must materially improve:
- memory profile
- throughput
- progress granularity
- operational stability

The redesign is not successful if it only preserves correctness while still saturating memory before making durable progress.

### 3. Resumability

Resumability is a core requirement.

We must preserve resumability at sane stage boundaries:
- `stage_01_exact` completed
- `near_signatures` completed
- `near_candidates` completed
- `near_clusters` completed
- `final` completed

Important clarification:
- we do not need to preserve every broken intermediate artifact format
- we do need to preserve resumability from the last stable stage boundary

So if `near_candidates` is redesigned, the safe resume boundary is:
- keep everything through `near_signatures`
- rebuild `near_candidates` and later stages

## Current Diagnosed Problems

### Semantic

- the old hard near `length_ratio` gate caused obvious false negatives
- this has now been removed in the repo-backed code

### Execution

- current near-candidate work is too coarse
  - one whole band is too large a unit of work
- worker memory usage is too high
  - per-worker state is duplicated too heavily
- progress visibility is too poor
  - progress stays at `0 / 32` until a whole band finishes
- the live run showed that `16` workers on `976 GB` still reached about `955 / 960 GB`

## Reference Architecture

The redesign should borrow from the execution pattern used by Hugging Face DataTrove MinHash:
- staged file-based processing
- sorted signature shards
- streaming / merge-style candidate generation
- separate clustering and filtering stages

Reference:
- [HF_DEDUP_INVESTIGATION.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/HF_DEDUP_INVESTIGATION.md)

This is not a drop-in replacement.
We are borrowing the execution pattern, not replacing our semantics wholesale.

## Concrete Redesign Goals

### Goal A. Replace the near-candidate data plane

Current issue:
- workers duplicate too much in-memory signature state

Target:
- compact on-disk or memory-mapped signature shards
- streaming reads
- no full Python signature map per worker

### Goal B. Make work units smaller

Current issue:
- band-level work units are too coarse

Target:
- partition by band plus hash range, or equivalent sub-band chunking
- make each chunk small enough to:
  - complete sooner
  - checkpoint sooner
  - fail/retry cheaply

### Goal C. Separate generation from aggregation

Current issue:
- too much work happens before any durable completion is recorded

Target:
- each chunk writes candidate-pair shard outputs
- a later aggregation step combines them

### Goal D. Improve observability

Current issue:
- `0 / 32` is too coarse to understand live progress

Target:
- per-chunk progress markers
- per-chunk row counts
- per-chunk memory summaries
- early visible progress before full stage completion

### Goal E. Preserve stage-aware resume

Current issue:
- resume correctness has already failed once at stage handoff

Target:
- if a stage summary exists and artifacts are valid, resume must not re-enter the stage
- each redesigned substage must have a durable completion marker

## Tests Required

### A. Small correctness tests

Purpose:
- prove the redesigned logic still works on small inputs

Required:
- candidate generation on synthetic duplicate and non-duplicate pairs
- cluster resolution tests
- keeper-choice tests
- explicit test for short-vs-long high-similarity pair

### B. Contract tests

Purpose:
- prove stage outputs still match next-stage expected inputs

Required:
- near-signatures -> near-candidates
- near-candidates -> near-clusters
- near-clusters -> final exports
- final outputs -> overlay / builder metadata consumers

### C. Resumability tests

Purpose:
- prove stable resume boundaries remain valid

Required:
- resume from completed exact stage into near
- resume from completed near-signatures into near-candidates
- resume from partially completed near-candidates
- resume from partially completed near-clusters
- never re-enter a completed stage if its outputs are valid

### D. End-to-end tests

Purpose:
- prove the pipeline still runs across all affected stages

Required:
- tiny real-doc end-to-end smoke
- medium-scale end-to-end run through:
  - exact
  - near-signatures
  - near-candidates
  - near-clusters
  - final dedup outputs
  - overlay
  - mix build
  - tiny training launch

### E. Efficiency tests

Purpose:
- prove the redesign is actually better under strain

Required metrics:
- peak total RSS
- peak per-worker RSS
- first chunk completion time
- completed chunks per hour
- total wall time
- stability under no-swap conditions

Required benchmark layers:

1. medium-scale replay benchmark
- same sampled real-doc corpus
- old vs new near-candidate path
- compare semantics and performance

2. worker-count sweep
- `8`
- `10`
- `12`
- `14`
- `16`

3. large-box stress test
- worker-class machine
- real-ish larger subset
- must remain below a defined memory ceiling

## Efficiency Acceptance Criteria

These should be refined as we gain measurements, but the redesign should aim for:

- first durable chunk completion well before current whole-band timing
- materially lower peak memory than the current `16`-worker attempt
- no guest-access instability under the selected operating point
- enough headroom that `available` memory does not approach zero during normal operation

## Resume Policy

Resume should work like this:

1. If `stage_01_exact` is complete:
- reuse it
- do not rerun it

2. If `near_signatures` is complete:
- reuse them
- do not rebuild them

3. If the near-candidate artifact layout changes:
- invalidate only `near_candidates` and later
- keep exact and near-signatures

4. If `near_candidates` completes under the new format:
- later reruns must resume from there

## Immediate Next Steps

1. redesign `near_candidates` to eliminate full-worker signature-map duplication
2. introduce smaller candidate-generation chunks
3. add per-chunk progress markers
4. run small correctness + resumability tests
5. run medium-scale efficiency benchmark
6. choose a stable worker count
7. only then restart the full live pipeline
8. once the redesigned near-dedup path is validated, fold it back into the canonical tokenizer pipeline and core project plan as the standard dedup path

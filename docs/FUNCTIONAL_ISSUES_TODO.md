# Functional Issues TODO

This file tracks functional issues that should be improved without changing the intended high-level pipeline goal.

## Semantics

- Remove the hard near-dedup `length_ratio` admission gate.
  - Status: done in repo branch `repo-canonicalization-20260414`.
  - Reason: it caused obvious false negatives by preserving highly similar short-vs-long duplicates.

- Re-review near-dedup admission semantics after the next stable run.
  - Check whether any other early hard gate is blocking plausible duplicates before representative selection.
  - Keep the similarity threshold unchanged until the post-length-gate behavior is inspected.

## Near-Dedup Execution

- Replace the current `near_candidates` memory-heavy worker state with a streaming / external-merge design.
  - Current issue: each worker duplicates too much signature state in memory.
  - Reference: [HF_DEDUP_INVESTIGATION.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/HF_DEDUP_INVESTIGATION.md)

- Repartition near-candidate work below the current whole-band granularity.
  - Current issue: progress stays at `0 / 32` until a full band completes.
  - Target: band-plus-hash-range or similarly smaller chunks.

- Make near-candidate outputs shard-first and aggregate later.
  - Current issue: too much heavy work happens before any durable chunk completion is recorded.

- Improve near-candidate observability.
  - Add per-chunk progress markers.
  - Add per-chunk memory summaries.
  - Add candidate-pair row counts before full stage completion.

## Resumability / Orchestration

- Keep resume stage-aware at every stage boundary.
  - Avoid any regression where a completed stage is rerun because the progress marker was overwritten.

- Keep watcher scripts repo-rooted and argument-complete.
  - Current issue already seen: watchers launched with missing required args fail silently except for `usage:` logs.

- Keep worker boot/rearm flows simple.
  - Avoid overbroad `pkill` or brittle startup-script bootstrap logic.

## Throughput Validation

- Add a medium-scale near-candidate stress test.
  - Goal: measure memory growth by worker count on real-ish data, not only tiny smoke data.

- Add a throughput acceptance target for near-candidates.
  - Example dimensions:
    - first chunk completion time
    - max RSS per worker
    - total machine memory ceiling
    - completed chunks per hour

- Add a worker-count sweep harness for near-candidates.
  - Suggested values:
    - `8`
    - `10`
    - `12`
    - `14`
    - `16`
  - Use this to choose the stable operating point rather than guessing live.

## Pipeline Validation

- Re-run the full small real-doc end-to-end smoke after each near-dedup semantic or execution change.

- Add one medium-scale end-to-end pipeline run before the next full live rerun.
  - Goal: prove the current repo-backed path through near-dedup, overlay, mix build, training launch, and uploader handoff under moderate strain.

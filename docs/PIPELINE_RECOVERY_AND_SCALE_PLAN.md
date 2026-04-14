# Pipeline Recovery And Scale Plan

## Purpose

Recover the current end-to-end pipeline without losing the expensive work already completed, then upgrade the dedup path so the same failure mode does not recur on larger corpus snapshots.

## Current Diagnosis

- HPLT `clean60` build is complete.
- HPLT integration into the working source release is complete.
- The live dedup run is not in near dedup.
- The live dedup run is still in `stage_01_exact`, specifically the `relaxed_exact` finalization/export path.
- Row-group chunk computation is already complete.
- The pathological behavior is the SQLite-heavy exact-stage export/finalization tail, not the parallel chunk phase.

## Guardrails

These must not change during the upgrade:

- golden rule:
  - functionality must remain the same
  - efficiency may change
  - if a change improves speed but changes dedup decisions, it is rejected
- exact hashing semantics:
  - `exact_strict_hash`
  - `exact_relaxed_hash`
- survivor semantics:
  - strict exact first
  - relaxed exact only over strict survivors
- keeper selection semantics:
  - `selection_priority_tuple(...)`
- downstream contract:
  - same builder metadata meaning
  - same dedup decision meaning
  - same mixed-source handling
- tokenizer data contract:
  - published source parquet remains physically undeduplicated
  - downstream builder applies dedup using refreshed `dedup_metadata`

## Semantic Equivalence Rule

Allowed to change:

- storage layout
- intermediate artifact layout
- checkpoint structure
- export strategy
- degree of parallelism
- runtime and memory profile

Not allowed to change:

- exact keep/drop decisions
- near keep/drop decisions
- cluster membership meaning
- keeper selection behavior
- builder-visible dedup metadata meaning
- downstream tokenizer-corpus eligibility

## Integrated Plan

### Phase 1: Freeze And Snapshot The Current Run

1. Stop the live dedup process cleanly.
2. Preserve the whole exact-state snapshot before any migration:
   - `state.sqlite`
   - `state.sqlite-wal`
   - `state.sqlite-shm`
   - full run root `exact_stage_20260413T025237Z`
3. Record a run-forensics summary:
   - active stage
   - current outputs already written
   - open temp files
   - file sizes for SQLite, WAL, and run outputs
4. Do not touch the HPLT build or HPLT integration artifacts.

### Phase 2: Salvage What Already Exists

1. Reuse the finished row-group chunk work from the current state DB.
2. Reuse strict exact outputs if they validate:
   - `strict_exact_groups.parquet`
   - `strict_exact_drop_list.parquet`
3. Reuse strict exact membership already materialized in `run_exact_results` if counts and keys are coherent.
4. Treat current relaxed exact `.tmp` files as disposable.
5. Resume from exact-stage finalization, not from HPLT build or chunk hashing.

### Phase 3: Upgrade Dedup For Scale

1. Replace the exact-stage finalization path so it no longer uses SQLite as the heavy merge/export engine for the full run.
2. Keep SQLite as a control-plane and resumability store only:
   - run registry
   - stage progress
   - artifact registry
   - chunk status
3. Move heavy exact-stage membership and drop-list materialization to parquet shard outputs.
4. Make strict and relaxed exports partitionable and mergeable.
5. Make exact-stage output generation resumable at the shard level.
6. Ensure later stages consume stage-local parquet artifacts rather than forcing full replay into giant SQLite tables.

### Phase 4: Verify The Upgrade Before Resuming Live Work

1. Synthetic contract tests:
   - same exact keep/drop decisions as before
   - same output schemas
   - same stage markers
2. Golden exact-stage equivalence tests:
   - run old and repaired exact-stage implementations on the same tiny real-document dataset
   - require exact equality, after deterministic sort if needed, for:
     - `strict_exact_groups.parquet`
     - `strict_exact_drop_list.parquet`
     - `relaxed_exact_groups.parquet`
     - `relaxed_exact_drop_list.parquet`
     - `docs_exact.parquet`
     - `exact_survivor_manifest.parquet`
   - require exact equality for final exact keep/drop mapping per `doc_key`
3. Resume equivalence tests:
   - create a partial exact-stage run
   - resume it with the repaired code
   - compare outputs against a clean completed golden run
4. Downstream contract equivalence tests:
   - feed repaired exact-stage outputs into:
     - near dedup
     - final exports
     - builder metadata export
     - dedup overlay publish
     - mix build
   - require the same downstream meaning and compatible schemas
5. Tiny real-doc smoke:
   - exact strict
   - exact relaxed
   - near
   - builder metadata export
6. Medium-scale resume test:
   - enough rows to exercise shard merging and resumability
   - confirm no pathological WAL growth
7. Only then resume the live state.

### Phase 5: Resume The Current Live Run

1. Load the frozen live state snapshot.
2. Validate which strict artifacts can be trusted.
3. Drop only the invalid or incomplete relaxed-stage material.
4. Re-enter exact-stage finalization with the upgraded code.
5. Continue into:
   - near dedup
   - final exports
   - refreshed `dedup_metadata`
   - tokenizer mixes
   - `50k` discovery training
   - uploader handoff

### Phase 6: Improve Parallelism

1. Parallelize exact-stage export by partition rather than one global serial pass.
2. Use the worker CPUs for:
   - per-partition group materialization
   - per-partition drop-list writing
   - per-partition survivor export
3. Avoid one giant serial exact-stage merge before near dedup.
4. Keep the uploader on the cheap instance and keep training on the CPU worker.

## Expected Result

After this recovery:

- the current run is resumed rather than restarted from zero
- exact-stage export becomes resumable and partitioned
- SQLite remains small enough to function as control metadata instead of the main data plane
- near dedup can start in a predictable amount of time
- downstream tokenizer and uploader stages can proceed automatically once dedup completes

## Required Test Deliverables

Before the repaired path becomes the default live path, the repo must contain:

- a golden equivalence test for exact-stage outputs
- a resume equivalence test for exact-stage continuation
- a downstream contract equivalence test for builder metadata and mix consumption
- a tiny real-doc end-to-end smoke that exercises the repaired path

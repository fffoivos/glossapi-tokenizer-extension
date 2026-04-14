# Dedup Script Repair Plan

## Problem Statement

The current dedup implementation scales poorly in `stage_01_exact` finalization.

The expensive chunk phase completes, but the run then spends many hours in exact-stage export because:

- `build_stage_results(...)` rebuilds stage outputs through SQLite
- `run_exact_results` is deleted and repopulated for each exact stage
- the state DB WAL grows to hundreds of GB
- finalization is effectively single-process

This is a design problem, not just a tuning problem.

## Design Constraints To Preserve

- golden rule:
  - functionality stays the same
  - efficiency changes
  - semantics do not change
- exact strict and relaxed normalization semantics must remain unchanged
- `selection_priority_tuple(...)` must remain the keeper decision
- relaxed exact must still operate only on strict survivors
- output contracts must remain stable:
  - exact group parquet
  - exact drop-list parquet
  - docs-exact export
  - exact survivor manifest
  - later builder metadata semantics

## Repair Objectives

1. Keep exact results accurate.
2. Make exact-stage finalization resumable.
3. Remove the large SQLite write-amplification path.
4. Make exact export parallelizable.
5. Preserve downstream schema and meaning.

## What Must Stay Equivalent

The repaired path must preserve:

- exact group membership
- exact drop lists
- exact kept-doc mapping
- docs-exact export semantics
- exact survivor manifest semantics
- downstream builder metadata semantics

If parquet row ordering changes after parallelization, comparisons must normalize ordering by deterministic keys before equivalence checks.

## Target Architecture

### 1. Split Control Plane From Data Plane

Keep in SQLite only:

- run registry
- input snapshot registry
- chunk status
- stage status
- artifact registry

Do not use SQLite as the primary store for full exact group membership on large runs.

### 2. Materialize Exact Results As Sharded Parquet

For both strict and relaxed exact:

- partition by hash prefix or deterministic bucket
- write stage-local shard files:
  - group membership shards
  - drop-list shards
  - kept-doc shards if needed
- write a small manifest describing shard completeness

This turns exact-stage finalization into a shard job rather than one serial SQLite pass.

### 3. Rework Relaxed Exact To Consume Strict Survivors Directly

Current relaxed exact reads strict survivors via SQLite joins.

Replace that with:

- read strict survivor shards or manifest
- stream strict survivors directly into relaxed grouping
- emit relaxed group/drop shards directly

This preserves semantics while removing the worst SQLite replay path.

### 4. Make Docs-Exact And Survivor Export Consume Parquet Shards

The current export path builds `docs_exact.parquet` from SQLite joins.

Replace with:

- build from stage shard artifacts
- use SQLite only to look up minimal metadata if needed
- prefer direct parquet joins or DuckDB/PyArrow scans over large SQLite joins

### 5. Add Explicit Resume Points

Add stage-local completion markers for:

- strict exact materialization
- relaxed exact materialization
- docs-exact export
- exact survivor export

On resume:

- if strict exact outputs are complete, do not recompute them
- if relaxed exact outputs are incomplete, discard only that stage’s incomplete temp files and continue

## Implementation Plan

### Step 1: Add Validation And Forensics Helpers

- script to inspect a run root and state root
- report:
  - stage completeness
  - strict artifact presence
  - relaxed artifact completeness
  - manifest consistency

### Step 2: Refactor `build_stage_results(...)`

Current function:

- deletes stage rows from `run_exact_results`
- scans SQLite
- reinserts all stage membership rows
- writes final parquet outputs

Replace with:

- stream grouped rows from a reader
- write parquet shards directly
- write only compact stage summary and optional artifact registry to SQLite

### Step 3: Introduce Shard Manifests

For strict and relaxed exact:

- `groups_manifest.json`
- `drops_manifest.json`
- optional `kept_manifest.json`

Each manifest should record:

- shard count
- hash partition scheme
- row counts
- completion marker

### Step 4: Refactor Exact Survivor Export

Make exact survivor export read the exact kept-doc shard manifest, not reconstruct all decisions through giant SQLite joins.

### Step 5: Keep Backward-Compatible Final Outputs

Even after sharding, continue producing the canonical single-file outputs expected downstream:

- `strict_exact_groups.parquet`
- `strict_exact_drop_list.parquet`
- `relaxed_exact_groups.parquet`
- `relaxed_exact_drop_list.parquet`
- `docs_exact.parquet`
- `exact_survivor_manifest.parquet`

But produce them as a final merge of completed shards, not as the primary working representation.

### Step 6: Add Parallel Export

Parallelize by partition:

- partition creation
- per-partition group emission
- per-partition drop emission
- final shard merge

The CPU worker should be busy in this stage, not mostly idle.

## Salvage Plan For The Current Run

1. Stop the live run.
2. Snapshot current state and run root.
3. Validate strict exact outputs and strict exact membership.
4. Keep strict exact artifacts if valid.
5. Delete only incomplete relaxed exact temp files and any invalid relaxed stage rows.
6. Resume with the refactored exact-stage export path.

## Test Plan

### 1. Golden Exact-Stage Equivalence

Run old and repaired exact-stage code on the same tiny real-document dataset and require exact equality, after deterministic sort if needed, for:

- `strict_exact_groups.parquet`
- `strict_exact_drop_list.parquet`
- `relaxed_exact_groups.parquet`
- `relaxed_exact_drop_list.parquet`
- `docs_exact.parquet`
- `exact_survivor_manifest.parquet`

Also require exact equality for:

- exact kept-doc mapping per `doc_key`
- exact dropped-doc mapping per `doc_key`

### 2. Resume Equivalence

Create a partial run, then resume it with the repaired path.

Require equivalence against a fresh complete repaired run for:

- exact outputs
- stage summaries
- survivor manifest

### 3. Downstream Contract Equivalence

Use the repaired exact-stage outputs as input to:

- near dedup
- final exports
- builder metadata export
- dedup overlay publish
- mix build

Require:

- same output schemas
- same field meaning
- same dedup decisions visible to downstream builder consumers

### 4. Performance Regression Checks

Measure and record:

- SQLite main DB size
- WAL size
- exact-stage wall time
- CPU utilization during exact-stage finalization

Acceptance expectation:

- WAL no longer grows pathologically
- exact-stage finalization uses meaningful parallel CPU work
- total exact-stage runtime is materially lower on medium-scale smoke

## Acceptance Criteria

The repair is good enough only if all of these are true:

1. The current live run can resume from saved state instead of restarting from raw corpus input.
2. Exact-stage finalization no longer grows a massive SQLite WAL.
3. Exact-stage uses meaningful parallel CPU work on the worker.
4. Golden equivalence tests confirm unchanged exact-stage semantics.
5. Resume equivalence tests confirm unchanged continuation semantics.
6. Tiny real-doc smoke still passes.
7. Medium-scale smoke completes in a sane time.
8. Downstream overlay, mix build, and training still accept the outputs without contract changes.

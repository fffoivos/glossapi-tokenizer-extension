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

## Wave-2 empirical findings (2026-04-26 → 2026-04-27, 49.3M docs)

Concrete per-call-site notes from the wave-2 production run, including
where parallelism actually broke and what fixed (or could fix) it. All
line numbers reference `glossapi_corpus_cli/text_dedup.py` at commit
state of `cleanup/cleaner-pipeline-20260425`.

### Single-threaded paths confirmed pathological at 49M scale

1. **`build_stage_results` strict + relaxed exact rebuild** (line 2243-2415)
   - Outer Python loop reads `iter_exact_stage_rows` ordered by
     `exact_strict_hash` (or relaxed) for partitions=1 (strict) /
     partitions=256 (relaxed), accumulates a `current_group` list,
     calls `flush_group` per hash boundary. DuckDB threads=8 inside
     `iter_exact_stage_rows` only.
   - **Observed cost.** Killed strict at ~2h with 38 GB membership
     parquet half-written; projected total 4-8h serial.
   - **Wave-2 workaround.** `subprojects/01_1_corpus_dedup/scripts/parallel_exact_stages.py`
     pre-builds the 6 parquet files (strict_exact_{memberships,groups,
     drop_list}, relaxed_exact_{memberships,groups,drop_list}) in
     **32 seconds total** using DuckDB at 64 threads with
     `arg_min(doc_key, prio_struct)` and an inline priority STRUCT
     mirroring `selection_priority_tuple`. Then `build_stage_results`
     hits the `{stage}:reuse_existing_parquet` fast-path at line 2262-2269.
   - **Library-side repair.** Replace the inner Python group-by with
     a single DuckDB query per stage. Drop `partitions=1` for strict
     entirely. The script already proves the semantics match.

2. **`write_run_docs_inventory`** (line 1769)
   - SQLite stream → parquet, single-threaded.
   - **Observed cost.** 24 min for 49.3M rows.
   - **Wave-2 workaround.** Patched line 3946-3953 to skip
     regeneration if `run_docs_inventory.parquet` already exists +
     non-empty (uses `pq.read_metadata().num_rows` for the row count
     it would have returned).
   - **Library-side repair.** Same skip-if-exists guard, OR use
     DuckDB attached to sqlite to parallelize the SELECT *.

3. **`write_snapshot_manifest`** (line 2418-2440)
   - SQLite SELECT * stream → parquet, single thread.
   - **Observed cost.** 10 min for 49.3M rows, 9.13 GB output.
   - **Library-side repair.** Materialize directly from
     `run_docs_inventory.parquet` via DuckDB (the inventory has the
     same superset of fields). No need to round-trip through sqlite.

4. **`write_docs_exact_export`** (line 2442+)
   - Internal DuckDB query at 64 threads, but the consumer side
     `fetch_record_batch(rows_per_batch=2048)` → `ParquetWriter.write_table`
     is single-threaded.
   - **Observed cost.** 3:45 min (relatively fast because in-memory
     DuckDB held the whole join result before flushing). Output
     22.97 GB.
   - **Library-side repair.** Have DuckDB write the parquet
     directly with `COPY TO 'docs_exact.parquet' (FORMAT parquet, …)`
     instead of streaming through Python.

5. **`_run_near_cluster_stage` SQLite insertion loop** (lines 6236-6280)
   - **The single biggest pathology this run.** After `near_clusters.parquet`
     and `cluster_summary.parquet` are written, the code re-reads
     `near_clusters.parquet` row-by-row and `INSERT`s every row
     (~40M, including singleton rows) into `run_near_results` with
     `conn.commit()` per batch. SQLite WAL fsyncs serialize.
   - **Observed cost.** ~1.5+ hours single-threaded. State.sqlite
     grew from 71 GB to 89 GB. write_bytes on the proc grew at
     10.8 GB/min (mostly WAL churn).
   - **Use of run_near_results.** Only consumed at line 7271 for a
     `COUNT(*)` and `SUM(dropped)` summary stat — pure cosmetic.
   - **Library-side repair.** Drop the per-row INSERT entirely.
     Replace the summary at line 7271 with `SELECT COUNT(*),
     SUM(dropped) FROM read_parquet('near_clusters.parquet')` via
     duckdb. Saves ~1-2 hours of single-threaded work on every
     49M-doc dedup run.

6. **`_build_final_exports`** (line 6410+)
   - Same anti-pattern as `write_docs_exact_export`: DuckDB query
     internal-parallel, single-thread parquet writer downstream.
   - **Observed cost.** Currently in flight; expected 10-30 min.
   - **Library-side repair.** Same — use `COPY (... ) TO ... (FORMAT
     parquet)` instead of fetch-record-batch.

### Parallel-but-undersized

7. **`_run_near_cluster_stage` chunk resolution** uses
   `cluster_worker_count` (default 8) — see line 6092 trace. The
   `--max-workers 64` flag does not propagate. For 12,619 chunks at
   8 workers it took ~6 min, so not load-bearing today, but worth
   raising the default cap or honoring `--max-workers`.

### Storage architecture issues

8. **Per-shard + consolidated double storage** in Stage 2.
   `combine_parquet_files` writes the consolidated `signatures.parquet`
   (59 GB) and `lsh_buckets.parquet` (117 GB) but leaves the
   per-shard `shards/signatures/` (65 GB) and `shards/lsh_buckets/`
   (225 GB) on disk. **Cost.** 290 GB of redundant intermediate.
   - **Repair.** After `combine_parquet_files` succeeds, delete the
     source shard dirs. Add a `--keep-shards` flag for debugging.

9. **`exact_survivor` shards never cleaned up** after near_signatures
   consumed them. 199 GB of stage_01 survivor shards persisted
   through Stage 2 into final_export.
   - **Repair.** Delete `stage_01_exact/shards/exact_survivors/` at
     end of `_run_near_signature_stage` (or guard with --keep-shards).

10. **No disk preflight.** A 49M-doc dedup run requires ≥1 TB of
    intermediate + state. The CLI doesn't check `df` upfront, so the
    run died with `OSError errno 28` 6h into Stage 2 with no
    actionable warning.
    - **Repair.** At `_run_near_signature_stage` start, estimate
      `lsh_bucket_rows ≈ rows × bands × 1.5` and required disk; bail
      with a clear "needs N GB more, free up or attach more disk"
      message if not satisfied.

### Order of repairs by ROI

For the next dedup library iteration, attack in this order:

1. Drop the `run_near_results` per-row INSERT loop (saves 1-2 hr per run).
2. Replace strict/relaxed `build_stage_results` with the
   parallel_exact_stages.py DuckDB approach (saves 4-8 hr per run).
3. Skip-if-exists guards for inventory + snapshot_manifest (saves 30+ min on every resume).
4. Auto-cleanup of consolidated stages' per-shard intermediates
   (saves 200-300 GB disk, prevents disk-full crash).
5. Use `COPY TO parquet` directly instead of fetch_record_batch +
   ParquetWriter for docs_exact and final_exports (saves several min each).
6. Disk preflight at Stage 2 start.

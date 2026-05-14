# Builder And Tokenizer Efficiency Plan

## Scope

This plan covers the downstream stages after dedup:
- builder-time dedup replay during mix construction
- tokenizer training over the built parquet mixes
- uploader staging for source-only and later full dataset publication

It is separate from the near-dedup redesign work, but it follows the same core constraints:
- no semantic drift in builder decisions
- resumability at sane stage boundaries
- avoid OOM by design, not by emergency throttling
- maximize useful CPU throughput without losing recoverability

## Current Contract Assumptions

### Builder

- builder-time dedup consumes `dedup_metadata/latest.json`
- the active builder path prefers `builder_metadata_v2`
- `builder_metadata_v2` exports:
  - `doc_dedup_metadata.parquet`
  - `dedup_family_membership.parquet`
  - `near_candidate_pairs.parquet`
- when `dedup_family_membership.parquet` exists, builder replay should not need to load `near_candidate_pairs.parquet`
- `near_candidate_pairs.parquet` is still intentionally retained in the bundle as an evidence/audit artifact

### Tokenizer

- tokenizer training consumes parquet mixes, not the raw source-parquet corpus
- tokenizer input is streamed batch-by-batch from parquet
- tokenizer training should remain independent from the source upload track

## Applied Fixes

1. Builder now avoids loading `near_candidate_pairs.parquet` when exported family membership is present.
2. Builder reuses existing `doc_key` values instead of recomputing them row-by-row when they are already present in the mix input.
3. Uploader handoff now supports `source_only` scope, so filtered HPLT/source upload can start before dedup metadata is ready.
4. Uploader staging now respects `sync_paths`, instead of implicitly syncing the whole working release root.

## Remaining Risks

### Builder

- `doc_dedup_metadata.parquet` is still loaded into pandas for the reduced duplicate subset.
- extremely large duplicate subsets may still create a high-memory replay.
- overlap accounting and some source-mix helpers still use pandas-heavy intermediate frames.

### Tokenizer

- throughput still depends on:
  - `RAYON_NUM_THREADS`
  - parquet batch size
  - filesystem read bandwidth on the worker
- we do not yet have a worker-side throughput sweep on the real mix shards.

## Required Benchmarks

### Builder Replay Sweep

Run on worker hardware with progressively larger duplicate subsets and record:
- peak RSS
- wall time
- duplicate rows before replay
- duplicate rows after replay
- family count
- shared-family count

Required checks:
- no contract drift in kept rows
- no unexpected fallback to legacy near-pair replay
- no OOM at the expected duplicate-subset ceiling

### Tokenizer Throughput Sweep

Run on worker hardware across:
- `RAYON_NUM_THREADS = 16, 24, 32`
- parquet `batch_size = 2048, 4096, 8192`

Record:
- runtime
- peak RSS
- CPU utilization
- tokenizer output identity

Success target:
- keep the streaming path
- find the best throughput point without pushing RSS into unsafe territory

## Next Changes If Needed

### Builder

1. Push more of the reduced-bundle replay into DuckDB if duplicate-subset pandas loads become too large.
2. Replace remaining pandas-heavy overlap/accounting steps with parquet-backed SQL where possible.
3. Add a dedicated worker stress harness for builder replay, not just near-dedup.

### Tokenizer

1. Expose batch size as an orchestration-level knob if the worker sweep shows a better setting than the current default.
2. Pin a worker-class default for `RAYON_NUM_THREADS` after the throughput sweep.
3. Keep tokenizer training on GCP workers only; do not move it back to `home`.

## Integration Back Into The Main Plan

Once:
- source-only upload is running cleanly,
- builder replay is validated against the current dedup bundle,
- tokenizer throughput defaults are benchmarked,

fold the resulting defaults back into the canonical tokenizer pipeline and the core project plan.

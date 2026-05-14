# Near Dedup Memory Footprint TODO

This file tracks concrete memory-reduction work for `stage_02_near`, separate
from semantic decisions. The constraints stay the same:

- dedup functionality must stay the same
- resumability must be preserved at stable stage boundaries
- throughput should improve, not just memory safety

## Priority Order

1. Stream candidate generation by bucket instead of materializing a full band in memory.
Status: completed
Why:
- current `build_candidate_band_chunk(...)` still builds a full-band
  `bucket_members` map before processing
- this keeps memory growing for too long before the first durable completion

2. Replace band-level work units with smaller resumable chunks.
Status: in progress
Target shape:
- `band + bucket-prefix partition`
or
- `band + shard group`
Why:
- keeps many workers active without forcing each worker to carry a whole band
- gives earlier durable completions and better observability

Current implementation:
- the near-candidate stage now checkpoints `band + bucket-prefix` chunks
- bucket-member partition files are materialized per band and per prefix before candidate generation
- this is a substantial reduction in chunk size, but not yet the final shard-level design

3. Make shard-level progress authoritative for resume.
Status: in progress
Current issue:
- partial bucket shard outputs are durable on disk
- but resume still treats the band as the authoritative completion boundary
Goal:
- if a shard is done, resume should not recompute it

Current implementation:
- prefix chunks are now authoritative instead of whole bands
- resume is materially better than before, but still not at individual bucket-shard granularity

4. Stop using large Python object collections in the hot path.
Status: pending
Candidates:
- replace `list[dict]` buffers with typed tuples/arrays where practical
- eventually prefer integer doc ids over repeated string doc keys in worker-local state

5. Stream candidate rows to parquet writers instead of building large Python lists.
Status: completed
Why:
- oversized buckets can still create very large in-memory `candidate_rows` buffers

6. Stream bucket-summary and touched-doc outputs incrementally.
Status: completed
Why:
- these outputs do not need a full in-memory accumulation before write

7. Add a memory guard for live runs.
Status: pending
Goal:
- if available memory falls below the configured floor, stop taking new work
- let in-flight chunks finish and checkpoint cleanly

8. Consider allocator-level reductions after algorithmic fixes.
Status: pending
Candidates:
- `MALLOC_ARENA_MAX=2`
- `jemalloc`
Note:
- only after chunking and streaming fixes are in place

## Validation

Each item above is only considered complete when all of the following are true:

1. Correctness:
- targeted tests pass
- contract tests pass
- no semantic drift in final dedup decisions on the comparison corpus

2. Resumability:
- interrupted run resumes from the latest completed near stage boundary
- redesigned chunks resume without recomputing already completed shard work

3. Efficiency:
- lower peak worker memory on the stress harness
- first durable progress appears earlier than before
- higher safe worker count on the large-memory worker

4. End to end:
- small real-doc pipeline smoke still passes through downstream stages
- live large run reaches `near_candidates -> near_clusters -> final` without memory collapse

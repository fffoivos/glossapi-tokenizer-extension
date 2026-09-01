# patches — inherited-trainer benchmark-range offset

> **In one line:** one patch against subproject 07's `train_segment.sbatch`, so that a profile benchmark could run at a non-zero starting iteration.
> **Period:** committed 2026-08-16 (`de6d9b79`). **Status:** complete; superseded in practice once profile research stopped being a separate allocation.

## Why this existed

The targeted experiments reused 07's segment launcher unchanged wherever possible. That launcher hard-coded the benchmark iteration window as `0 <= start < end <= 288`, which is only correct for a benchmark starting from update 0. A targeted benchmark that starts from an existing checkpoint needs the same 288-update width measured from its own base.

## What it does

[`train_segment_targeted_benchmark_offset.patch`](train_segment_targeted_benchmark_offset.patch) adds a `FULL8_BENCHMARK_BASE_ITERATION` argument (default `0`) and changes the range check to `benchmark_base <= start < end <= benchmark_base + 288`. Nothing else in the contract-derivation block changes, so a base of `0` reproduces the inherited behaviour exactly.

## Outcome

The already-promoted 8B profile was reused rather than re-benchmarked, and 1.5B dynamic evidence was collected by its own qualification work, so this patch saw little production use. It is retained because it is the only in-repo modification of an inherited trainer script.

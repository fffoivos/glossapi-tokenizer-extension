# HF Dedup Investigation

## Why This Exists

This is an explicit diversion from the original execution plan.

Reason:
- the repo-backed `near_candidates` stage scaled poorly on the large live run
- on `m3-megamem-64` (`976 GB` RAM), `16` near-candidate workers drove memory to about `955 / 960 GB`
- the stage still showed `0 / 32` completed bands
- that means the current near-candidate execution shape is not efficient enough for the intended scale

## Primary Sources Used

- Hugging Face DataTrove README:
  - `https://github.com/huggingface/datatrove/blob/main/README.md`
- Hugging Face DataTrove MinHash implementation:
  - `https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py`
- Hugging Face DataTrove MinHash example pipeline:
  - `https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication.py`
- Hugging Face DataTrove MinHash tests:
  - `https://github.com/huggingface/datatrove/blob/main/tests/pipeline/dedup/test_minhash.py`

## Our Current Semantics

These are the parts we must preserve:

- exact dedup first:
  - `strict_exact`
  - then `relaxed_exact`
- near dedup after exact
- near admission rules:
  - MinHash/LSH candidateing
  - similarity threshold `>= 0.85`
  - `length_ratio >= 0.70`
- keeper selection:
  - OCR validity
  - `needs_ocr`
  - representative score from length and badness
  - metadata tie-breaks
- downstream builder metadata contracts
- resumability across stages

## What HF/DataTrove Does Better

The main strength is execution shape, not semantic equivalence.

DataTrove MinHash is deliberately staged and file-based:

1. stage 1:
- compute MinHash signatures per task
- write compact binary signature files per bucket
- sort them on disk

2. stage 2:
- process one bucket per task, optionally split further by hash range
- merge sorted signature streams with a heap
- emit duplicate pairs as compact `.dups` files

3. stage 3:
- cluster duplicate pairs with union-find

4. stage 4:
- filter the original dataset based on cluster assignments

The important scaling properties are:
- no global in-memory signature map per worker
- work is file-streaming and merge-based
- partitioning is explicit
- resume is task-based and coarse-grained
- memory usage is bounded by the active stream/merge windows, not by the full signature universe times worker count

## Why HF Is Not A Drop-In Replacement

It does not match our full semantics.

Mismatches:
- it does not include our upstream `strict_exact` / `relaxed_exact` path
- it does not include our `length_ratio >= 0.70` validation rule in the same way
- it does not include our representative/keeper policy
- it does not produce our builder metadata outputs
- it does not implement our stage contracts or resume markers

So the answer is:
- not a direct drop-in replacement
- yes, a strong reference architecture for the near-candidate execution path

## Best Path Forward

Preferred direction:
- keep our semantics
- replace the current `near_candidates` execution shape with a DataTrove-like staged external-merge design

That means:

1. precompute compact near-signature shards
- one row per doc
- compact binary or Arrow-native layout
- partitioned by band and sorted by the band key

2. make near-candidate work chunkable within a band
- current unit is too coarse: `32` giant bands
- replace with band-plus-hash-range chunks
- progress should update per chunk, not only when a whole band completes

3. stop loading a full signature map in every worker
- current `load_signature_index(...)` duplicates too much state
- use streaming sorted readers or memory-mapped columnar data instead

4. emit candidate pairs as shard files
- one shard per chunk
- aggregate later
- do not hold the full candidate set in worker memory

5. keep clustering and keeper resolution as separate later stages
- candidate generation should only generate candidate pairs
- representative selection should remain in our own later logic

## Concrete Changes To Our Script

These changes preserve semantics while improving efficiency:

1. Replace `load_signature_index(...)` in `near_candidates`.
- do not build a global Python signature map per worker
- use sorted per-band shards and streaming scans

2. Change the unit of work.
- from: one full band task
- to: band plus hash-range task

3. Use a compact on-disk data plane.
- binary fixed-width records or Arrow-native arrays
- not Python dict-heavy structures

4. Separate pair generation from pair aggregation.
- shard outputs first
- aggregate afterward

5. Improve observability.
- per-chunk progress markers
- per-chunk memory and pair-count summaries
- do not rely on `0 / 32` until a whole band finishes

6. Keep the representative rules out of candidate generation.
- HF-style pair generation can be borrowed
- our keeper logic stays ours

## Recommended Decision

Recommended:
- do not try to drop in DataTrove as-is
- transplant the staged external-merge candidate-generation architecture into our near-dedup path

This gives us:
- the same semantics
- much lower per-worker memory duplication
- better progress granularity
- better resume behavior

## Status

Current operational consequence:
- the large live run was stopped intentionally after the `16`-worker `near_candidates` attempt saturated memory
- the next dedup implementation step should be a near-candidate redesign based on the points above before resuming the full pipeline again

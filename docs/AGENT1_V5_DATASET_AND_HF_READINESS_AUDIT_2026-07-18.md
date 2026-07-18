# Agent 1 v5 construction, HF storage, and dedup operational audit

**Audit date:** 2026-07-18
**Run:** `agent1-v5-clariden-debug-20260715T111552Z-30c72e9`
**Operational verdict:** **construction and HF storage passed; deduplication is healthy and in progress**

## Scope and execution directive

This document answers two operational questions:

1. Was the combined pre-dedup dataset constructed, preserved, and stored
   correctly on Hugging Face?
2. Is the active deduplication run correct, and what must change to make it
   faster without changing its semantics?

The answer to the first question is **yes**. The answer to the second is that
the serial chain is correct and healthy but unnecessarily slow.

The `greek>60` distribution and the structural-quality counters are recorded
for later quality-policy work. They are **deferred, non-blocking observations**
for the current construction and dedup operation. They do not instruct an
execution agent to stop, rebuild, refilter, cancel, or restart the live chain.

> **Execution directive:** continue the current serial dedup chain unchanged
> until a deliberate migration at the acceleration plan's passed-receipt
> boundary. Never cancel a running rank because of an observation in this
> document.

## Executive operational verdict

| Operation | Verdict | Evidence |
|---|---|---|
| Raw acquisition integrity | **Pass** | 293 / 293 files and 153,031,751,003 / 153,031,751,003 bytes verified; every mismatch counter is zero |
| Candidate transformation | **Pass** | 3,576,290 inputs accounted for; 4,703 transform quarantines recorded; 3,571,587 survivors |
| GlossAPI processing | **Pass** | One additional quarantine recorded; 3,571,586 candidate survivors |
| Metadata preservation | **Pass** | All 158 transform→GlossAPI→release lineage tasks passed; IDs, cleaned-text hashes, titles, authors, and source metadata remained attached |
| Combined release construction | **Pass** | 431 ZSTD Parquet shards, 53,046,533 rows, 149,746,029,389 payload bytes, one 44-column schema |
| Private HF storage | **Pass** | Every authored object matched by path, size, and SHA-256; Dataset Viewer reports the same rows, bytes, split, and schema |
| Exact dedup preparation | **Pass** | Exact-index manifest passed and the frozen near-dedup configuration is bound |
| Near deduplication | **Healthy, in progress** | Contiguous passed signature ranks with 32 verified outputs per completed rank; the self-chain continues normally |
| Quality-score handling | **Deferred, non-blocking** | Diagnostic counters retained for later adjudication; no filter-policy change is part of the current operation |

No missing shard, corrupt Parquet footer, schema drift, checksum mismatch,
document/metadata misattachment, or failed completed signature rank was found.

## Construction and metadata evidence

The construction manifests account for every candidate input:

```text
3,576,290 candidate inputs
-    4,703 recorded transform quarantines
=3,571,587 transform survivors
-        1 recorded GlossAPI quarantine
=3,571,586 released candidate rows
```

These are recorded pipeline outcomes, not silent data loss. The combined
pre-dedup release then adds 49,474,947 protected NanoChat rows for a total of
53,046,533 rows.

The exhaustive lineage audit checked both boundaries:

1. transform→GlossAPI: source row UID, source identity, candidate document ID,
   title, author, and source metadata;
2. GlossAPI→release: source identity, collision-resolved document ID,
   cleaned-text SHA-256, title, author, and source metadata.

All 158 tasks passed. The envelope found 52 duplicate source-ID keys and
deterministically rewrote 105 affected row IDs; every expected rewritten ID
matched the correct released document.

The 273 NanoChat inputs contain three column-set shapes and 14 physical schema
variants. Construction cast them into the reviewed 44-column union without
changing protected NanoChat text. All 431 release shards use that ordered
schema and ZSTD compression.

## Acquisition integrity

Clariden job `2789755` completed the read-only acquisition audit. Receipt
SHA-256:

```text
7b975687e0a8e3f3869a8c5cef4600cf55865523472ef150303c860e598f4f5c
```

It rehashed all 293 acquired files across the 18 candidate sources plus the
NanoChat base. Hash, size, device, inode, mtime, ctime, path containment,
contract, manifest, and task-identity mismatch counters are all zero. Task
closure matched 273 / 273 base tasks and 158 / 158 transform tasks.

## HF storage verification

Private repository:

```text
fffoivos/greek-nanochat-plus-new-pre-dedup-agent1-v5-clariden-debug-20260715T111552Z-30c72e9
```

Immutable base-data commit:

```text
362d99a79dece48e2c54a8924cf4419e02e3bee0
```

Verified base composition:

- 273 NanoChat shards / 49,474,947 rows;
- 158 candidate shards / 3,571,586 rows;
- 431 Parquet shards / 53,046,533 rows;
- 149,746,029,389 Parquet bytes;
- 435 authored files / 149,746,154,488 bytes;
- 436 Hub tree entries including `.gitattributes`.

The base publication receipt SHA-256 is:

```text
ebad0b3374b7863c558b9f1ac6eb128fe8a1abaf4e89b8bd46aea67ad62bd681
```

The operational-clarification card and provenance were published in a later
metadata-only commit:

```text
c0870ebd6887000d64137077d8126706940356d6
```

That commit changes only `README.md` and `manifests/provenance.json`. Its
publication receipt SHA-256 is:

```text
c7f1cd8dfb7713dfbed027e4f89ef804ce25c6168d9e28fdf5393fa468b6f832
```

Every other remote object must remain unchanged from its parent commit. The
data should always be loaded from the immutable base-data commit above.

## Deferred quality observations — no current stop-work

The exhaustive candidate audit receipt is:

```text
4fdeaa54283fcb90d30302bf3b73a412bb949db035300d9265e743c39f134ad0
```

It inspected all 3,571,586 candidate rows and recorded:

| Observation | Rows |
|---|---:|
| `filter="greek>60"` | 275,816 |
| Forbidden C0 control | 10,728 |
| U+FFFD replacement character | 389 |
| Complete recognized residual HTML tag | 29 |

The receipt itself says `status="blocked"` because the audit tool was written
with conservative release-grade quality gates. For the operational scope set
here, that status is **not inherited by dataset construction or deduplication**.
Quality handling is deferred; the current dedup input remains unchanged.

`greek>60` means `greek_badness_score > 60`, not Greek content above 60%.
Flags are concentrated in Diavgeia (270,034) and the modern-Greek dictionary
(5,372), with 410 across all other sources. The score includes both useful
signals and false positives, so no blanket deletion is part of this run.

## Dedup operational status

The frozen recipe remains unchanged:

```text
language: ell_Grek
Greek diacritics: preserved
token shingles: 5
MinHash permutations: 128
bands x hashes: 32 x 4
seed: 1
hash precision: 64
verified Jaccard: 0.85
NanoChat origin preference: protected
```

At the `2026-07-18 16:50 EEST` live check, ranks `0..49` had passed
contiguously with 32 outputs each and rank 50 (job `2790183`) was running
normally. No signature job was stopped, cancelled, or modified.

The current implementation rereads and SHA-256 validates all 149.746 GB before
processing each single signature rank. That repeated validation is safe but is
the principal performance problem. It is not evidence of incorrect dedup
outputs.

## Required operational changes

Only the following changes are required by the present scope:

1. Replace per-rank full-corpus validation with one receipt-bound full input
   audit plus validation of only the assigned shard in each rank.
2. Deploy an immutable runner that preserves every existing MinHash parameter,
   NanoChat protection rule, schema field, and metadata value.
3. Benchmark 1→2→4→5 local workers on one `normal` node and select the fastest
   fan-out that passes the throughput, memory, and I/O gates.
4. Migrate only at a completed receipt boundary using the held-fence procedure;
   never cancel a running rank.
5. Finish all signature, bucket, shingle, verifier, cluster, and filter stages
   with complete receipt/output closure.
6. Materialize the deduplicated dataset and publish it to the intended private
   final HF repository.
7. Verify the final HF repository by complete path, byte, checksum, schema,
   metadata, and row-count inventory.

Quality-score policy remains a separate later task and must not be inserted
into this performance migration.

## Acceleration plan

The implementation plan is
`AGENT1_V5_CSCS_DEDUP_ACCELERATION_PLAN_2026-07-18.md`.

Measured at the rank-44 boundary, the serial design implied about 9.82 days of
remaining signature work and 57.8 TB of redundant validation reads. The plan
uses one full audit, per-shard validation, and a bounded 1→2→4→5 worker canary
on one normal-partition node. The estimated remaining signature time after
migration is 42–49 hours, or roughly 4–5× faster, without changing dedup
semantics.

The plan has not yet altered the live chain. Until the technical preflight,
canary, and safe cutover are deliberately executed, the existing serial chain
must continue normally.

## Operational conclusion

The verified current pre-dedup release remains the input to the active dedup
run. There is no operational reason to rebuild it or stop the chain. The next
work is to implement the performance-only acceleration safely, finish dedup,
and publish the verified private deduplicated dataset.

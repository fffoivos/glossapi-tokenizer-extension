# Agent1 v5 — Oversized LSH group diagnosis (2026-07-21)

**Status: diagnosis complete, awaiting owner review. No cluster job launched. No resolution applied.**

The dedup pipeline is blocked at LSH pair-merge by **312 LSH groups above the frozen
`dedup.max_bucket_documents = 5000`** cap. This is the read-only diagnosis of what those
groups are and the recommended resolution.

RUN = `/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/agent1-v5-clariden-debug-20260715T111552Z-30c72e9`

## Method (read-only, faithful to the merge)

Reconstructed the 312 groups directly from the raw bucket files
(`60-dedup/minhash-buckets/{b:05d}_00.dups`) using the **same decoder and head-to-tail
chain logic** as `merge_lsh_pairs` (`PAIR_STRUCT="<4I"`, `ref=(rank<<32)|doc`), then joined
document refs to sources/text. Reconstruction is **exact** — it reproduces the merge's own
counters bit-for-bit:

| Counter | Reconstructed | Merge manifest |
|---|---|---|
| raw pairs | 111,039,644 | 111,039,644 |
| oversized groups | 312 | 312 |
| oversized docs (with multiplicity) | 3,265,439 | 3,265,439 |
| oversized edges excluded | 3,265,127 | 3,265,127 |

So this is real LSH candidate data, **not a grouping/encoding defect.**

## What the groups actually are

An oversized group = documents that share **one** band (4 of 128 min-hashes) — i.e. LSH
**candidates**, *not* confirmed duplicates. Representative groups (sampled real
`source_dataset` + text):

| Group | Real source | Text | Reading |
|---|---|---|---|
| **41,413** (largest; `origin=nanochat_base`) | `HPLT/ell_Grek_ge8_no_mt_clean60` | real articles, p50 **2,030** chars, 0% empty, all *different* | **Coincidental single-band collision** → will fail Jaccard≥0.85 → **kept**. |
| **32,551** (largest candidate) | `glossAPI/diavgeia` | templated legal acts, p50 3,083 chars, headers **byte-identical** | **Genuine near-dup family** → verify confirms → **deduplicated**. |
| **8,459** (collide in **all 32 bands**) | `openarchives.gr` | short (p50 310), mostly `<!-- image -->` placeholders | Near-empty image-only records, near-identical → collapsed to one exemplar. |

Distinct docs touched by oversized groups: **796,275** (candidate 577,218 + nanochat_base
219,057). Cross-band recurrence confirms a single degenerate family: **8,459 docs appear in
all 32 bands** (the "8457/8459 ×N" signal), all `openarchives.gr` image-only records.

## Verdict

**No defect. A mix of (a) genuine near-dup families and (b) coincidental single-band
collisions — every case is handled correctly by admitting the groups to `verify`.** The
exact-Jaccard≥0.85 verify stage is the real filter: it removes the real dupes (Diavgeia,
openarchives) and keeps the coincidental collisions (HPLT). NanoChat/HPLT stays protected as
representative. Because groups are stored as **spanning paths (N−1 edges)**, admitting all
312 adds only ~3.265M candidate pairs (**+~8%** verify cost), not O(N²).

Net effect of admitting them: **more *correct* removals** — which is the goal of dedup.

### Deferred quality observation (not a blocker)
The ~8,459 `openarchives.gr` all-32-band records are mostly `<!-- image -->` placeholders
(near-empty, image-only archive entries). Dedup will collapse each near-identical set to one
exemplar. A *dedicated* quality drop of image-only records is a **separate, explicitly-deferred**
concern per the 2026-07-18 docs and is **not** folded into this step.

## Recommended resolution (decided: raise the cap; owner greenlight pending)

- Raise **only** `dedup.max_bucket_documents: 5000 → 50000** (above the 41,413 max, keeps a
  genuine runaway guard) via a new `configs/agent1_v5_eiger_pipeline.resolved.json`; **all other
  dedup params byte-identical**; immutable deployed code untouched.
- **Config-pin risk cleared:** `contract()` validates only `schema_version`/`status`; every dedup
  param is read from `--config`, and no code compares the live config to the contract's inlined
  `dedup`/`config.sha256`. Supplying a raised-cap config is safe.
- Execution (not yet run): write resolved config + `lsh_oversized_resolution.json` receipt →
  `pair_merge_capacity_canary.sh` at the new cap → re-run `merge-lsh-pairs` to a **fresh**
  output (`lsh_pairs.sqlite`; keep `.blocked` as audit trail) → `audit-pairs` (`ok:true, oversized:[]`)
  → downstream chain (shingles → verify → cluster → filter → release → publish).

## Evidence on the cluster (under RUN)

- `lsh_oversized_diagnosis.json` — consolidated verdict receipt.
- `lsh_oversized_diagnosis_partA.json` — per-group sizes, source composition, cross-band
  multiplicity histogram (no parquet).
- `lsh_oversized_diagnosis_partB.json` — sampled real `source_dataset` + text length/preview.
- `lsh_oversized_groups_nodes.npz` — the 312 groups' document-ref arrays (for any re-check).

Scripts: `diag_oversized_partA.py`, `diag_oversized_partB.py` (on Clariden `~/`, read-only).

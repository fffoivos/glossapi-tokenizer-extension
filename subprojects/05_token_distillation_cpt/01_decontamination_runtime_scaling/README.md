# 01 — Decontamination runtime scaling

> **In one line:** one measured 5 B-token decontamination run, extrapolated into Slurm
> reservations for every corpus the CPT track might screen.
> **Period:** measured 2026-06-02; committed 2026-06-11 (`a19c136f`). **Status:** completed.
> **Came from / led to:** the 5 B decontamination pilot → this → the Stage-A
> GreekMMLU decontamination in [`../03_training_experiments`](../03_training_experiments/README.md)
> and the audit-first passes in [`../04_full_corpus_preparation`](../04_full_corpus_preparation/).

## Why this existed

Decontamination is exact normalized word-n-gram overlap, so its cost scales with the data
scanned, not with the model. Before committing a full-pool scan the team needed a defensible
walltime request — and a statement that the job needs **no GPU**, which mattered because
Clariden GPU allocations were the scarce resource.

## History

The single artifact is [`estimate_inputs.json`](estimate_inputs.json)
(`schema: decontamination-runtime-scaling-v1`, `generated_date: 2026-06-02`). It records one
real measurement — Slurm job `2453791`, `COMPLETED 0:0`, `01:12:38` on 8 CPUs / 64 G on the
`xfer` partition, 3,739,911 rows / 22.9 GB / 5.0 B target tokens scanned at 872 rows/s
(5.35 MB/s), 62,572 hit records written — and scales it three ways (rows, bytes, tokens)
across four candidate pools. Byte scaling is declared the preferred basis for HPLT + GlossAPI
because GlossAPI rows are very large.

| Pool | Size | Scaled estimate | Reservation |
|---|---|---|---|
| Existing CPT 7 B mix | 5.75 M rows | 1.7–2.3 h | **3 h** |
| HPLT clean60 wave-4 staged | 48.7 M rows / 44.2 B tokens | 10.5–17.2 h | **24 h** |
| GlossAPI sources (modern-148k) | 746 k rows / 17.4 B tokens | 0.24–5.3 h | **6 h** |
| HPLT + GlossAPI combined | 49.5 M rows / **61.6 B tokens** | 14.7–22.5 h | **30 h** |

A fifth row estimates ~21 h / **36 h reserved** for a built 88 B-token training stream, with
an explicit caveat that it assumes the stream is already materialized as JSONL and is not a
raw source-pool estimate for replay, code or math candidates.

## Outcome

- The CPU-only, no-GRES posture for decontamination became policy for the whole subproject
  (restated in [`../ARCHIVE.md`](../ARCHIVE.md) under "Corpus-Prep Method Summary").
- The 24 h / 30 h reservations are the numbers later corpus jobs were sized against.
- The referenced audit report `../../reports/decontamination_audit_5b_20260602T011447Z.json`
  is **not present** in the repo, and one `corpus_counts` source path points at a Clariden
  scratch summary that is likewise not tracked. The measurement itself is self-contained in
  this file.

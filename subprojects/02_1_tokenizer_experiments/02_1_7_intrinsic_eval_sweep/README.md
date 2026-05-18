# 02.1.7 Intrinsic Eval Sweep

This subproject is the cutoff-decision stage for the C3 tokenizer arm.
It evaluates C3-derived Apertus-compatible tokenizer variants and freezes
the current ship target.

## Canonical Decision

**Chosen cutoff:** `17,408` added units.

**Canonical tokenizer:** `c3_added_17408_curated_padded`.

The selected tokenizer preserves Apertus base ids `0..131,071`, adds
`17,408` Greek-extension BPE units, keeps total vocab aligned at
`148,480`, and structurally removes the 69 curated noise tokens by
backfilling with the next valid C3 merges.

The decision is documented in:

- [`CHOSEN_CUTOFF.md`](CHOSEN_CUTOFF.md) — pinned ship contract
- [`REPORT.md`](REPORT.md) — evidence base for the sweep

## Post-Decision Firing Counts

After the cutoff was chosen, the canonical tokenizer was run over the
exact C3 BPE training corpus (`train.parquet` + row-aligned
`train_manifest.parquet`) to count token firings by source component and
by exact `source_dataset`.

Canonical local output:

```text
variants/c3_added_17408_curated_padded.firing_counts/
```

The compact artifact bundle is published in the Apertus-extension HF
repo under:

```text
experiments/02_1_7_intrinsic_eval_sweep_20260518/firing_counts_c3_added_17408_curated_padded/
```

Tracked provenance:

- [`FIRING_COUNT_RUN_20260518.md`](FIRING_COUNT_RUN_20260518.md) —
  completed-run summary and interpretation.
- [`FIRING_COUNT_README.md`](FIRING_COUNT_README.md) — runbook for the
  cloud firing-count workflow.
- [`manifests/firing_count_20260518_run_summary_augmented.json`](manifests/firing_count_20260518_run_summary_augmented.json)
  — compact machine-readable provenance with input hashes, smoke
  benchmark, shard summary, denominators, and source summaries.

Headline validation:

- `14,401,554` rows, `99.257B` chars, `24.892B` tokenized tokens.
- GlossAPI-nanochat: `49.79%` of token mass; HPLT: `50.21%`.
- All `17,408` added tokens fire in GlossAPI-nanochat; `27` are zero in
  HPLT; `0` are zero in the combined C3 corpus.
- Component and per-source count invariants pass exactly.

## What Git Tracks

Git keeps the reproducible process and the compact decision record:

- scripts in [`scripts/`](scripts/)
- evaluation configs in [`configs/`](configs/)
- the final reports
- canonical small manifests in [`manifests/`](manifests/)
- plots embedded by the report
- firing-count process scripts and compact completed-run provenance

The large generated tokenizer variants, raw metric tables, parquet
outputs, local tokenizer copies, and vendored TokEval checkout are
intentionally ignored. The canonical tokenizer artifact and minimal
evidence bundle are published to the Apertus-extension HF repo.

## Key Files

- [`manifests/curated_padded_at_17408_manifest.json`](manifests/curated_padded_at_17408_manifest.json)
  records the exact curation/backfill construction.
- [`manifests/removal_mask_at_17408.jsonl`](manifests/removal_mask_at_17408.jsonl)
  records the 69 in-cutoff tokens filtered from the raw 17,408 variant.
- [`manifests/tokeval_commit.txt`](manifests/tokeval_commit.txt)
  pins the TokEval implementation used for the intrinsic suite.
- [`manifests/firing_count_20260518_run_summary_augmented.json`](manifests/firing_count_20260518_run_summary_augmented.json)
  pins the completed firing-count run metadata.

## Local Large Artifact

The canonical local tokenizer lives at:

```text
variants/c3_added_17408_curated_padded/tokenizer.json
```

It is intentionally not tracked by Git. Its pinned SHA-256 is:

```text
358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394
```

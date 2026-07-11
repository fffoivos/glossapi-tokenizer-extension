# Phase 05 — validated release to Megatron training data

This directory bridges the validated **private** Phase-04 training release to
fresh Megatron binaries for the frozen full-corpus Token-Distillation probe. It
does not reuse deleted curriculum-v2 binaries, publish data, or launch GPUs.

## Frozen recipe

- Mix: 79% new Greek, 20% Apertus-family foreign replay, 1% old-Greek replay.
- Horizon: 25,000,000,000 nominal tokens. Integer batches produce exactly
  5,960 steps / 24,998,051,840 effective tokens; the floor residual is
  1,948,160 tokens.
- Sequence/global batch: 4,096 / 1,024 sequences.
- Tokenizer/init: ModernGreek-148480, TD layer 11.
- LR: `5.5e-5` → `5.5e-6`; 400-step warmup; final 20% `1-sqrt` cooldown.
- AdEMAMix: `0.9/0.999/0.999`, alpha 4; Goldfish 50/50.

The shares are Megatron `--data-path` sampling weights. Each eligible document
is encoded once; builders never reach a target by copying rows. Finalization
discounts exact-content duplicates and requires every pool and weighted foreign
source to have at least 1.005× unique capacity. Every physical prefix must also
have `ceil(planned_samples * 1.005) + 1` non-repeating samples; `+1` is the
sequence-boundary allowance.

## Identity, heldout, and code contracts

`freeze_inputs.py` accepts only a passed Phase-04 local release. It binds the
exact private Parquet inventory, replay-restaging receipts, tokenizer tree,
GreekMMLU decontamination policy, clean repository commit, and complete clean
Megatron source tree.

Document IDs use `full-cpt-document-identity-v2`. Shard-local upstream IDs are
file-scoped. Old Greek is globally identified by
`(source_dataset, source_doc_id)`; `source_doc_id` alone is forbidden.

Nine deterministic LM-loss heldouts are rebuilt: HPLT, OpenArchives, Greek
PhD, English, German, Russian, Chinese, code, and old Greek. Every required
exclusion is checksum-bound and mandatory. Finalization proves each selected ID
was excluded exactly once. Heldout, shard, finalizer, asset, and launch
programs verify their own bytes against the frozen code receipt.

Each binary task atomically writes `.bin`, `.idx`, retained identity/content
ledger, contamination ledger, and finally its manifest. The manifest binds the
exact task, input, pool, source, heldout identity, exclusion, tokenizer, code,
and output prefix. Finalization uses a disk-backed exact uniqueness audit; it
does not hold corpus identities in RAM.

## Receipt-bound replay restaging

Clariden currently has empty replay and Megatron skeletons. Restaging is a
separate CPU prerequisite configured by `configs/replay_acquisition.json`.
Nanochat and the Apertus-overlap overlay cross-check the immutable revisions in
Phase-04 `sources.json`. The historical FineWeb-Edu, FineWeb-2/HQ, FineMath and
StarCoderData commits are now pinned from retained acquisition-day cache refs,
the StarCoder snapshot inventory and its completed staging log. The config
records the exact evidence paths and hashes. The old payload copies are gone,
so the restaging receipt must still validate every newly acquired byte.

```bash
cd subprojects/05_token_distillation_cpt/05_training_dataset_bridge/clariden
export BRIDGE_RUN_ID=full-corpus-25b-v1

# Preview only.
DRY_RUN=1 ./submit.sh restage

# HF_TOKEN must be in the submitted environment.
# Replacement of an unreceipted skeleton is a separate explicit switch.
DRY_RUN=0 CONFIRM_RESTAGE=PINNED_REPLAY_V1 RESTAGE_REPLACE=1 ./submit.sh restage
```

The dependent old-Greek CPU stage reconstructs `greek_replay.parquet` from the
receipted Nanochat shards and exact overlap overlay. It writes a build receipt
binding all input hashes, implementation bytes, composite identity policy,
counts, and output hash.

## CPU data build on Clariden

Use one immutable run ID and the exact completed Phase-04 stage:

```bash
export PHASE04_STAGE=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/pipeline_runs/<run>/stages/80-materialize-validate

DRY_RUN=1 ./submit.sh freeze
DRY_RUN=0 CONFIRM_BUILD=1 ./submit.sh freeze

# Run after input_receipt.json exists.
DRY_RUN=1 ./submit.sh after-freeze
DRY_RUN=0 CONFIRM_BUILD=1 ./submit.sh after-freeze
```

The dependency graph builds heldouts, training and heldout binary arrays,
performs exact accounting/uniqueness/capacity finalization, and finally freezes
launch assets. Final outputs are `bridge_manifest.json`,
`training_mix_79_20_1.json`, `training_data.env`, and
`training_assets_receipt.json`.

## Receipt-gated 25B launcher

The launcher re-hashes every `.bin`/`.idx`, the complete TD init checkpoint,
the semantic layer-11 roundtrip evidence bundle, the frozen trainer/config, and
the complete clean effective Megatron tree. The evidence must name the same
patched checkpoint, layer 11, parallel geometry, and Megatron commit, with zero
standard/R17/QK/xIELU tensor and logit drift.

```bash
export BRIDGE_STAGE_ROOT=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/training_bridge/full-corpus-25b-v1
DRY_RUN=1 ../train/submit_25b_probe.sh
DRY_RUN=0 CONFIRM_GPU_LAUNCH=FULL_CORPUS_TD_25B ../train/submit_25b_probe.sh
```

One invocation submits only one segment. It never assumes that a future
checkpoint exists merely because a Slurm dependency was registered. After an
intermediate segment completes, submit the CPU
`70_freeze_resume_checkpoint.sbatch` stage with `PROBE_OUTPUT_DIR` and
`COMPLETED_ITERATION`. Then relaunch with the exact
`START_ITERATION` and `RESUME_CHECKPOINT_RECEIPT`. A missing, altered,
wrong-iteration, or already-submitted segment fails closed.

## Current external prerequisites

- Run the receipt-bound restaging and old-Greek build on Clariden CPU nodes.
- Restore the full ModernGreek-148480 tokenizer tree.
- Restore a clean Swiss-AI Megatron checkout at commit `c92402e...`; the
  restaging config can replace the current skeleton only through the explicit
  replacement gate.
- Restore the complete patched TD layer-11 checkpoint at the exact path named
  by its tracked roundtrip manifest. The current candidate checkpoint
  directories are empty skeletons.
- Produce a passed Phase-04 private release and retain its query/manifests.

If unique capacity is insufficient, finalization writes
`bridge_capacity_failure.json` and stops. The remedies are more data from the
same reviewed immutable source or an explicit recipe review—never silent reuse
or duplication.

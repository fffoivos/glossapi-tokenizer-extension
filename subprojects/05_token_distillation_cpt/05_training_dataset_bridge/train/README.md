# 05/train — the receipt-gated 25B launcher

> **In one line:** a two-file GPU launch path for the frozen 25B Token-Distillation probe that re-verifies every byte it depends on twice; it was never used to launch a run.
> **Period:** 2026-07-12. **Status:** superseded by [`../../06_25b_midtraining_probe`](../../06_25b_midtraining_probe).
> **Came from / led to:** consumes the bridge outputs from [`../clariden`](../clariden) → would have driven `03_apertus_extension_and_embedding_adaptation/.../bakeoff_training/bakeoff_train.sbatch`.

## What it does

`submit_25b_probe.sh` refuses to run without `BRIDGE_STAGE_ROOT` pointing at a finalised bridge run, defaults to a dry run, and needs `CONFIRM_GPU_LAUNCH=FULL_CORPUS_TD_25B` to submit. Before submission it re-hashes every `.bin`/`.idx`, the complete TD init checkpoint, the semantic layer-11 roundtrip evidence bundle, both training environments, the trainer, the runtime wrapper, the launcher itself and the entire clean effective Megatron tree — including the whole tokenizer tree receipt frozen at bridge-input time. The evidence must name the same patched checkpoint, layer 11, parallel geometry and Megatron commit with zero standard/R17/QK/xIELU tensor and logit drift.

The same verifier then runs **again at job start** inside the pinned uenv, before either training environment is sourced; `full_corpus_25b.env` sets `FULL_CPT_REQUIRE_JOB_START_VERIFY=1` so the config refuses to train without that hook. Only after both shared environments are sourced does the file overwrite and assert the complete effective recipe — mock data off, exact uenv/TP/PP, optimizer and overlap, warmup start, seed and order, loss, geometry and batch, tokenizer and data roots, validation cadence, resume and exit boundary. Legacy bakeoff jobs share `common_cpt.env` but do not opt in, so they keep their existing path.

One invocation submits one segment only. It never assumes a future checkpoint exists because a Slurm dependency was registered: after an intermediate segment completes, `../clariden/70_freeze_resume_checkpoint.sbatch` must freeze that checkpoint, and the relaunch must pass the exact `START_ITERATION` and `RESUME_CHECKPOINT_RECEIPT`. A missing, altered, wrong-iteration or already-submitted segment fails closed.

## Outcome

- No launch was made from here. [`../../06_25b_midtraining_probe/README.md`](../../06_25b_midtraining_probe/README.md) describes this as the "stale single-blend launch path" it replaces, and its own `clariden/train_segment.sbatch` adds the phase-relative data-index fix that converts a checkpoint's global consumed-sample count on every segment.
- The verify-twice pattern — once on the login node, once at job start inside the pinned runtime — carried forward into the probe and into [`../../../07_full_8b_cpt`](../../../07_full_8b_cpt).

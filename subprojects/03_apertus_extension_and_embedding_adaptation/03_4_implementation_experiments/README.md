# 03_4 — Implementation experiments

> **In one line:** where the plans became Slurm jobs — the Clariden reconnaissance, the 4-arm init bakeoff and its whole eval stack, and (much later) the polytonic cutoff probe that picked the production tokenizer.
> **Period:** 2026-05-20 (`af438d4d`) → 2026-08-21 (`7dce0efb`). The bakeoff itself ran 2026-05-21 → 2026-05-26.
> **Status:** the bakeoff completed 2026-05-26 and was never resumed; the polytonic decision closed 2026-07-29; the training and eval scripts stayed in service for subprojects 05–08 until 2026-08-21.
> **Came from / led to:** [`../03_3_cscs_experiments_kickoff/`](../03_3_cscs_experiments_kickoff/README.md) → this → [`../../04_cpt_training_regime_on_vanilla/`](../../04_cpt_training_regime_on_vanilla/) and [`../../05_token_distillation_cpt/`](../../05_token_distillation_cpt/).

## Why this existed

Everything upstream of here is a document. This is where a real cluster, a real 8 B model and a real corpus meet: what partitions exist, how fast a GH200 node is, whether the HF→Megatron conversion is lossless, whether four different ways of initialising 17,408 embedding rows produce measurably different models, and how to measure that fairly when two of the arms use a different tokenizer from the other two.

## History

**2026-05-20 — reconnaissance.** Three probe documents were written from live Clariden queries before any job was submitted:

- [`AUTH_AND_NODE_FINDING.md`](AUTH_AND_NODE_FINDING.md) — cert validity, partition/QoS probe, expected start times, and the recommended job shape (1 node × 4 GH200, partition `normal`, 12 h cap) that every bakeoff arm then used.
- [`STORAGE_AND_EXISTING_WORK.md`](STORAGE_AND_EXISTING_WORK.md) — the filesystem map, the "no Eiger access, use `xfer` for CPU work" finding, and the headline discovery that **another member of the same `a0140` project (p-skarvelis) had been running Apertus-Greek CPT + SFT since 2026-04-17** on HF Trainer + FineWeb-2-HQ at seq 2048. Their setup was tokenizer-incompatible and was explicitly **not** adopted as the scaffold, but their measured 6,702 tok/s/GPU calibrated the throughput estimates.
- [`ENVIRONMENT_AND_BENCHMARKS.md`](ENVIRONMENT_AND_BENCHMARKS.md) — inventory of swiss-ai training/eval repos, Apertus's own reported eval set, the ILSP Greek suite, and the staging plan for getting all of it onto Clariden scratch.

**2026-05-20 → 2026-05-26 — the bakeoff.** [`init_bakeoff/`](init_bakeoff/README.md) is the substance of this directory: init arms, corpus build, Megatron patches, training, eval, TD challenger, production launcher. Its own README carries the run-by-run history.

**2026-06 → 2026-08 — the scripts outlive the experiment.** `init_bakeoff/bakeoff_training/bakeoff_train.sbatch` and `init_bakeoff/eval/run_native_greek_mcq_eval.py` kept being extended by later subprojects (curriculum sweeps, full-8B CPT, targeted experiments) without any new experiment running here.

**2026-07-29 — the polytonic cutoff.** [`polytonic_cutoff_probe/`](polytonic_cutoff_probe/README.md) reopened the polytonic question that the bakeoff had dropped, with a pre-committed gate this time, and selected **+512 merges → vocab 148,992**.

## Outcome

- A 4-arm bakeoff run to 5 B tokens, with a full evidence tree (per-iteration lm-eval JSONs, tokenizer-fair intrinsics, new-token diagnostics, training logs, plots) preserved under `init_bakeoff/eval/trajectory_analysis_20260524/`.
- A reusable Clariden stack: an Apertus HF→Megatron loader with an R17 patcher, a parameterised trainer with chained-job walltime handoff, a checkpoint→HF→eval sidecar bridge, a tokenizer-fair metric computer, and a native-Greek MCQ runner.
- The production tokenizer (148,992) used by the full-8B run.
- What did **not** happen here: the 15–20 B production CPT. `init_bakeoff/production_cpt/` is dry-run validated only.

## Sub-subprojects

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`init_bakeoff/`](init_bakeoff/README.md) | The 4-arm init experiment, end to end | 2026-05-20 → 2026-08-21 | experiment completed 2026-05-26; scripts reused after | TD leads downstream at 5 B, Vanilla leads BPB and native Greek MCQ, Apertus-Base leads everything on native Greek |
| [`polytonic_cutoff_probe/`](polytonic_cutoff_probe/README.md) | Pre-committed choice between +512 and +1,024 polytonic merges | 2026-07-29 (scripts committed 2026-07-31, `f0dc31a0`) | completed | **+512 selected**; vocab 148,992 frozen for the production corpus |

## Where things are

| What | Where |
|---|---|
| Job-shape and partition findings | [`AUTH_AND_NODE_FINDING.md`](AUTH_AND_NODE_FINDING.md) |
| Clariden filesystem map + the p-skarvelis baseline | [`STORAGE_AND_EXISTING_WORK.md`](STORAGE_AND_EXISTING_WORK.md) |
| Repo/benchmark inventory and staging plan | [`ENVIRONMENT_AND_BENCHMARKS.md`](ENVIRONMENT_AND_BENCHMARKS.md) |
| CPU-only Slurm guard (keeps dataset work off GPU partitions) | [`init_bakeoff/check_cpu_only_slurm.sh`](init_bakeoff/check_cpu_only_slurm.sh), [`init_bakeoff/slurm_cpu_only_guard.sh`](init_bakeoff/slurm_cpu_only_guard.sh) |

## Working documents

All three top-level docs are 2026-05-20 reconnaissance snapshots and carry "v0.7 supersedes this" banners on framing they got wrong at the time:

- [`STORAGE_AND_EXISTING_WORK.md`](STORAGE_AND_EXISTING_WORK.md) — §3.4's "adopt p-skarvelis's pipeline" recommendation was withdrawn; read §3 as *what they did*, not what was done here.
- [`ENVIRONMENT_AND_BENCHMARKS.md`](ENVIRONMENT_AND_BENCHMARKS.md) — §1.1 lists `apertus-finetuning-recipes` as the likely trunk; it is not, Megatron-LM-Swiss-AI is. Its tokenizer-staging section names the composite 153,600 bundle, which the bakeoff did not use.
- [`AUTH_AND_NODE_FINDING.md`](AUTH_AND_NODE_FINDING.md) — the cert window and queue numbers are from 2026-05-20 and have long drifted; the methodology and the chosen job shape are what survived.

The old README's "(planned)" sections — `01_vanilla_calibration_v1/`, `02_pilot_runs/`, `sbatch_templates/`, `job_log.jsonl` — were never created; that work happened inside `init_bakeoff/` instead.

# bakeoff_training — the trainer

> **In one line:** one parameterised Megatron-LM-Swiss-AI sbatch that trained all four arms under identical settings, plus the chained submitters that worked around Clariden's 12 h walltime cap; it outlived the bakeoff and became the trainer for subprojects 05–08.
> **Period:** 2026-05-21 → 2026-08-21 (`7dce0efb`). Bakeoff runs: 2026-05-22 → 2026-05-26. **Status:** bakeoff completed; the sbatch is still live shared infrastructure.
> **Reads:** [`../arms/`](../arms/README.md) + [`../megatron_patches/`](../megatron_patches/README.md) + [`../corpus_build/`](../corpus_build/README.md). **Feeds:** [`../eval/`](../eval/README.md).

## Why this existed

For the comparison to isolate initialisation, everything else has to be byte-identical across arms: optimizer, schedule, sequence length, batch shape, dataloader seed, document stream. One config file (`_train_config_common.env`) and one sbatch make that auditable; the per-arm switch chooses only the init checkpoint, the tokenizer, and the matching Megatron data prefix.

## What was held constant

AdEMAMix at Apertus's pretraining β1/β2/β3/α and weight decay 0.1; gradient clip 0.1 global-norm; sequence length 4,096; global batch ~4.19 M tokens; bf16 with fp32 master grads; cross-document attention mask ON; EoD loss mask ON; xIELU + QK-Norm inherited from the R17-patched checkpoints; NTP loss (Goldfish deferred to production); shared dataloader seed. The three deliberate divergences from Apertus pretraining were peak LR **1.5e-5** (≈14 % of pretrain's 1.1e-4, standard for CPT), AdEMAMix α/β3 warmup **238 steps** (50 % of the 477-step bakeoff, because Apertus's 2.8 %-of-run policy collapses to ~14 steps at this scale), and NTP instead of Goldfish. Two further deviations were forced by hardware: microbatch 2 (not 4, GH200 memory, global-batch tokens preserved) and a fixed global batch with no ramp. Apertus's own optimizer state was **not** loaded — the first 1–2 % of each run acts as optimizer-state warmup. Full table in [`../../../CPT_MASTER_20260526.md`](../../../CPT_MASTER_20260526.md) §3.

## History

| Date | Run | What happened | Evidence |
|---|---|---|---|
| 2026-05-21 | — | Q D1 resolved: `swiss-ai/Megatron-LM` pinned at `c92402e3`; sbatch flag names corrected against `submit_apertus_8b.sh` (`--xielu`, `--ademamix-beta3-warmup`, `--ademamix-alpha-warmup`) | `_archive/2026-05-24_2B_bakeoff_review/AUDIT_FINDINGS.md` §A |
| 2026-05-22 | `bakeoff_1node_chain_20260522_005620` | Vanilla / ReTok / Centroid to iter 476 (~2.0 B tokens), one node × 4 GH200, `normal`, chained across walltime boundaries; `torch_dist` resume-metadata fallback fixed mid-run (`56c594f6`) | `9cc53d2a`, `4941a983`, `b572f90f` |
| 2026-05-23 | `smoke_td_layer11_2357596` | Bounded load/train smoke on the TD layer-11 R17-patched checkpoint — **passed** | [`smoke_td_layer11_2357596/`](smoke_td_layer11_2357596/README.md) |
| 2026-05-23 | `smoke_td_layer11_2node_2357684` | Two-node efficiency smoke — **failed before iteration 1** with NCCL/OFI `NO_SPACE`. The one-node path was kept for everything afterwards, including the production launcher | [`smoke_td_layer11_2node_2357684/`](smoke_td_layer11_2node_2357684/README.md) |
| 2026-05-23 → 05-24 | `td_full25_layer11_2b_20260523T165038Z` | The TD arm's own 2 B chained run to iter 476 | `5d5c4613`, `6c95ff1a` |
| 2026-05-24 → 05-25 | `continuation_3p5b_20260524T143012Z` | Three arms × three chained segments (476→585→715→834, ~3.5 B), dry-run-first with an explicit `CONFIRM_3P5B_LAUNCH` cost gate, eval sidecars submitted as dependencies of each checkpoint-producing segment | [`dryrun_3p5b_continuation_20260524T020000Z/`](dryrun_3p5b_continuation_20260524T020000Z/README.md), `f1f1bf3c`, `c2db4e5b` |
| 2026-05-25 → 05-26 | `continuation_5b_td_vs_vanilla_20260525T142522Z` | Vanilla + TD only, 834 → 1013 → 1192 (~5.0 B). Jobs `2382982`–`2382985`. Babysat by a systemd monitor on 600 s polls plus a finalizer that collected iter-1192 artifacts and regenerated the summary; **no restart or manual intervention was needed** | [`RUN_LOG_5B_TD_VS_VANILLA_20260525.md`](RUN_LOG_5B_TD_VS_VANILLA_20260525.md), `e320d8d0`, `bae88b55` |
| 2026-06-10 → 2026-08-21 | — | The sbatch keeps being extended for other subprojects: Clariden CXI force-flush launch path (`102ac8a6`), physical-order curriculum (`d06b1ac4`), curriculum sweeps v2 (`cfdd0e7b`), receipt-bound full-corpus pipeline (`76a44479`, `3d063bfd`), receipt-gated two-phase preparation (`8dbb6d25`), torchrun agent count in nested allocations (`f85599b0`), preserved srun uenv launch path (`7dce0efb`) | commit log |

## Outcome

- 21 training jobs across the 2 B / 3.5 B / 5 B stages, with all Megatron stdout preserved under `../eval/trajectory_analysis_20260524/per_iter_results/training_logs/`.
- **V3 confirmed**: dataloader state survived every resume; both continuations picked up cleanly.
- One-node is the proven path; the two-node route was never made to work here.
- The chained-submission pattern (dry-run first, explicit launch confirmation, `afterok` dependencies, more chain jobs than expected runtime) became the house style for every later CPT run.

## Where things are

| What | Where |
|---|---|
| The constants | [`_train_config_common.env`](_train_config_common.env) |
| The trainer | [`bakeoff_train.sbatch`](bakeoff_train.sbatch) — takes `ARM` + `INIT_CKPT` + `OUTPUT_DIR` |
| Submitters | [`submit_all_arms.sh`](submit_all_arms.sh), [`submit_td_layer11_2b_chain.sh`](submit_td_layer11_2b_chain.sh), [`submit_3p5b_continuation_chain.sh`](submit_3p5b_continuation_chain.sh), [`submit_5b_td_vs_vanilla_chain.sh`](submit_5b_td_vs_vanilla_chain.sh), smoke wrappers |
| Data preprocessing job | [`preprocess_data.sbatch`](preprocess_data.sbatch) (CPU, `xfer`; run once per tokenizer family) |
| Log parsing | [`summarize_training_logs.py`](summarize_training_logs.py), [`monitor_5b_td_vs_vanilla_status.sh`](monitor_5b_td_vs_vanilla_status.sh) |
| Checkpoints | Clariden `/capstor/scratch/cscs/fffoivos/runs/bakeoff/` (~5.1 TB) |

**Loss-reading rule:** raw Megatron `lm loss` is per-token CE and is not comparable across the 131,072-vocab and 148,480-vocab arms. Selection uses heldout BPB — see [`../eval/LOSS_MEASUREMENT_POLICY.md`](../eval/LOSS_MEASUREMENT_POLICY.md).

## Working documents

- [`RUN_LOG_5B_TD_VS_VANILLA_20260525.md`](RUN_LOG_5B_TD_VS_VANILLA_20260525.md) — 37 KB of poll-by-poll monitoring of the final continuation. Mostly repeated health checks; read the last section for the finalizer outcome.
- Per-run audit dirs: [`smoke_td_layer11_2357596/`](smoke_td_layer11_2357596/), [`smoke_td_layer11_2node_2357684/`](smoke_td_layer11_2node_2357684/), [`dryrun_3p5b_continuation_20260524T020000Z/`](dryrun_3p5b_continuation_20260524T020000Z/) — job ids, `sacct` output, checkpoint listings. Historical receipts.

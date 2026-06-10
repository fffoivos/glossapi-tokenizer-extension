# Greek Apertus CPT Runbook

This is the canonical operator runbook for the two-arm 13.5B Greek CPT launch.
For policy rationale, use `ARCHIVE.md` and `LOG.md`; for exact live values, the
single source of truth is the env config under `03_training_experiments/configs/`.

## Current State

- Dataset, ordered Stage-C stream, base/ext Megatron binaries, held-out
  validation binaries, init checkpoints, TE runtime guard, per-set validation
  patch, and artifact gate are ready on Clariden.
- Launch-scale CXI is validated at 16 nodes / 64 GPUs per arm with
  `NCCL_NET_FORCE_FLUSH=0`; do not use the earlier Socket fallback unless CXI
  regresses.
- Expected allocated runtime is 8.3-8.5h per arm, excluding Slurm queue wait and
  sidecar benchmark jobs.
- The two arms are independent and normally run in parallel. Parallel launch
  requests 32 nodes / 128 GPUs total. Serial launch roughly doubles wall time.

## Fixed Paths

```bash
SC=/iopsstor/scratch/cscs/fffoivos
REPO=$SC/repo/glossapi-tokenizer-extension
SUB=$REPO/subprojects/05_token_distillation_cpt
EXP=$SUB/03_training_experiments
RUN_ROOT=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b
```

Use Clariden account `a0140`. Training runs on `normal`; CPU dataset work should
run without GPU/GRES. The current software image is `pytorch/v2.9.1:v2`.

## Experiment Arms

| Setting | Vanilla arm | Modern-Greek TD arm |
|---|---|---|
| Submit arg | `vanilla` | `td` |
| Tokenizer | `$SC/models/apertus-8b-2509` | `$SC/tokenizers/apertus_greek_modern_only_148480` |
| Vocab | 131,072 | 148,480 = 131,072 + 17,408; divisible by 256 |
| Init checkpoint | `$SC/init_checkpoints/cpt_2arm_13b/vanilla_base131072/megatron_tp2_r17patched` | `$SC/init_checkpoints/cpt_2arm_13b/modern_greek_td148480/megatron_tp2_r17patched` |
| Data prefix | `$SC/cpt_corpus/cpt_2arm_13b/megatron/bulk_mix_ordered_replay_fixed_base_text_document` | `$SC/cpt_corpus/cpt_2arm_13b/megatron/bulk_mix_ordered_replay_fixed_ext_text_document` |

The TD arm uses layer-11 Token Distillation for both input embeddings `E`
(hidden-state MSE) and output/lm-head rows `U` (CE), with original rows frozen
during initialization. The training run itself is otherwise identical to the
vanilla arm.

## Training Settings

Hyperparameter policy reference: `../CURRENT_HYPERPARAMETERS.md` v1.0. Anything
not explicitly changed is intended to match Apertus-8B pretraining.

The exact live settings are in `configs/common_cpt.env`:

- Base model: `swiss-ai/Apertus-8B-2509` main, converted to Megatron TP=2/PP=1
  and R17-patched.
- Geometry: sequence length 4096, max positions 4096, RoPE base 500000, RoPE
  scaling enabled with factor 8. Preserve Apertus main-pretraining
  `rope_scaling`; only revert the long-context `rope_theta`/context length.
- Batch: microbatch 2, global batch 1024, global batch tokens 4,194,304.
- Precision: bf16 with fp32 main grads.
- Optimizer: AdEMAMix, beta1 0.9, beta2 0.995, beta3 0.999, alpha 4.0,
  weight decay 0.1, grad clip 0.1, init std 0.008944.
- LR: WSD, peak 5.5e-5, final 5.5e-6, warmup init 5.5e-6.
- Warmup: `2/(1-beta2)` = 400 iterations = about 1.678B tokens.
- Cooldown: final 20% of the full 13.5B run, `1-sqrt` decay.
- AdEMAMix beta3/alpha warmup: full run, `TRAIN_ITERS`.
- Loss: Goldfish, k=50 and h=50.
- Cross-document behavior: reset attention mask, reset position IDs, EOD loss
  mask.
- Parallelism: TP=2, PP=1, 16 nodes per arm, 4 GPUs per node,
  `LAUNCH_MODE=torchrun`.
- Save/eval: save every 119 iters; held-out validation every 25 iters with
  `EVAL_ITERS=1`.

Do not export shell overrides for LR, optimizer, tokenizer, data, or checkpoint
paths unless `ALLOW_OVERRIDES=1` is intentional and recorded.

## Dataset Recipe

Production is the full 13.5B-token run, not the earlier 5B diagnostics.

- New Greek target: 10B tokens.
- New Greek split: 70% HPLT Greek, then 30% OpenArchives/GlossAPI.
- Replay, measured as a share of new Greek: 24% multilingual replay, 4% code,
  2% math, 5% Greek replay.
- Total stream: 13.5B tokens.
- Held-out validation sets: 0.5B each from HPLT, OpenArchives, and Greek PhD;
  their doc ids are excluded from training.
- Stage-C ordering: replay/code/math/Greek-replay positions stay fixed; only
  the new-Greek subsequence is rewritten so HPLT comes before OpenArchives.
- Stage-A cleaning/decontamination: HPLT confident-only E001 clean, then
  GreekMMLU `correct_only` decontamination only.
- Stage-B: anonymize after decontamination, then preprocess.
- Stage-C: ordered replay-fixed preprocess, then base/ext tokenization.

The committed recipe file is
`03_training_experiments/dataset_build/bulk_13b.json`.

## Prelaunch Gate

Run this on Clariden before any live launch:

```bash
cd $EXP
bash scripts/gate_cpt2arm_artifacts.sh
```

The gate checks the regime invariants, tokenizer divisibility, checkpoint files,
TE guard, CXI runtime settings, per-set validation patch, held-out binaries,
ordered replay-fixed manifest, and both full training binaries.

## Launch

Dry-run first:

```bash
cd $EXP
DRY_RUN=1 SUBMIT_WATCHERS=0 bash scripts/launch_all.sh
```

Live launch both arms in parallel:

```bash
cd $EXP
DRY_RUN=0 CONFIRM_LAUNCH=1 bash scripts/launch_all.sh
```

Per-arm production defaults:

- `NODES=16`
- `GPUS_PER_NODE=4`
- `LAUNCH_MODE=torchrun`
- `EXIT_INTERVAL=952`
- `N_SEGMENTS=4`
- `SEGMENT_TIME_LIMIT=12:00:00`
- `SAVE_INTERVAL=119`
- `EVAL_INTERVAL=25`
- `NCCL_NET=AWS Libfabric`
- `NCCL_NET_FORCE_FLUSH=0`

Watchers submit benchmark sidecars every 238 iterations plus final. If `xfer`
is unavailable, use `SUBMIT_WATCHERS=0` for training launch and submit/watch
benchmarks separately.

## Monitoring

```bash
squeue -u fffoivos -o "%.18i %.9P %.40j %.8T %.10M %.6D %R"

python3 $EXP/scripts/collect_metrics.py \
  --arm vanilla:"$RUN_ROOT/cpt13b_vanilla_*/*.out" \
  --arm td:"$RUN_ROOT/cpt13b_td_*/*.out" \
  --out "$RUN_ROOT/metrics_latest.csv"
```

Expected timing at 16 nodes per arm:

- Iteration 1: about 15.5s.
- Steady iterations 2-10: mean about 8.63s.
- Three-set held-out validation event: about 11s.
- Checkpoint save event: about 22s.
- Total per arm: about 8.3-8.5h allocated runtime.

## Validated Diagnostics

- Job `2515665`: 16-node mock-data CXI no-flush smoke passed.
- Job `2515841`: 16-node real-data timing passed.
- Job `2515891`: 16-node real-data per-set validation smoke passed.
- Job `2515966`: 16-node checkpoint-save smoke passed.

The root cause of the previous multi-node `NET/OFI ... NO_SPACE` failure was
the trainer-forced `NCCL_NET_FORCE_FLUSH=1`. Production must keep
`NCCL_NET_FORCE_FLUSH=0`.

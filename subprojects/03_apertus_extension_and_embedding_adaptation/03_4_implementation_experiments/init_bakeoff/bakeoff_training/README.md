# Bakeoff training — Megatron-LM-Swiss-AI sbatch templates

The three arms (Vanilla / ReTok / Centroid) train under **identical** conditions
on Clariden (one node, 4× GH200, partition `normal`, 12 h cap). They differ
only in which init checkpoint they load.

The training engine is **Megatron-LM-Swiss-AI** (Apertus's pretraining fork),
not HuggingFace Trainer — for Apertus-fidelity reasons documented in
[`../../TRAINING_RECIPE.md`](../../TRAINING_RECIPE.md).

## Files

| File | Purpose |
|---|---|
| [`README.md`](README.md) | this file |
| [`_train_config_common.env`](_train_config_common.env) | sourced by all sbatch jobs — AdEMAMix hyperparams, fidelity flags, Megatron CLI args |
| [`preprocess_data.sbatch`](preprocess_data.sbatch) | one-time CPU job (`xfer` partition) — `tools/preprocess_data.py` JSONL → Megatron `.bin/.idx` |
| [`bakeoff_train.sbatch`](bakeoff_train.sbatch) | parameterized training job, takes `ARM` + `INIT_CKPT` + `OUTPUT_DIR` |
| [`submit_all_arms.sh`](submit_all_arms.sh) | thin wrapper: submits all three arms in parallel with a shared seed |

## End-to-end sequence

```
[once, before any arm]
  sbatch preprocess_data.sbatch        # ~2-4 h on xfer; CPU-only
                                       #   in:  /iopsstor/.../bulk_mix.jsonl
                                       #   out: /iopsstor/.../bulk_mix_megatron/{bin,idx}

[once, before any arm]
  python3 ../arms/build_init_checkpoints.py \
      --arms vanilla retok centroid \
      --vocab-size 148480 \
      --out-root /iopsstor/.../init_checkpoints
                                       # ~30 min on a 1-GPU debug allocation
                                       # produces vanilla/  retok/  centroid/  HF-format dirs
                                       # then converts each to Megatron-LM-Swiss-AI format
                                       # via swiss-ai/hfconverter (gated on Q D1)

[the bakeoff]
  bash submit_all_arms.sh              # submits 3 × sbatch bakeoff_train.sbatch
                                       # each: 1 node, 4 × GH200, ~11 h, 2 B tokens
```

## What's same across arms (the constants)

Documented authoritatively in `_train_config_common.env`:

- AdEMAMix (β1, β2, β3, α, weight_decay) — Apertus pretraining values
- Gradient clipping: 0.1 global-norm
- LR schedule: WSD with re-warmup (~1-2 % of tokens), peak LR (per cpt_plan §3.3), final LR
- Sequence length: 4,096
- Global batch: ~4 M tokens (target Apertus pretraining shape)
- Goldfish loss: **disabled for the bakeoff** (NTP — per v0.7 §10 Q B4)
- xIELU activation + QK-Norm: inherited from base checkpoint
- Cross-doc attention mask: ON
- EoD loss mask: ON
- Mixed precision: bf16
- Dataloader seed: shared across arms (token streams identical → only init differs)

## What differs across arms

Only `--load <init-checkpoint>`. The init differential is built upstream
by `arms/build_init_checkpoints.py`. After that point, every flag, every
data shard, every seed is identical.

## Q D1 status

The exact Megatron-LM-Swiss-AI fork branch / commit is the open dependency
([`../../cpt_plan_v0.7_status.md`](../../cpt_plan_v0.7_status.md)
Q D1). Placeholders below assume the Apertus pretraining branch; the
substitution is one-line once confirmed.

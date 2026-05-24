# Bakeoff training — Megatron-LM-Swiss-AI sbatch templates

The three arms (Vanilla / ReTok / Centroid) train under **identical** conditions
on Clariden (one node, 4× GH200, partition `normal`, 12 h cap). They differ
only in which init checkpoint they load.

The training engine is **Megatron-LM-Swiss-AI** (Apertus's pretraining fork),
not HuggingFace Trainer — for Apertus-fidelity reasons documented in
[`../../../TRAINING_RECIPE.md`](../../../TRAINING_RECIPE.md).

## Files

| File | Purpose |
|---|---|
| [`README.md`](README.md) | this file |
| [`_train_config_common.env`](_train_config_common.env) | sourced by all sbatch jobs — AdEMAMix hyperparams, fidelity flags, Megatron CLI args |
| [`preprocess_data.sbatch`](preprocess_data.sbatch) | one-time CPU job (`xfer` partition) — `tools/preprocess_data.py` JSONL → Megatron `.bin/.idx` |
| [`bakeoff_train.sbatch`](bakeoff_train.sbatch) | parameterized training job, takes `ARM` + `INIT_CKPT` + `OUTPUT_DIR` |
| [`submit_all_arms.sh`](submit_all_arms.sh) | thin wrapper: submits all three arms in parallel with a shared seed |
| [`submit_td_layer11_smoke.sh`](submit_td_layer11_smoke.sh) | bounded load/train smoke for the selected TD layer11 R17-patched checkpoint |
| [`submit_td_layer11_2b_chain.sh`](submit_td_layer11_2b_chain.sh) | one-arm chained 2B TD layer11 training run; uses a resume job because `normal` is capped at 12h |
| [`submit_3p5b_continuation_chain.sh`](submit_3p5b_continuation_chain.sh) | dry-run-first Vanilla/ReTok/TD continuation from iter 476 to iter 834, with sidecar eval dependencies |

## End-to-end sequence

```
[twice, before any arm] # one Megatron binary per tokenizer family
  sbatch --export=ALL,TOKENIZER_DIR=$BASE_TOKENIZER_DIR,OUTPUT_PREFIX=$BASE_DATA_PREFIX \
      preprocess_data.sbatch           # Vanilla data: base 131,072 tokenizer
  sbatch --export=ALL,TOKENIZER_DIR=$EXT_TOKENIZER_DIR,OUTPUT_PREFIX=$EXT_DATA_PREFIX \
      preprocess_data.sbatch           # ReTok/Centroid data: extended 148,480 tokenizer
                                       # ~2-4 h each on xfer; CPU-only
                                       # Both binaries from the same bulk_mix.jsonl —
                                       # only the tokenization differs (reviewer round-2 Blocker 2).

[once, before any arm]
  bash ../arms/submit_init_pipeline.sh # CPU-only chain on xfer
                                       # produces vanilla/  retok/  centroid/  HF-format dirs
                                       # then converts each to Megatron-LM-Swiss-AI format
                                       # via tools/checkpoint/convert.py --loader apertus_hf

[the bakeoff]
  bash submit_all_arms.sh              # submits 3 × sbatch bakeoff_train.sbatch
                                       # Vanilla loads $BASE_DATA_PREFIX,
                                       # ReTok/Centroid load $EXT_DATA_PREFIX.
                                       # each: 1 node, 4 × GH200, ~11 h, 2 B tokens
```

## 3.5B continuation bakeoff

After the 2B bakeoff, the continuation question is Vanilla vs ReTok vs TD
layer11 from iter 476 to iter 834 (~3.5B total tokens). The continuation
submitter is dry-run-first:

```bash
DRY_RUN=1 RUN_TAG=continuation_3p5b_review bash submit_3p5b_continuation_chain.sh
```

Live submission requires an explicit cost-event confirmation:

```bash
DRY_RUN=0 CONFIRM_3P5B_LAUNCH=1 RUN_TAG=continuation_3p5b_<stamp> \
  bash submit_3p5b_continuation_chain.sh
```

The script submits three parallel arms and three chained segments per arm:

- iter 476 -> 585 (~2.45B total tokens)
- iter 585 -> 715 (~3.00B total tokens)
- iter 715 -> 834 (~3.50B total tokens)

It writes `training_chain.tsv` plus the exact sbatch commands under
`/capstor/scratch/cscs/fffoivos/runs/bakeoff/${RUN_TAG}_submit_state/`.
By default it also calls
[`../eval/submit_3p5b_eval_sidecars.sh`](../eval/submit_3p5b_eval_sidecars.sh),
which submits conversion, intrinsic, diagnostics, and packed downstream eval
jobs that depend on the checkpoint-producing training segment. Later training
segments do not depend on eval jobs.

## What's same across arms (the constants)

Documented authoritatively in `_train_config_common.env`:

- AdEMAMix (β1, β2, β3, α, weight_decay) — Apertus pretraining values
- Gradient clipping: 0.1 global-norm
- LR schedule: WSD with re-warmup (~1-2 % of tokens), peak LR (per cpt_plan §3.3), final LR
- Sequence length: 4,096
- Global batch: ~4 M tokens (target Apertus pretraining shape)
- Goldfish loss: **disabled for the bakeoff** (NTP — per v0.7 §10 Q B4)
- xIELU activation + QK-Norm: same converted-Megatron defaults across arms unless the production R17 patcher is implemented
- Cross-doc attention mask: ON
- EoD loss mask: ON
- Mixed precision: bf16
- Dataloader seed: shared across arms; text stream is identical, while Vanilla uses base token IDs and ReTok/Centroid use extended token IDs

## What differs across arms

The per-arm switch chooses the init checkpoint, tokenizer, and matching
Megatron data prefix. Vanilla uses the base 131,072-token tokenizer/data;
ReTok and Centroid use the extended 148,480-token tokenizer/data. The
underlying JSONL document stream and seed are shared.

## Q D1 status

Resolved 2026-05-21: swiss-ai/Megatron-LM main HEAD pinned at
`c92402e39ef3c8e69ea378a59e79059dc14541f4`. See [`../../../TRAINING_RECIPE.md`](../../../TRAINING_RECIPE.md) §1.

## HF → Megatron conversion (the bridge between init checkpoint build and training)

`build_init_checkpoints.py` produces HF-format model checkpoints (one per
arm). To train them in Megatron-LM, they have to be converted to Megatron
format. The conversion uses our custom Apertus loader:

```bash
# Once per Clariden setup, after cloning swiss-ai/Megatron-LM:
bash ../megatron_patches/install.sh $MEGATRON_LM_DIR

# Per init checkpoint:
cd $MEGATRON_LM_DIR
python3 tools/checkpoint/convert.py \
    --loader apertus_hf \
    --saver core \
    --load-dir   /iopsstor/.../init_checkpoints/<arm>/hf \
    --save-dir   /iopsstor/.../init_checkpoints/<arm>/megatron \
    --tokenizer-model /iopsstor/.../tokenizers/apertus_greek_modern_only_148480 \
    --bf16
```

See [`../megatron_patches/README.md`](../megatron_patches/README.md) for the
full conversion + roundtrip-validation procedure. The roundtrip on
unmodified Apertus-8B-2509 should run **before** the first bakeoff sbatch
submission as a one-time correctness gate on our loader.

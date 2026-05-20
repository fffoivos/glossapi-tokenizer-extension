#!/usr/bin/env bash
#SBATCH --job-name=bakeoff_train
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/bakeoff/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/bakeoff/%x-%j.err
#
# Parameterized bakeoff training job — Megatron-LM-Swiss-AI on the Greek
# CPT corpus. One arm per submission.
#
# All flags below mirror swiss-ai/pretrain-code/pretraining/submit_apertus_8b.sh
# (the canonical "what was run" for Apertus pretraining) — see TRAINING_RECIPE.md
# §1 for the deviation list (CPT-specific: LR peak, token budget, NTP not Goldfish,
# shorter α/β3 warmup proportional to the bakeoff horizon).
#
# Required env vars (export via --export=...):
#   ARM         — vanilla | retok | centroid (label only; for run name)
#   INIT_CKPT   — Megatron-format init checkpoint path
#   OUTPUT_DIR  — where to save checkpoints + logs
#
# All other config is in _train_config_common.env. Submit via submit_all_arms.sh.

set -euo pipefail
source "$(dirname "$0")/_train_config_common.env"

ARM="${ARM:?ARM is required (vanilla|retok|centroid)}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT path is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR path is required}"

mkdir -p "$OUTPUT_DIR"

# Derived sizes
TRAIN_SAMPLES=$(( TRAIN_TOKENS / SEQ_LENGTH ))
LR_WARMUP_SAMPLES=$(( LR_WARMUP_TOKENS / SEQ_LENGTH ))

echo "=== bakeoff_train.sbatch ==="
date -u
echo "host: $(hostname)"
echo "ARM:         $ARM"
echo "INIT_CKPT:   $INIT_CKPT"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "MEGATRON:    $MEGATRON_LM_SWISSAI_DIR (commit $MEGATRON_LM_SWISSAI_COMMIT)"
echo "uenv:        $UENV_IMAGE"
echo
echo "config snapshot (vs. Apertus pretraining; cite paper §2.3 + Table C.4):"
echo "  SEQ_LENGTH=$SEQ_LENGTH                    [Apertus: 4096]"
echo "  GLOBAL_BATCH=$GLOBAL_BATCH_SIZE samples (= $GLOBAL_BATCH_TOKENS tokens)  [Apertus: 1024→2048 samples]"
echo "  MICROBATCH=$MICROBATCH_SIZE per GPU       [Apertus: 4]"
echo "  TRAIN_TOKENS=$TRAIN_TOKENS  (TRAIN_SAMPLES=$TRAIN_SAMPLES, ~$TRAIN_ITERS iters)"
echo "  LR peak=$LR_PEAK final=$LR_FINAL warmup_samples=$LR_WARMUP_SAMPLES decay=$LR_DECAY_STYLE"
echo "    [Apertus: peak 1.1e-4 / final 1.1e-5 / warmup 16.78BT / decay 1-sqrt]"
echo "  GRAD_CLIP=$GRAD_CLIP                      [Apertus: 0.1 — load-bearing for norm convergence]"
echo "  AdEMAMix β1=$ADEMA_BETA1 β2=$ADEMA_BETA2 β3=$ADEMA_BETA3 α=$ADEMA_ALPHA wd=$ADEMA_WEIGHT_DECAY"
echo "    α/β3 warmup_steps=$ADEMA_ALPHA_WARMUP_STEPS,$ADEMA_BETA3_WARMUP_STEPS  [Apertus pretrain: 100,000 both]"
echo "  LOSS=$LOSS_OBJECTIVE                      [Apertus: Goldfish k=h=50; bakeoff intentionally NTP per v0.7 §10 Q B4]"
echo "  bf16+fp32mg=$MIXED_PRECISION_FLAG,$MAIN_GRADS_DTYPE  [Apertus: same — FP8 was rolled back at 8T]"
echo "  Activation=$ACTIVATION_FLAG (αp=αn=$XIELU_INIT_AP, β=0.5)  [Apertus: xIELU]"
echo "  QK-Norm=$QK_LAYERNORM_FLAG impl=$QKNORM_IMPL  [Apertus: per-head RMSNorm before RoPE]"
echo "  Cross-doc/EoD=$CROSS_DOC_ATTN_FLAGS $EOD_LOSS_MASK_FLAG  [Apertus: both ON]"
echo "  DATA_SEED=$DATA_SEED                      [shared across all 3 arms — token streams identical]"
echo "  TRAIN_DATA_PREFIX=$TRAIN_DATA_PREFIX"
echo

# Record run metadata for the audit trail
cat > "$OUTPUT_DIR/run_metadata.json" <<META
{
  "arm": "$ARM",
  "init_ckpt": "$INIT_CKPT",
  "output_dir": "$OUTPUT_DIR",
  "megatron_dir": "$MEGATRON_LM_SWISSAI_DIR",
  "megatron_commit": "$MEGATRON_LM_SWISSAI_COMMIT",
  "seq_length": $SEQ_LENGTH,
  "global_batch_size_samples": $GLOBAL_BATCH_SIZE,
  "global_batch_tokens": $GLOBAL_BATCH_TOKENS,
  "microbatch_size": $MICROBATCH_SIZE,
  "train_tokens": $TRAIN_TOKENS,
  "train_samples": $TRAIN_SAMPLES,
  "train_iters_estimate": $TRAIN_ITERS,
  "lr_peak": "$LR_PEAK",
  "lr_final": "$LR_FINAL",
  "lr_warmup_samples": $LR_WARMUP_SAMPLES,
  "lr_decay_style": "$LR_DECAY_STYLE",
  "grad_clip": $GRAD_CLIP,
  "optimizer": "ademamix",
  "ademamix_beta1": $ADEMA_BETA1,
  "ademamix_beta2": $ADEMA_BETA2,
  "ademamix_beta3": $ADEMA_BETA3,
  "ademamix_alpha": $ADEMA_ALPHA,
  "ademamix_weight_decay": $ADEMA_WEIGHT_DECAY,
  "ademamix_beta3_warmup_steps": $ADEMA_BETA3_WARMUP_STEPS,
  "ademamix_alpha_warmup_steps": $ADEMA_ALPHA_WARMUP_STEPS,
  "init_method_std": $INIT_STD,
  "loss_objective": "$LOSS_OBJECTIVE",
  "activation": "xielu",
  "xielu_init_ap": $XIELU_INIT_AP,
  "xielu_init_an": $XIELU_INIT_AN,
  "qk_layernorm": true,
  "qknorm_impl": "$QKNORM_IMPL",
  "normalization": "$NORMALIZATION",
  "cross_doc_attention_mask": true,
  "eod_loss_mask": true,
  "mixed_precision": "bf16",
  "main_grads_dtype": "$MAIN_GRADS_DTYPE",
  "data_seed": $DATA_SEED,
  "data_prefix": "$TRAIN_DATA_PREFIX",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META

# ===== Megatron-LM-Swiss-AI launch =====
# Flag set mirrors swiss-ai/pretrain-code/pretraining/submit_apertus_8b.sh.
# Where the bakeoff diverges from Apertus pretraining, the divergence is documented
# inline. Every flag name was verified against the actual sbatch script + Megatron
# argument plumbing (megatron/training/arguments.py).

uenv run "$UENV_IMAGE" --view=default -- torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NODES \
    --node_rank=0 \
    --master_addr=$(hostname) \
    --master_port=29500 \
    "$MEGATRON_LM_SWISSAI_DIR/pretrain_gpt.py" \
    \
    `# === model load (init checkpoint from build_init_checkpoints.py → hfconverter) ===` \
    --load "$INIT_CKPT" \
    --use-checkpoint-args \
    --no-load-optim \
    --no-load-rng \
    \
    `# === sequence + batch ===` \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings 4096 \
    --micro-batch-size $MICROBATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    \
    `# === optimizer (AdEMAMix; values from paper Table C.4 + sbatch L207-213) ===` \
    --optimizer ademamix \
    --adam-beta1 $ADEMA_BETA1 \
    --adam-beta2 $ADEMA_BETA2 \
    --ademamix-beta3 $ADEMA_BETA3 \
    --ademamix-alpha $ADEMA_ALPHA \
    --ademamix-beta3-warmup-steps $ADEMA_BETA3_WARMUP_STEPS \
    --ademamix-alpha-warmup-steps $ADEMA_ALPHA_WARMUP_STEPS \
    --weight-decay $ADEMA_WEIGHT_DECAY \
    --clip-grad $GRAD_CLIP \
    --init-method-std $INIT_STD \
    \
    `# === LR schedule (WSD with 1-sqrt cooldown — Apertus shape) ===` \
    --lr $LR_PEAK \
    --min-lr $LR_FINAL \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --lr-decay-style WSD \
    --lr-wsd-decay-style $LR_DECAY_STYLE \
    --lr-wsd-decay-samples $TRAIN_SAMPLES \
    \
    `# === training duration ===` \
    --train-samples $TRAIN_SAMPLES \
    \
    `# === mixed precision (bf16 + fp32 master grads; FP8 rolled back per paper Appendix D) ===` \
    $MIXED_PRECISION_FLAG \
    --main-grads-dtype $MAIN_GRADS_DTYPE \
    \
    `# === Apertus arch flags ===` \
    --normalization $NORMALIZATION \
    $QK_LAYERNORM_FLAG \
    --qknorm-impl $QKNORM_IMPL \
    --untie-embeddings-and-output-weights \
    --no-bias-gelu-fusion \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-base 500000 \
    \
    `# === xIELU activation (αp_init = αn_init = 0.8, β = 0.5; code-canonical) ===` \
    $ACTIVATION_FLAG \
    \
    `# === document boundaries (Apertus: both ON) ===` \
    $CROSS_DOC_ATTN_FLAGS \
    $EOD_LOSS_MASK_FLAG \
    \
    `# === data ===` \
    --data-path $TRAIN_DATA_PREFIX \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480" \
    --seed $DATA_SEED \
    --num-workers $DATALOADER_WORKERS \
    \
    `# === checkpointing + eval ===` \
    --save "$OUTPUT_DIR/checkpoints" \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 10 \
    \
    `# === logging ===` \
    --log-interval 10 \
    --tensorboard-dir "$OUTPUT_DIR/tensorboard" \
    \
    `# === distributed ===` \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --use-distributed-optimizer

echo
echo "=== bakeoff arm $ARM done ==="
date -u
echo "checkpoints under: $OUTPUT_DIR/checkpoints"
echo "tensorboard:       $OUTPUT_DIR/tensorboard"
echo
echo "Next: bash ../eval/run_bakeoff_arm_eval.sh $OUTPUT_DIR/checkpoints/<latest>"

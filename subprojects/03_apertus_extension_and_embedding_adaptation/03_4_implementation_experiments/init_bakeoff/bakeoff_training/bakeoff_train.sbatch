#!/usr/bin/env bash
#SBATCH --job-name=bakeoff_train
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR2@600
#SBATCH --no-requeue
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/bakeoff/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/bakeoff/%x-%j.err
#
# Parameterized bakeoff training job — Megatron-LM-Swiss-AI on the Greek CPT
# corpus. One arm per submission.
#
# ============================================================================
# Every flag below is byte-for-byte mirrored from
#   references/repos/swiss-ai_pretrain-code/pretraining/submit_apertus_8b.sh
#   commit 531cc8be2f76064127cad99a61019f985a7c7ee2
# except for the explicit CPT-specific deviations marked [CPT-DIFF: …].
# ============================================================================
#
# Required env vars (export via --export=...):
#   ARM         — vanilla | retok | centroid (label only; for run name)
#   INIT_CKPT   — Megatron-format init checkpoint path
#   OUTPUT_DIR  — where to save checkpoints + logs

set -euo pipefail
source "$(dirname "$0")/_train_config_common.env"

ARM="${ARM:?ARM is required (vanilla|retok|centroid)}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT path is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR path is required}"

mkdir -p "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/tensorboard" "$OUTPUT_DIR/triggers"
TRIGGER_DIR="$OUTPUT_DIR/triggers"
SEQ_LEN="$SEQ_LENGTH"   # alias to match Apertus sbatch var naming

TRAIN_SAMPLES=$(( TRAIN_TOKENS / SEQ_LEN ))
LR_WARMUP_SAMPLES=$(( LR_WARMUP_TOKENS / SEQ_LEN ))

echo "=== bakeoff_train.sbatch ==="
date -u
echo "host: $(hostname)"
echo "ARM=$ARM  INIT_CKPT=$INIT_CKPT  OUTPUT_DIR=$OUTPUT_DIR"
echo "Megatron-LM-Swiss-AI: $MEGATRON_LM_SWISSAI_DIR @ $MEGATRON_LM_SWISSAI_COMMIT"
echo "uenv: $UENV_IMAGE"
echo "seq_len=$SEQ_LEN  microbatch=$MICROBATCH_SIZE  global_batch_samples=$GLOBAL_BATCH_SIZE"
echo "train_samples=$TRAIN_SAMPLES (train_tokens=$TRAIN_TOKENS) lr_warmup_samples=$LR_WARMUP_SAMPLES"
echo

# Audit-trail run_metadata
cat > "$OUTPUT_DIR/run_metadata.json" <<META
{
  "arm": "$ARM",
  "init_ckpt": "$INIT_CKPT",
  "output_dir": "$OUTPUT_DIR",
  "megatron_dir": "$MEGATRON_LM_SWISSAI_DIR",
  "megatron_commit": "$MEGATRON_LM_SWISSAI_COMMIT",
  "seq_length": $SEQ_LEN,
  "global_batch_samples": $GLOBAL_BATCH_SIZE,
  "global_batch_tokens": $GLOBAL_BATCH_TOKENS,
  "microbatch_size": $MICROBATCH_SIZE,
  "train_tokens": $TRAIN_TOKENS,
  "train_samples": $TRAIN_SAMPLES,
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
  "ademamix_beta3_warmup": $ADEMA_BETA3_WARMUP_STEPS,
  "ademamix_alpha_warmup": $ADEMA_ALPHA_WARMUP_STEPS,
  "init_method_std": $INIT_STD,
  "loss_objective": "$LOSS_OBJECTIVE",
  "data_seed": $DATA_SEED,
  "data_prefix": "$TRAIN_DATA_PREFIX",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META

# ===== Flag groups (mirror Apertus sbatch shape) =====

TRANSFORMER_ENGINE_ARGS=(
    --main-grads-dtype $MAIN_GRADS_DTYPE
)

# Network architecture explicitly declared (matches Apertus sbatch L181-199).
# --max-position-embeddings is the pretrain value 4096 (not the 64K long-context
# extension; Apertus does the 64K extension as a separate post-pretrain phase).
NETWORK_SIZE_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 21504
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization $NORMALIZATION
    --xielu
    --qk-layernorm
    --qknorm-impl $QKNORM_IMPL
    --untie-embeddings-and-output-weights
)

LOGGING_ARGS=(
    --log-throughput
    --log-params-norm
    --tensorboard-dir "$OUTPUT_DIR/tensorboard"
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
    --log-interval 1
)

# AdEMAMix flags exactly as Apertus declares (sbatch L209-220).
# Note flag names: `--ademamix-beta3-warmup` not `--ademamix-beta3-warmup-steps`
# (verified against actual sbatch and against the optimizer impl
#  `megatron/core/optimizer/ademamix.py::__init__(beta3_warmup=…, alpha_warmup=…)`).
REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay $ADEMA_WEIGHT_DECAY
    --clip-grad $GRAD_CLIP
    --adam-beta1 $ADEMA_BETA1
    --adam-beta2 $ADEMA_BETA2
    --ademamix-alpha $ADEMA_ALPHA
    --ademamix-beta3 $ADEMA_BETA3
    --ademamix-beta3-warmup $ADEMA_BETA3_WARMUP_STEPS
    --ademamix-alpha-warmup $ADEMA_ALPHA_WARMUP_STEPS
)

TRAINING_ARGS=(
    --micro-batch-size $MICROBATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --no-check-for-nan-in-loss-and-grad
    --train-samples $TRAIN_SAMPLES
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer ademamix
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 500
    --exit-signal-handler
    --trigger-path "$TRIGGER_DIR"
)

INITIALIZATION_ARGS=(
    --seed $DATA_SEED
    --init-method-std $INIT_STD
)

LEARNING_RATE_ARGS=(
    --lr $LR_PEAK
    --min-lr $LR_FINAL
    --lr-decay-style WSD
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --lr-wsd-decay-style $LR_DECAY_STYLE
    --lr-wsd-decay-samples $TRAIN_SAMPLES
)

# Checkpointing: `--dist-ckpt-strictness assume_ok_unexpected` is critical for
# our CPT — it allows Megatron to load a checkpoint whose embedding shape was
# resized from 131,072 to 148,480 by build_init_checkpoints.py + the
# HF→Megatron conversion (without it Megatron refuses the shape mismatch).
#
# --no-load-optim --no-load-rng: we have model weights from Apertus but NOT
# the pretraining optimizer state, so don't try to restore it.
CHECKPOINTING_ARGS=(
    --load "$INIT_CKPT"
    --save "$OUTPUT_DIR/checkpoints"
    --no-load-optim
    --no-load-rng
    --save-interval $SAVE_INTERVAL
    --ckpt-format torch_dist
    --async-save
    --ckpt-fully-parallel-load
    --dist-ckpt-strictness assume_ok_unexpected
    --override-opt_param-scheduler
    --distributed-timeout-minutes 60
)

MIXED_PRECISION_ARGS=(
    $MIXED_PRECISION_FLAG
)

# Tensor-parallel = 1 (single 4-GPU node, model fits without TP). Apertus's
# 256-node run used TP=2; for our 1-node bakeoff TP=1 is enough.
DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# Tokenizer: HF format, our extended 148,480 bundle (or 131,072 base for Vanilla arm).
TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480"
)

# Data: single mix-prefix (Apertus uses comma-separated weighted datasets;
# we use one bulk_mix that's already weighted by mix_builder.py).
DATA_ARGS=(
    --split 100,0,0
    --seq-length $SEQ_LEN
    $CROSS_DOC_ATTN_FLAGS
    $EOD_LOSS_MASK_FLAG
    --num-workers $DATALOADER_WORKERS
    --num-dataset-builder-threads 4
    --data-path $TRAIN_DATA_PREFIX
)

# Goldfish loss is INTENTIONALLY DISABLED for the bakeoff (NTP only).
# For production CPT, add the three flags below (matching Apertus pretrain L290-292):
#   --goldfish-loss --goldfish-k $GOLDFISH_K --goldfish-h $GOLDFISH_H

# ===== Launch =====
export PYTHONPATH="$MEGATRON_LM_SWISSAI_DIR"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

cd "$MEGATRON_LM_SWISSAI_DIR"

uenv run "$UENV_IMAGE" --view=default -- bash -c "
set -euo pipefail
srun --cpus-per-task \${SLURM_CPUS_PER_TASK:-4} \
    -lu bash -c \"
        RANK=\\\$SLURM_PROCID LOCAL_RANK=\\\$SLURM_LOCALID \
        python3 $MEGATRON_LM_SWISSAI_DIR/pretrain_gpt.py \
            ${TRANSFORMER_ENGINE_ARGS[@]} \
            ${NETWORK_SIZE_ARGS[@]} \
            ${LOGGING_ARGS[@]} \
            ${REGULARIZATION_ARGS[@]} \
            ${TRAINING_ARGS[@]} \
            ${INITIALIZATION_ARGS[@]} \
            ${LEARNING_RATE_ARGS[@]} \
            ${CHECKPOINTING_ARGS[@]} \
            ${MIXED_PRECISION_ARGS[@]} \
            ${DISTRIBUTED_ARGS[@]} \
            ${TOKENIZER_ARGS[@]} \
            ${DATA_ARGS[@]}
    \"
"

echo
echo "=== bakeoff arm $ARM done ==="
date -u
echo "checkpoints: $OUTPUT_DIR/checkpoints"
echo "tensorboard: $OUTPUT_DIR/tensorboard"
echo "Next: bash ../eval/run_bakeoff_arm_eval.sh $OUTPUT_DIR/checkpoints/<latest>"

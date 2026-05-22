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
# Slurm runs a copied script from /var/spool/slurmd; use an explicit script
# directory when submitted via submit_all_arms.sh, with SLURM_SUBMIT_DIR as
# fallback for manual submissions from this directory.
SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
source "$SCRIPT_DIR/_train_config_common.env"

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
MODEL_WORLD_SIZE=$(( ${SLURM_NNODES:-$NODES} * GPUS_PER_NODE ))
echo "slurm: account=${SLURM_JOB_ACCOUNT:-$ACCOUNT} partition=${SLURM_JOB_PARTITION:-$PARTITION} nodes=${SLURM_NNODES:-unknown} ntasks=${SLURM_NTASKS:-unknown} gpus_per_node=$GPUS_PER_NODE time_limit=${TIME_LIMIT} launch_mode=$LAUNCH_MODE model_world_size=$MODEL_WORLD_SIZE"
echo "checkpoint mode: RESUME_TRAINING=$RESUME_TRAINING"
echo "seq_len=$SEQ_LEN  microbatch=$MICROBATCH_SIZE  global_batch_samples=$GLOBAL_BATCH_SIZE"
echo "train_samples=$TRAIN_SAMPLES (train_tokens=$TRAIN_TOKENS) lr_warmup_samples=$LR_WARMUP_SAMPLES"
echo

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
if [ -n "$EXIT_INTERVAL" ]; then
    TRAINING_ARGS+=(--exit-interval "$EXIT_INTERVAL")
fi

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
# Initial CPT load uses model weights from Apertus but NOT the pretraining
# optimizer/RNG state, so it sets --no-load-optim --no-load-rng. Continuation
# loads from the arm's own torch_dist checkpoint directory and must restore
# optimizer/RNG.
CHECKPOINTING_ARGS=(
    --load "$INIT_CKPT"
    --save "$OUTPUT_DIR/checkpoints"
    --save-interval $SAVE_INTERVAL
    --ckpt-format torch_dist
    --async-save
    --ckpt-fully-parallel-load
    --dist-ckpt-strictness assume_ok_unexpected
    --override-opt_param-scheduler
    --distributed-timeout-minutes 60
)
if [ "$RESUME_TRAINING" = "0" ]; then
    CHECKPOINTING_ARGS+=(--no-load-optim --no-load-rng)
elif [ "$RESUME_TRAINING" != "1" ]; then
    echo "ERROR: RESUME_TRAINING=$RESUME_TRAINING not recognized (expected 0|1)" >&2
    exit 2
fi

MIXED_PRECISION_ARGS=(
    $MIXED_PRECISION_FLAG
)

# Tensor-parallel = 2 matches Apertus's pretraining launcher and is required on
# one 4-GPU GH200 node; TP=1 loads the full 8B training state on every GPU and
# OOMs on the first forward step.
DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# Tokenizer + data prefix: per-arm.
# Vanilla → 131,072 base tokenizer + base-tokenized data binary.
# ReTok / Centroid → 148,480 extended tokenizer + extended-tokenized data binary.
# submit_all_arms.sh selects the right pair based on ARM. The two binaries are
# byte-identical document streams (same shuffled mix, same seed) — they differ
# ONLY in how the corpus is tokenized. Apertus-fidelity for Vanilla requires
# the base tokenization; ReTok/Centroid evaluate the extended vocab.
case "$ARM" in
    vanilla)
        ARM_TOKENIZER="$BASE_TOKENIZER_DIR"
        ARM_DATA_PREFIX="$BASE_DATA_PREFIX"
        ;;
    retok|centroid)
        ARM_TOKENIZER="$EXT_TOKENIZER_DIR"
        ARM_DATA_PREFIX="$EXT_DATA_PREFIX"
        ;;
    *)
        echo "ERROR: ARM=$ARM not recognized (expected vanilla|retok|centroid)" >&2; exit 2 ;;
esac
echo "ARM=$ARM  →  tokenizer=$ARM_TOKENIZER  data_prefix=$ARM_DATA_PREFIX"

# Audit-trail run_metadata. This must live after ARM_DATA_PREFIX is selected;
# Slurm runs with `set -u`, and there is no arm-neutral TRAIN_DATA_PREFIX.
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
  "tokenizer": "$ARM_TOKENIZER",
  "data_prefix": "$ARM_DATA_PREFIX",
  "uenv_image": "$UENV_IMAGE",
  "resume_training": $RESUME_TRAINING,
  "slurm_account": "${SLURM_JOB_ACCOUNT:-$ACCOUNT}",
  "slurm_partition": "${SLURM_JOB_PARTITION:-$PARTITION}",
  "slurm_nodes": ${SLURM_NNODES:-0},
  "slurm_ntasks": ${SLURM_NTASKS:-0},
  "slurm_gpus_per_node": $GPUS_PER_NODE,
  "slurm_time_limit": "$TIME_LIMIT",
  "launch_mode": "$LAUNCH_MODE",
  "model_world_size": $MODEL_WORLD_SIZE,
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$ARM_TOKENIZER"
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
    --data-path $ARM_DATA_PREFIX
)

# Goldfish loss is INTENTIONALLY DISABLED for the bakeoff (NTP only).
# For production CPT, add the three flags below (matching Apertus pretrain L290-292):
#   --goldfish-loss --goldfish-k $GOLDFISH_K --goldfish-h $GOLDFISH_H

RUNTIME_PATCH_DIR="$SCRIPT_DIR/../megatron_patches/runtime"
TRAINING_CMD=(
    python3 "$RUNTIME_PATCH_DIR/pretrain_gpt_te_guard.py"
    "$MEGATRON_LM_SWISSAI_DIR/pretrain_gpt.py"
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${INITIALIZATION_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${CHECKPOINTING_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
)
printf -v TRAINING_CMD_Q '%q ' "${TRAINING_CMD[@]}"
printf '%s\n' "$TRAINING_CMD_Q" > "$OUTPUT_DIR/training_command.sh"

# ===== Launch =====
export PYTHONPATH="$MEGATRON_LM_SWISSAI_DIR"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export FI_CXI_DEFAULT_TX_SIZE=16384
export NCCL_NET_FORCE_FLUSH=1
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
export NCCL_RAS_ENABLE=0
export CUDA_CACHE_DISABLE=1

# torch.distributed env:// needs these before srun. RANK/LOCAL_RANK are set per
# task in the bash -c command below.
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((32000 + (${SLURM_JOB_ID:-0} % 1000)))}"
if [ "$LAUNCH_MODE" = "torchrun" ]; then
    export WORLD_SIZE="$MODEL_WORLD_SIZE"
else
    export WORLD_SIZE="${SLURM_NPROCS:-${SLURM_NTASKS:-4}}"
fi
echo "distributed: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE LAUNCH_MODE=$LAUNCH_MODE"

cd "$MEGATRON_LM_SWISSAI_DIR"

case "$LAUNCH_MODE" in
    slurm)
        uenv run "$UENV_IMAGE" --view=default -- \
            srun --cpus-per-task "${SLURM_CPUS_PER_TASK:-4}" \
                -lu bash -c "export PYTHONPATH='$MEGATRON_LM_SWISSAI_DIR' RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID; exec $TRAINING_CMD_Q"
        ;;
    torchrun)
        uenv run "$UENV_IMAGE" --view=default -- \
            srun --ntasks-per-node=1 --cpus-per-task "${SLURM_CPUS_PER_TASK:-4}" \
                -lu bash -c "export PYTHONPATH='$MEGATRON_LM_SWISSAI_DIR' MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT'; exec torchrun --no-python --nnodes='${SLURM_NNODES:-$NODES}' --nproc_per_node='$GPUS_PER_NODE' --node_rank=\$SLURM_PROCID --rdzv_backend=c10d --rdzv_endpoint='$MASTER_ADDR:$MASTER_PORT' $TRAINING_CMD_Q"
        ;;
    *)
        echo "ERROR: LAUNCH_MODE=$LAUNCH_MODE not recognized (expected slurm|torchrun)" >&2
        exit 2
        ;;
esac

echo
echo "=== bakeoff arm $ARM done ==="
date -u
echo "checkpoints: $OUTPUT_DIR/checkpoints"
echo "tensorboard: $OUTPUT_DIR/tensorboard"
echo "Next: bash ../eval/run_bakeoff_arm_eval.sh $OUTPUT_DIR/checkpoints/<latest>"

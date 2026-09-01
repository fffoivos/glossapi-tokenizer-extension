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
#   ARM         — vanilla | retok | centroid | extension | td
#   INIT_CKPT   — Megatron-format init checkpoint path
#   OUTPUT_DIR  — where to save checkpoints + logs

set -euo pipefail
# Slurm runs a copied script from /var/spool/slurmd; use an explicit script
# directory when submitted via submit_all_arms.sh, with SLURM_SUBMIT_DIR as
# fallback for manual submissions from this directory.
SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
TRAIN_CONFIG="${TRAIN_CONFIG_OVERRIDE:-$SCRIPT_DIR/_train_config_common.env}"
FULL_CPT_JOB_START_VERIFY_COMPLETED=0
case "${FULL_CPT_LAUNCH_VERIFY:-0}" in
    0)
        # Legacy bakeoff submissions do not opt into the full-corpus receipt hook.
        ;;
    1)
        for required in \
            FULL_CPT_VERIFY_UENV FULL_CPT_VERIFY_PYTHON FULL_CPT_VERIFY_SCRIPT \
            FULL_CPT_VERIFY_BRIDGE_MANIFEST FULL_CPT_VERIFY_TRAINING_DATA_ENV \
            FULL_CPT_VERIFY_ASSETS_RECEIPT FULL_CPT_VERIFY_REPO_ROOT \
            FULL_CPT_VERIFY_TRAINING_ENV FULL_CPT_VERIFY_COMMON_TRAINING_ENV \
            FULL_CPT_VERIFY_TRAINER FULL_CPT_VERIFY_RUNTIME_WRAPPER \
            FULL_CPT_VERIFY_LAUNCHER FULL_CPT_VERIFY_EXPECTED_LOAD_CHECKPOINT \
            FULL_CPT_VERIFY_EXPECTED_MEGATRON_DIR \
            FULL_CPT_VERIFY_EXPECTED_TOKENIZER_DIR FULL_CPT_VERIFY_START_ITERATION \
            FULL_CPT_VERIFY_EXPECTED_EXIT_INTERVAL; do
            [[ -n "${!required:-}" ]] || {
                echo "ERROR: full-corpus job-start verification requires $required" >&2
                exit 10
            }
        done
        [[ "$FULL_CPT_VERIFY_TRAINING_ENV" == "$TRAIN_CONFIG" ]] || {
            echo "ERROR: job training config differs from the verified full-corpus config" >&2
            exit 10
        }
        [[ "$FULL_CPT_VERIFY_TRAINER" == "$SCRIPT_DIR/bakeoff_train.sbatch" ]] || {
            echo "ERROR: job trainer differs from the verified full-corpus trainer" >&2
            exit 10
        }
        EXPECTED_RUNTIME_WRAPPER="$(cd "$SCRIPT_DIR/../megatron_patches/runtime" && pwd)/pretrain_gpt_te_guard.py"
        [[ "$FULL_CPT_VERIFY_RUNTIME_WRAPPER" == "$EXPECTED_RUNTIME_WRAPPER" ]] || {
            echo "ERROR: job runtime wrapper differs from the verified full-corpus wrapper" >&2
            exit 10
        }
        [[ "$FULL_CPT_VERIFY_START_ITERATION" =~ ^[0-9]+$ ]] || {
            echo "ERROR: full-corpus verification start iteration must be a nonnegative integer" >&2
            exit 10
        }
        [[ "$FULL_CPT_VERIFY_EXPECTED_EXIT_INTERVAL" =~ ^[0-9]+$ ]] \
            && (( FULL_CPT_VERIFY_EXPECTED_EXIT_INTERVAL > FULL_CPT_VERIFY_START_ITERATION )) || {
            echo "ERROR: full-corpus exit boundary must be after its start boundary" >&2
            exit 10
        }
        verify_args=(
            --bridge-manifest "$FULL_CPT_VERIFY_BRIDGE_MANIFEST"
            --training-data-env "$FULL_CPT_VERIFY_TRAINING_DATA_ENV"
            --training-assets-receipt "$FULL_CPT_VERIFY_ASSETS_RECEIPT"
            --repo-root "$FULL_CPT_VERIFY_REPO_ROOT"
            --training-env "$FULL_CPT_VERIFY_TRAINING_ENV"
            --common-training-env "$FULL_CPT_VERIFY_COMMON_TRAINING_ENV"
            --trainer "$FULL_CPT_VERIFY_TRAINER"
            --runtime-wrapper "$FULL_CPT_VERIFY_RUNTIME_WRAPPER"
            --launcher "$FULL_CPT_VERIFY_LAUNCHER"
            --expected-load-checkpoint "$FULL_CPT_VERIFY_EXPECTED_LOAD_CHECKPOINT"
            --expected-megatron-dir "$FULL_CPT_VERIFY_EXPECTED_MEGATRON_DIR"
            --expected-tokenizer-dir "$FULL_CPT_VERIFY_EXPECTED_TOKENIZER_DIR"
            --start-iteration "$FULL_CPT_VERIFY_START_ITERATION"
        )
        if (( FULL_CPT_VERIFY_START_ITERATION > 0 )); then
            [[ -n "${FULL_CPT_VERIFY_PROBE_PLAN:-}" && -n "${FULL_CPT_VERIFY_RESUME_RECEIPT:-}" ]] || {
                echo "ERROR: resumed full-corpus verification requires probe-plan and checkpoint receipts" >&2
                exit 10
            }
            verify_args+=(
                --probe-plan "$FULL_CPT_VERIFY_PROBE_PLAN"
                --resume-checkpoint-receipt "$FULL_CPT_VERIFY_RESUME_RECEIPT"
            )
        elif [[ -n "${FULL_CPT_VERIFY_PROBE_PLAN:-}" || -n "${FULL_CPT_VERIFY_RESUME_RECEIPT:-}" ]]; then
            echo "ERROR: initial full-corpus verification must not carry resume inputs" >&2
            exit 10
        fi
        command -v uenv >/dev/null || { echo "ERROR: uenv is required for full-corpus verification" >&2; exit 10; }
        uenv run "$FULL_CPT_VERIFY_UENV" --view=default -- \
            "$FULL_CPT_VERIFY_PYTHON" "$FULL_CPT_VERIFY_SCRIPT" "${verify_args[@]}"
        FULL_CPT_JOB_START_VERIFY_COMPLETED=1
        ;;
    *)
        echo "ERROR: FULL_CPT_LAUNCH_VERIFY must be unset/0 or exactly 1" >&2
        exit 10
        ;;
esac
unset FULL_CPT_REQUIRE_JOB_START_VERIFY FULL_CPT_EFFECTIVE_COMMON_TRAINING_ENV
source "$TRAIN_CONFIG"

if [[ "${FULL_CPT_REQUIRE_JOB_START_VERIFY:-0}" == "1" ]]; then
    [[ "$FULL_CPT_JOB_START_VERIFY_COMPLETED" == "1" ]] || {
        echo "ERROR: full-corpus training config requires the job-start receipt verifier" >&2
        exit 10
    }
    [[ "${INIT_CKPT:-}" == "$FULL_CPT_VERIFY_EXPECTED_LOAD_CHECKPOINT" ]] || {
        echo "ERROR: effective init/load checkpoint differs from the verified receipt" >&2
        exit 10
    }
    [[ "${MEGATRON_LM_SWISSAI_DIR:-}" == "$FULL_CPT_VERIFY_EXPECTED_MEGATRON_DIR" ]] || {
        echo "ERROR: effective Megatron root differs from the verified receipt" >&2
        exit 10
    }
    [[ "${BRIDGE_DATA_ENV:-}" == "$FULL_CPT_VERIFY_TRAINING_DATA_ENV" ]] || {
        echo "ERROR: effective bridge data environment differs from the verified receipt" >&2
        exit 10
    }
    [[ "${FULL_CPT_EFFECTIVE_COMMON_TRAINING_ENV:-}" == "$FULL_CPT_VERIFY_COMMON_TRAINING_ENV" ]] || {
        echo "ERROR: effective common training environment differs from the verified receipt" >&2
        exit 10
    }
    [[ "${FULL_CPT_TOKENIZER_DIR:-}" == "$FULL_CPT_VERIFY_EXPECTED_TOKENIZER_DIR" ]] || {
        echo "ERROR: effective tokenizer root differs from the verified receipt" >&2
        exit 10
    }
    declare -F full_cpt_assert_frozen_recipe >/dev/null || {
        echo "ERROR: full-corpus config did not define its frozen effective-recipe contract" >&2
        exit 10
    }
    full_cpt_assert_frozen_recipe || exit 10
    [[ "${UENV_IMAGE:-}" == "$FULL_CPT_VERIFY_UENV" ]] || {
        echo "ERROR: effective training uenv differs from the verified job-start uenv" >&2
        exit 10
    }
    expected_resume=0
    (( FULL_CPT_VERIFY_START_ITERATION == 0 )) || expected_resume=1
    [[ "${RESUME_TRAINING:-}" == "$expected_resume" ]] || {
        echo "ERROR: effective resume mode disagrees with the verified start boundary" >&2
        exit 10
    }
    [[ "${EXIT_INTERVAL:-}" == "$FULL_CPT_VERIFY_EXPECTED_EXIT_INTERVAL" ]] || {
        echo "ERROR: effective exit boundary differs from the submitted segment" >&2
        exit 10
    }
fi

ARM="${ARM:?ARM is required (vanilla|retok|centroid|extension|td)}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT path is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR path is required}"

mkdir -p "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/tensorboard" "$OUTPUT_DIR/triggers"
TRIGGER_DIR="$OUTPUT_DIR/triggers"
if [ -e "$TRIGGER_DIR/exit" ] || [ -e "$TRIGGER_DIR/save" ]; then
    echo "clearing stale trigger files in $TRIGGER_DIR"
    rm -f "$TRIGGER_DIR/exit" "$TRIGGER_DIR/save"
fi
SEQ_LEN="$SEQ_LENGTH"   # alias to match Apertus sbatch var naming

TRAIN_SAMPLES=$(( TRAIN_TOKENS / SEQ_LEN ))
TRAIN_ITERS=$(( TRAIN_SAMPLES / GLOBAL_BATCH_SIZE ))
LR_WARMUP_SAMPLES=$(( LR_WARMUP_TOKENS / SEQ_LEN ))

echo "=== bakeoff_train.sbatch ==="
date -u
echo "host: $(hostname)"
echo "ARM=$ARM  INIT_CKPT=$INIT_CKPT  OUTPUT_DIR=$OUTPUT_DIR"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"
echo "Megatron-LM-Swiss-AI: $MEGATRON_LM_SWISSAI_DIR @ $MEGATRON_LM_SWISSAI_COMMIT"
echo "uenv: $UENV_IMAGE"
MODEL_WORLD_SIZE=$(( ${SLURM_NNODES:-$NODES} * GPUS_PER_NODE ))
echo "slurm: account=${SLURM_JOB_ACCOUNT:-$ACCOUNT} partition=${SLURM_JOB_PARTITION:-$PARTITION} nodes=${SLURM_NNODES:-unknown} ntasks=${SLURM_NTASKS:-unknown} gpus_per_node=$GPUS_PER_NODE time_limit=${TIME_LIMIT} launch_mode=$LAUNCH_MODE model_world_size=$MODEL_WORLD_SIZE"
echo "checkpoint mode: RESUME_TRAINING=$RESUME_TRAINING"
echo "save mode: DISABLE_SAVE=$DISABLE_SAVE SAVE_INTERVAL=$SAVE_INTERVAL"
echo "loss objective: $LOSS_OBJECTIVE"
echo "data mode: USE_MOCK_DATA=${USE_MOCK_DATA:-0}"
echo "curriculum order mode: ${CURRICULUM_ORDER_MODE:-randomized}"
echo "seq_len=$SEQ_LEN  microbatch=$MICROBATCH_SIZE  global_batch_samples=$GLOBAL_BATCH_SIZE"
echo "train_samples=$TRAIN_SAMPLES (train_tokens=$TRAIN_TOKENS) lr_warmup_samples=$LR_WARMUP_SAMPLES"
echo

# ===== Flag groups (mirror Apertus sbatch shape) =====

TRANSFORMER_ENGINE_ARGS=(
    --main-grads-dtype $MAIN_GRADS_DTYPE
)

# Network architecture explicitly declared (matches Apertus sbatch L181-199).
# Positional geometry is env-var driven. The wrapper keeps legacy defaults for
# old bakeoff reproducibility; current 03_training_experiments configs set the
# Apertus main-pretraining geometry explicitly:
#   MAX_POSITION_EMBEDDINGS=4096 ROTARY_BASE=500000
#   USE_ROPE_SCALING=1 ROPE_SCALING_FACTOR=8.0
# The released HF `main` checkpoint carries post-long-context rope 12M / seq
# 65536, so CPT init checkpoints must be geometry-reverted before training.
NETWORK_SIZE_ARGS=(
    --num-layers ${MODEL_NUM_LAYERS:-32}
    --hidden-size ${MODEL_HIDDEN_SIZE:-4096}
    --ffn-hidden-size ${MODEL_FFN_HIDDEN_SIZE:-21504}
    --num-attention-heads ${MODEL_NUM_ATTENTION_HEADS:-32}
    --group-query-attention
    --num-query-groups ${MODEL_NUM_QUERY_GROUPS:-8}
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS:-$SEQ_LEN}
    --position-embedding-type rope
    --rotary-base ${ROTARY_BASE:-500000}
    --make-vocab-size-divisible-by ${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-128}
    --normalization $NORMALIZATION
    --xielu
    --qk-layernorm
    --qknorm-impl $QKNORM_IMPL
    --untie-embeddings-and-output-weights
)
if [ "${USE_ROPE_SCALING:-0}" = "1" ]; then
    NETWORK_SIZE_ARGS+=(
        --use-rope-scaling
        --rope-scaling-factor ${ROPE_SCALING_FACTOR:-8.0}
    )
    echo "geometry override: USE_ROPE_SCALING=1 ROPE_SCALING_FACTOR=${ROPE_SCALING_FACTOR:-8.0}"
fi
echo "positional geometry: max_position=${MAX_POSITION_EMBEDDINGS:-$SEQ_LEN} rotary_base=${ROTARY_BASE:-500000} use_rope_scaling=${USE_ROPE_SCALING:-0}"
echo "vocab alignment: make_vocab_size_divisible_by=${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-128}"

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

LR_SCHEDULE_STYLE="${LR_SCHEDULE_STYLE:-WSD}"
LR_WARMUP_INIT="${LR_WARMUP_INIT:-0.0}"
LEARNING_RATE_ARGS=(
    --lr $LR_PEAK
    --min-lr $LR_FINAL
    --lr-decay-style $LR_SCHEDULE_STYLE
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --lr-warmup-init $LR_WARMUP_INIT
)
if [ "$LR_SCHEDULE_STYLE" = "WSD" ]; then
    # Cooldown window. Default = whole run (legacy bakeoff behavior, preserved
    # for reproducibility). A proper WSD trapezoid needs warmup + STABLE +
    # decay<100%; Apertus sets a SEPARATE cooldown count
    # (submit_apertus_8b.sh:250 --lr-wsd-decay-samples $COOLDOWN_SAMPLES), not
    # TRAIN_SAMPLES. Set LR_WSD_DECAY_SAMPLES (e.g. 20% of TRAIN_SAMPLES) to
    # get a real stable phase. Backward-compatible: unset => old behavior.
    LEARNING_RATE_ARGS+=(
        --lr-wsd-decay-style $LR_DECAY_STYLE
        --lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES:-$TRAIN_SAMPLES}
    )
fi

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
    --ckpt-format torch_dist
    --ckpt-fully-parallel-load
    --dist-ckpt-strictness assume_ok_unexpected
    --override-opt_param-scheduler
    --distributed-timeout-minutes 60
)
if [ "$DISABLE_SAVE" = "0" ]; then
    CHECKPOINTING_ARGS+=(
        --save "$OUTPUT_DIR/checkpoints"
        --save-interval $SAVE_INTERVAL
        --async-save
    )
elif [ "$DISABLE_SAVE" != "1" ]; then
    echo "ERROR: DISABLE_SAVE=$DISABLE_SAVE not recognized (expected 0|1)" >&2
    exit 2
fi
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
)
if [ "${USE_DISTRIBUTED_OPTIMIZER:-1}" = "1" ]; then
    DISTRIBUTED_ARGS+=(--use-distributed-optimizer)
elif [ "${USE_DISTRIBUTED_OPTIMIZER:-1}" != "0" ]; then
    echo "ERROR: USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-} not recognized (expected 0|1)" >&2
    exit 2
fi
if [ "${USE_COMM_OVERLAP:-1}" = "1" ]; then
    DISTRIBUTED_ARGS+=(
        --overlap-grad-reduce
        --overlap-param-gather
    )
elif [ "${USE_COMM_OVERLAP:-1}" != "0" ]; then
    echo "ERROR: USE_COMM_OVERLAP=${USE_COMM_OVERLAP:-} not recognized (expected 0|1)" >&2
    exit 2
fi
if [ -n "${DDP_BUCKET_SIZE:-}" ]; then
    DISTRIBUTED_ARGS+=(--ddp-bucket-size "$DDP_BUCKET_SIZE")
fi
echo "distributed flags: use_distributed_optimizer=${USE_DISTRIBUTED_OPTIMIZER:-1} use_comm_overlap=${USE_COMM_OVERLAP:-1} ddp_bucket_size=${DDP_BUCKET_SIZE:-default}"

# Tokenizer + data prefix: per-arm.
# Vanilla → 131,072 base tokenizer + base-tokenized data binary.
# ReTok / Centroid / Extension / TD → 148,480 extended tokenizer + extended-tokenized data binary.
# submit_all_arms.sh selects the right pair based on ARM. The two binaries are
# byte-identical document streams (same shuffled mix, same seed) — they differ
# ONLY in how the corpus is tokenized. Apertus-fidelity for Vanilla requires
# the base tokenization; ReTok/Centroid evaluate the extended vocab.
case "$ARM" in
    vanilla)
        ARM_TOKENIZER="$BASE_TOKENIZER_DIR"
        ARM_DATA_PREFIX="$BASE_DATA_PREFIX"
        ARM_VAL_TOKENIZATION="base"
        ;;
    retok|centroid|extension|td)
        ARM_TOKENIZER="$EXT_TOKENIZER_DIR"
        ARM_DATA_PREFIX="$EXT_DATA_PREFIX"
        ARM_VAL_TOKENIZATION="ext"
        ;;
    *)
        echo "ERROR: ARM=$ARM not recognized (expected vanilla|retok|centroid|extension|td)" >&2; exit 2 ;;
esac
echo "ARM=$ARM  →  tokenizer=$ARM_TOKENIZER  data_prefix=$ARM_DATA_PREFIX"

USE_MOCK_DATA="${USE_MOCK_DATA:-0}"
if [ "$USE_MOCK_DATA" = "1" ]; then
    ENABLE_EXTRA_VALID=0
elif [ "$USE_MOCK_DATA" != "0" ]; then
    echo "ERROR: USE_MOCK_DATA=$USE_MOCK_DATA not recognized (expected 0|1)" >&2
    exit 2
fi

ENABLE_EXTRA_VALID="${ENABLE_EXTRA_VALID:-1}"
VAL_DATA_DIR="${VAL_DATA_DIR:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt_2arm_13b/megatron}"
EXTRA_VALID_SETS="${EXTRA_VALID_SETS:-hplt openarchives greek_phd}"
NEW_GREEK_SETS=" ${NEW_GREEK_VALID_SETS:-hplt openarchives greek_phd} "
EXTRA_VALID_ARGS=()
if [ "$ENABLE_EXTRA_VALID" = "1" ]; then
    EXTRA_VALID_ARGS=(--extra-valid-data-path)
    for name in $EXTRA_VALID_SETS; do
        case "$NEW_GREEK_SETS" in
            *" $name "*) base="val_${name}" ;;
            *) base="val_forget_${name}" ;;
        esac
        prefix="$VAL_DATA_DIR/${base}_${ARM_VAL_TOKENIZATION}_text_document"
        if [ ! -s "${prefix}.idx" ] || [ ! -s "${prefix}.bin" ]; then
            echo "WARN: extra validation binary missing or empty for ARM=$ARM, skipping $name: ${prefix}.{bin,idx}" >&2
            continue
        fi
        EXTRA_VALID_ARGS+=("$name" "$prefix")
    done
    [ "${#EXTRA_VALID_ARGS[@]}" -le 1 ] && EXTRA_VALID_ARGS=()
elif [ "$ENABLE_EXTRA_VALID" != "0" ]; then
    echo "ERROR: ENABLE_EXTRA_VALID=$ENABLE_EXTRA_VALID not recognized (expected 0|1)" >&2
    exit 2
fi
echo "validation: eval_interval=$EVAL_INTERVAL eval_iters=$EVAL_ITERS extra_valid=$ENABLE_EXTRA_VALID tokenization=$ARM_VAL_TOKENIZATION"

# Audit-trail run_metadata. This must live after ARM_DATA_PREFIX is selected;
# Slurm runs with `set -u`, and there is no arm-neutral TRAIN_DATA_PREFIX.
cat > "$OUTPUT_DIR/run_metadata.json" <<META
{
  "arm": "$ARM",
  "init_ckpt": "$INIT_CKPT",
  "output_dir": "$OUTPUT_DIR",
  "megatron_dir": "$MEGATRON_LM_SWISSAI_DIR",
  "megatron_commit": "$MEGATRON_LM_SWISSAI_COMMIT",
  "model_num_layers": ${MODEL_NUM_LAYERS:-32},
  "model_hidden_size": ${MODEL_HIDDEN_SIZE:-4096},
  "model_ffn_hidden_size": ${MODEL_FFN_HIDDEN_SIZE:-21504},
  "model_num_attention_heads": ${MODEL_NUM_ATTENTION_HEADS:-32},
  "model_num_query_groups": ${MODEL_NUM_QUERY_GROUPS:-8},
  "seq_length": $SEQ_LEN,
  "max_position_embeddings": ${MAX_POSITION_EMBEDDINGS:-$SEQ_LEN},
  "rotary_base": ${ROTARY_BASE:-500000},
  "use_rope_scaling": ${USE_ROPE_SCALING:-0},
  "rope_scaling_factor": ${ROPE_SCALING_FACTOR:-8.0},
  "global_batch_samples": $GLOBAL_BATCH_SIZE,
  "global_batch_tokens": $GLOBAL_BATCH_TOKENS,
  "microbatch_size": $MICROBATCH_SIZE,
  "train_tokens": $TRAIN_TOKENS,
  "train_samples": $TRAIN_SAMPLES,
  "train_iters": $TRAIN_ITERS,
  "lr_peak": "$LR_PEAK",
  "lr_final": "$LR_FINAL",
  "lr_schedule_style": "$LR_SCHEDULE_STYLE",
  "lr_warmup_init": "$LR_WARMUP_INIT",
  "lr_warmup_samples": $LR_WARMUP_SAMPLES,
  "lr_wsd_decay_samples": ${LR_WSD_DECAY_SAMPLES:-$TRAIN_SAMPLES},
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
  "curriculum_order_mode": "${CURRICULUM_ORDER_MODE:-randomized}",
  "megatron_gpt_dataset_no_shuffle": "${MEGATRON_GPT_DATASET_NO_SHUFFLE:-0}",
  "data_seed": $DATA_SEED,
  "eval_interval": $EVAL_INTERVAL,
  "eval_iters": $EVAL_ITERS,
  "extra_valid_enabled": $ENABLE_EXTRA_VALID,
  "extra_valid_sets": "$EXTRA_VALID_SETS",
  "extra_valid_tokenization": "$ARM_VAL_TOKENIZATION",
  "extra_valid_data_dir": "$VAL_DATA_DIR",
  "mock_data": $USE_MOCK_DATA,
  "make_vocab_size_divisible_by": ${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-128},
  "tokenizer": "$ARM_TOKENIZER",
  "data_prefix": "$ARM_DATA_PREFIX",
  "uenv_image": "$UENV_IMAGE",
  "resume_training": $RESUME_TRAINING,
  "disable_save": $DISABLE_SAVE,
  "slurm_account": "${SLURM_JOB_ACCOUNT:-$ACCOUNT}",
  "slurm_partition": "${SLURM_JOB_PARTITION:-$PARTITION}",
  "slurm_nodes": ${SLURM_NNODES:-0},
  "slurm_ntasks": ${SLURM_NTASKS:-0},
  "slurm_gpus_per_node": $GPUS_PER_NODE,
  "slurm_time_limit": "$TIME_LIMIT",
  "launch_mode": "$LAUNCH_MODE",
  "model_world_size": $MODEL_WORLD_SIZE,
  "ddp_bucket_size": "${DDP_BUCKET_SIZE:-default}",
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
    --data-cache-path "${H2G_PHASE_CACHE_ROOT:?set frozen phase cache root}"
)
if [ "$USE_MOCK_DATA" = "1" ]; then
    DATA_ARGS+=(--mock-data)
else
    DATA_ARGS+=(--data-path $ARM_DATA_PREFIX)
fi

case "${CURRICULUM_ORDER_MODE:-randomized}" in
    randomized)
        unset MEGATRON_GPT_DATASET_NO_SHUFFLE
        ;;
    physical_order)
        if ! grep -q "MEGATRON_GPT_DATASET_NO_SHUFFLE" "$MEGATRON_LM_SWISSAI_DIR/megatron/core/datasets/gpt_dataset.py"; then
            echo "ERROR: CURRICULUM_ORDER_MODE=physical_order requires the GPTDataset no-shuffle patch in $MEGATRON_LM_SWISSAI_DIR" >&2
            echo "       Run: python3 $SCRIPT_DIR/../../../../05_token_distillation_cpt/03_training_experiments/scripts/patch_megatron_gpt_dataset_no_shuffle.py --megatron-dir $MEGATRON_LM_SWISSAI_DIR" >&2
            exit 6
        fi
        export MEGATRON_GPT_DATASET_NO_SHUFFLE=1
        ;;
    *)
        echo "ERROR: CURRICULUM_ORDER_MODE=${CURRICULUM_ORDER_MODE:-} not recognized (expected randomized|physical_order)" >&2
        exit 2
        ;;
esac

VALIDATION_ARGS=(
    --eval-interval "$EVAL_INTERVAL"
    --eval-iters "$EVAL_ITERS"
    "${EXTRA_VALID_ARGS[@]}"
)

case "$LOSS_OBJECTIVE" in
    ntp)
        # Goldfish loss is intentionally disabled for the bakeoff: NTP keeps
        # initialization quality as the only experimental variable.
        ;;
    goldfish)
        TRAINING_ARGS+=(
            --goldfish-loss
            --goldfish-k $GOLDFISH_K
            --goldfish-h $GOLDFISH_H
        )
        ;;
    *)
        echo "ERROR: LOSS_OBJECTIVE=$LOSS_OBJECTIVE not recognized (expected ntp|goldfish)" >&2
        exit 2
        ;;
esac

RUNTIME_PATCH_DIR="$SCRIPT_DIR/../megatron_patches/runtime"
PRE_WRAPPER=()
[ -n "${TRAINER_WRAPPER:-}" ] && PRE_WRAPPER=("$TRAINER_WRAPPER")
TRAINING_CMD=(
    python3 "${PRE_WRAPPER[@]}" "$RUNTIME_PATCH_DIR/pretrain_gpt_te_guard.py"
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
    "${VALIDATION_ARGS[@]}"
    "${DATA_ARGS[@]}"
)
printf -v TRAINING_CMD_Q '%q ' "${TRAINING_CMD[@]}"
printf '%s\n' "$TRAINING_CMD_Q" > "$OUTPUT_DIR/training_command.sh"

# ===== Launch =====
runtime_pythonpath="$MEGATRON_LM_SWISSAI_DIR"
if [[ -n "${RUNTIME_COMPAT_DIR:-}" ]]; then
    [[ -f "$RUNTIME_COMPAT_DIR/sitecustomize.py" ]] || {
        echo "ERROR: runtime compatibility shim is missing" >&2
        exit 2
    }
    runtime_pythonpath="$RUNTIME_COMPAT_DIR:$runtime_pythonpath"
fi
export PYTHONPATH="$runtime_pythonpath"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# Alps/uenv NCCL setup per CSCS guidance for Slingshot/libfabric. These are
# runtime transport knobs only; they do not change the model or optimizer.
export NCCL_NET="${NCCL_NET:-AWS Libfabric}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_PROTO="${NCCL_PROTO:-^LL128}"
export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"
export FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE:-16384}"
export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER:-1}"
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-software}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
# The AWS OFI NCCL SENDRECV path on Clariden fails with FI_ENOSPC when force
# flush is enabled. Leave the CSCS/default behavior in place unless explicitly
# overridden for diagnostics.
export NCCL_NET_FORCE_FLUSH="${NCCL_NET_FORCE_FLUSH:-0}"
if [ "${FI_CXI_USE_DEFAULT_RDZV:-0}" = "1" ]; then
    unset FI_CXI_RDZV_GET_MIN FI_CXI_RDZV_THRESHOLD FI_CXI_RDZV_EAGER_SIZE
else
    export FI_CXI_RDZV_GET_MIN="${FI_CXI_RDZV_GET_MIN:-0}"
    export FI_CXI_RDZV_THRESHOLD="${FI_CXI_RDZV_THRESHOLD:-0}"
    export FI_CXI_RDZV_EAGER_SIZE="${FI_CXI_RDZV_EAGER_SIZE:-0}"
fi
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD="${FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD:-16777216}"
[ -n "${FI_CXI_DEFAULT_RX_SIZE:-}" ] && export FI_CXI_DEFAULT_RX_SIZE
[ -n "${FI_CXI_REQ_BUF_SIZE:-}" ] && export FI_CXI_REQ_BUF_SIZE
[ -n "${FI_CXI_REQ_BUF_MIN_POSTED:-}" ] && export FI_CXI_REQ_BUF_MIN_POSTED
[ -n "${FI_CXI_OFLOW_BUF_SIZE:-}" ] && export FI_CXI_OFLOW_BUF_SIZE
[ -n "${FI_LOG_LEVEL:-}" ] && export FI_LOG_LEVEL
[ -n "${FI_LOG_PROV:-}" ] && export FI_LOG_PROV
export NCCL_RAS_ENABLE=0
export CUDA_CACHE_DISABLE=1
unset NCCL_NET_PLUGIN
echo "nccl/libfabric: NCCL_NET=$NCCL_NET NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC=$NCCL_CROSS_NIC NCCL_PROTO=$NCCL_PROTO NCCL_NET_FORCE_FLUSH=$NCCL_NET_FORCE_FLUSH OFI_NCCL_PROTOCOL=${OFI_NCCL_PROTOCOL:-<unset>} FI_CXI_DEFAULT_CQ_SIZE=$FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE=$FI_CXI_DEFAULT_TX_SIZE FI_CXI_DEFAULT_RX_SIZE=${FI_CXI_DEFAULT_RX_SIZE:-<unset>} FI_CXI_RX_MATCH_MODE=$FI_CXI_RX_MATCH_MODE FI_CXI_REQ_BUF_SIZE=${FI_CXI_REQ_BUF_SIZE:-<unset>} FI_CXI_REQ_BUF_MIN_POSTED=${FI_CXI_REQ_BUF_MIN_POSTED:-<unset>} FI_CXI_OFLOW_BUF_SIZE=${FI_CXI_OFLOW_BUF_SIZE:-<unset>} FI_CXI_RDZV_GET_MIN=${FI_CXI_RDZV_GET_MIN:-<unset>} FI_CXI_RDZV_THRESHOLD=${FI_CXI_RDZV_THRESHOLD:-<unset>} FI_CXI_RDZV_EAGER_SIZE=${FI_CXI_RDZV_EAGER_SIZE:-<unset>} FI_MR_CACHE_MONITOR=$FI_MR_CACHE_MONITOR FI_LOG_LEVEL=${FI_LOG_LEVEL:-<unset>} FI_LOG_PROV=${FI_LOG_PROV:-<unset>}"

# `uenv run` injects its own NCCL/CXI defaults after the parent shell exports.
# Re-apply our runtime knobs inside the uenv-launched shell so the trainer sees
# the same values the launcher audited above.
RUNTIME_ENV_PREFIX="unset NCCL_NET_PLUGIN; "
if [ "${FI_CXI_USE_DEFAULT_RDZV:-0}" = "1" ]; then
    RUNTIME_ENV_PREFIX+="unset FI_CXI_RDZV_GET_MIN FI_CXI_RDZV_THRESHOLD FI_CXI_RDZV_EAGER_SIZE; "
fi
RUNTIME_ENV_VARS=(
    TORCH_NCCL_AVOID_RECORD_STREAMS
    TORCH_NCCL_ASYNC_ERROR_HANDLING
    CUDA_DEVICE_MAX_CONNECTIONS
    PYTORCH_ALLOC_CONF
    PYTORCH_CUDA_ALLOC_CONF
    OMP_NUM_THREADS
    NCCL_NET
    NCCL_NET_GDR_LEVEL
    NCCL_CROSS_NIC
    NCCL_PROTO
    NCCL_NCHANNELS_PER_NET_PEER
    NCCL_NET_GDR_C2C
    NCCL_NET_GDR_READ
    NCCL_P2P_LEVEL
    NCCL_NET_FORCE_FLUSH
    NCCL_RAS_ENABLE
    NCCL_SOCKET_IFNAME
    NCCL_SOCKET_FAMILY
    NCCL_SOCKET_NTHREADS
    NCCL_NSOCKS_PERTHREAD
    NCCL_DEBUG
    NCCL_DEBUG_SUBSYS
    OFI_NCCL_PROTOCOL
    FI_CXI_DEFAULT_CQ_SIZE
    FI_CXI_DEFAULT_TX_SIZE
    FI_CXI_DEFAULT_RX_SIZE
    FI_CXI_DISABLE_HOST_REGISTER
    FI_CXI_RX_MATCH_MODE
    FI_MR_CACHE_MONITOR
    FI_CXI_RDZV_GET_MIN
    FI_CXI_RDZV_THRESHOLD
    FI_CXI_RDZV_EAGER_SIZE
    FI_CXI_RDZV_PROTO
    FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD
    FI_CXI_REQ_BUF_SIZE
    FI_CXI_REQ_BUF_MIN_POSTED
    FI_CXI_OFLOW_BUF_SIZE
    FI_CXI_OFLOW_BUF_COUNT
    FI_CXI_REQ_BUF_MAX_CACHED
    FI_CXI_HYBRID_PREEMPTIVE
    FI_CXI_HYBRID_RECV_PREEMPTIVE
    FI_CXI_HYBRID_POSTED_RECV_PREEMPTIVE
    FI_CXI_HYBRID_UNEXPECTED_MSG_PREEMPTIVE
    FI_UNIVERSE_SIZE
    FI_LOG_LEVEL
    FI_LOG_PROV
    CUDA_CACHE_DISABLE
    MEGATRON_GPT_DATASET_NO_SHUFFLE
)
for env_name in "${RUNTIME_ENV_VARS[@]}"; do
    if [ -n "${!env_name+x}" ]; then
        printf -v env_value_q '%q' "${!env_name}"
        RUNTIME_ENV_PREFIX+="export $env_name=$env_value_q; "
    fi
done

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
        srun --uenv="$UENV_IMAGE" --view=default --cpus-per-task "${SLURM_CPUS_PER_TASK:-4}" \
                -lu bash -c "$RUNTIME_ENV_PREFIX export PYTHONPATH='$runtime_pythonpath' RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID; exec $TRAINING_CMD_Q"
        ;;
    torchrun)
        srun --uenv="$UENV_IMAGE" --view=default --ntasks-per-node=1 --cpus-per-task "${SLURM_CPUS_PER_TASK:-4}" \
                -lu bash -c "$RUNTIME_ENV_PREFIX export PYTHONPATH='$runtime_pythonpath' MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT'; exec torchrun --no-python --nnodes='${SLURM_NNODES:-$NODES}' --nproc_per_node='$GPUS_PER_NODE' --node_rank=\$SLURM_PROCID --rdzv_backend=c10d --rdzv_endpoint='$MASTER_ADDR:$MASTER_PORT' $TRAINING_CMD_Q"
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

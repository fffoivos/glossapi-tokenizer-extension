#!/usr/bin/env bash
# Submit the Path-A geometry probe — one-shot 0.5 B training under Path A
# geometry (rope_theta=12M, max_pos=65536, llama3 scaling).
#
# Tests the geometry-perturbation hypothesis from
# subprojects/04_cpt_training_regime_on_vanilla/PATH_A_GEOMETRY_PROBE_PLAN.md.
#
# Cost: ~17 GPU-h training + ~5–7 GPU-h sidecar = ~25 GPU-h total.
# Single Slurm submission (not chained). Walltime ~5 h on 1 node / 4 GPUs.
#
# To submit live:
#   DRY_RUN=0 CONFIRM_PATH_A_LAUNCH=1 bash submit_04a_path_a_probe.sh
#
# To dry-run (default, prints sbatch command, doesn't submit):
#   bash submit_04a_path_a_probe.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/04_cpt_training_regime_on_vanilla"
LEGACY_TRAIN_DIR="${LEGACY_TRAIN_DIR:-/iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training}"
TRAIN_SCRIPT="$LEGACY_TRAIN_DIR/bakeoff_train.sbatch"
TRAIN_CONFIG="$SUBPROJECT_DIR/scripts/train_config_04a_path_a.env"
INIT_CKPT="${INIT_CKPT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/04a_vanilla_path_a_probe}"
DATASET_RUN_DIR="${DATASET_RUN_DIR:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/latest_dataset_5b}"
RUN_TAG="${RUN_TAG:-04a_vanilla_path_a_probe_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_TAG}"
STATE_DIR="${STATE_DIR:-$RUN_ROOT/${RUN_TAG}_submit_state}"

DRY_RUN="${DRY_RUN:-1}"
CONFIRM_PATH_A_LAUNCH="${CONFIRM_PATH_A_LAUNCH:-0}"
ACCOUNT="${ACCOUNT:-a0140}"
PARTITION="${PARTITION:-normal}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-36}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

# Scheduler-target = 1.5 B so warmup (1.2 B) is satisfied; actual training
# stops at iter 119 (= 0.5 B) via EXIT_INTERVAL. The TRAIN_TOKENS value is
# notional for the LR scheduler shape; effective training cost = iter 119.
TRAIN_TOKENS="${TRAIN_TOKENS:-1500000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-119}"
EXIT_INTERVAL="${EXIT_INTERVAL:-119}"
EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"
LR_SCHEDULE_STYLE="${LR_SCHEDULE_STYLE:-constant}"
LR_WARMUP_INIT="${LR_WARMUP_INIT:-1.1e-6}"
LR_WARMUP_TOKENS="${LR_WARMUP_TOKENS:-1200000000}"
ADEMA_BETA3_WARMUP_STEPS="${ADEMA_BETA3_WARMUP_STEPS:-287}"
ADEMA_ALPHA_WARMUP_STEPS="${ADEMA_ALPHA_WARMUP_STEPS:-287}"
LOSS_OBJECTIVE="${LOSS_OBJECTIVE:-goldfish}"
DATA_SEED="${DATA_SEED:-20260528}"
LAUNCH_MODE="${LAUNCH_MODE:-slurm}"

case "$DRY_RUN" in
  0|1) ;;
  *) echo "ERROR: DRY_RUN must be 0|1, got $DRY_RUN" >&2; exit 2 ;;
esac
case "$CONFIRM_PATH_A_LAUNCH" in
  0|1) ;;
  *) echo "ERROR: CONFIRM_PATH_A_LAUNCH must be 0|1, got $CONFIRM_PATH_A_LAUNCH" >&2; exit 2 ;;
esac
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_PATH_A_LAUNCH" != "1" ]; then
  echo "ERROR: live launch requires CONFIRM_PATH_A_LAUNCH=1" >&2
  exit 2
fi

test -f "$TRAIN_SCRIPT" || { echo "ERROR: TRAIN_SCRIPT missing: $TRAIN_SCRIPT" >&2; exit 2; }
test -f "$TRAIN_CONFIG" || { echo "ERROR: TRAIN_CONFIG missing: $TRAIN_CONFIG" >&2; exit 2; }
test -d "$INIT_CKPT"    || { echo "ERROR: INIT_CKPT missing: $INIT_CKPT" >&2; exit 2; }
test -d "$DATASET_RUN_DIR" || { echo "ERROR: DATASET_RUN_DIR missing: $DATASET_RUN_DIR" >&2; exit 2; }
test -f "$DATASET_RUN_DIR/dataset_paths.env" || { echo "ERROR: dataset_paths.env missing in $DATASET_RUN_DIR" >&2; exit 2; }

# Source dataset_paths.env to get DATA_PREFIX (Task-1 chain pattern).
source "$DATASET_RUN_DIR/dataset_paths.env"
test -n "${DATA_PREFIX:-}" || { echo "ERROR: DATA_PREFIX not set after sourcing dataset_paths.env" >&2; exit 2; }

mkdir -p "$RUN_ROOT" "$STATE_DIR"

echo "=== submit_04a_path_a_probe.sh ==="
date -u
echo "  RUN_TAG:    $RUN_TAG"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  STATE_DIR:  $STATE_DIR"
echo "  TRAIN_SCRIPT:  $TRAIN_SCRIPT"
echo "  TRAIN_CONFIG:  $TRAIN_CONFIG"
echo "  INIT_CKPT:     $INIT_CKPT"
echo "  DATASET_RUN_DIR: $DATASET_RUN_DIR"
echo "  geometry:   max_pos=65536 rotary_base=12000000 use_rope_scaling=1 factor=8.0"
echo "  schedule:   TRAIN_TOKENS=$TRAIN_TOKENS SAVE_INTERVAL=$SAVE_INTERVAL EXIT_INTERVAL=$EXIT_INTERVAL"
echo "  partition:  $PARTITION nodes=$NODES gpus=$GPUS_PER_NODE time=$TIME_LIMIT"
echo "  DRY_RUN=$DRY_RUN  CONFIRM_PATH_A_LAUNCH=$CONFIRM_PATH_A_LAUNCH"
echo

# Build the sbatch command. Match the Task-1 chain submitter's export shape.
ENV_VARS="ALL"
ENV_VARS+=",ARM=vanilla"
ENV_VARS+=",INIT_CKPT=$INIT_CKPT"
ENV_VARS+=",OUTPUT_DIR=$OUTPUT_DIR"
ENV_VARS+=",SCRIPT_DIR_OVERRIDE=$LEGACY_TRAIN_DIR"
ENV_VARS+=",TRAIN_CONFIG_OVERRIDE=$TRAIN_CONFIG"
ENV_VARS+=",BASE_DATA_PREFIX=$DATA_PREFIX"
ENV_VARS+=",EXT_DATA_PREFIX=$DATA_PREFIX"
ENV_VARS+=",DATASET_RUN_DIR=$DATASET_RUN_DIR"
ENV_VARS+=",RUN_TAG=$RUN_TAG"
ENV_VARS+=",MAX_POSITION_EMBEDDINGS=65536"
ENV_VARS+=",ROTARY_BASE=12000000"
ENV_VARS+=",USE_ROPE_SCALING=1"
ENV_VARS+=",ROPE_SCALING_FACTOR=8.0"
ENV_VARS+=",TRAIN_TOKENS=$TRAIN_TOKENS"
ENV_VARS+=",SAVE_INTERVAL=$SAVE_INTERVAL"
ENV_VARS+=",EXIT_INTERVAL=$EXIT_INTERVAL"
ENV_VARS+=",EVAL_INTERVAL=$EVAL_INTERVAL"
ENV_VARS+=",LR_SCHEDULE_STYLE=$LR_SCHEDULE_STYLE"
ENV_VARS+=",LR_WARMUP_INIT=$LR_WARMUP_INIT"
ENV_VARS+=",LR_WARMUP_TOKENS=$LR_WARMUP_TOKENS"
ENV_VARS+=",ADEMA_BETA3_WARMUP_STEPS=$ADEMA_BETA3_WARMUP_STEPS"
ENV_VARS+=",ADEMA_ALPHA_WARMUP_STEPS=$ADEMA_ALPHA_WARMUP_STEPS"
ENV_VARS+=",LOSS_OBJECTIVE=$LOSS_OBJECTIVE"
ENV_VARS+=",DATA_SEED=$DATA_SEED"
ENV_VARS+=",LAUNCH_MODE=$LAUNCH_MODE"

SBATCH_CMD=(
  sbatch
    --parsable
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes="$NODES"
    --ntasks-per-node="$GPUS_PER_NODE"
    --gpus-per-node="$GPUS_PER_NODE"
    --cpus-per-task="$CPUS_PER_TASK"
    --time="$TIME_LIMIT"
    --job-name="04a_path_a_probe_i119"
    --output="$RUN_ROOT/%x-%j.out"
    --error="$RUN_ROOT/%x-%j.err"
    --export="$ENV_VARS"
    "$TRAIN_SCRIPT"
)

echo "sbatch command:"
printf "  %q " "${SBATCH_CMD[@]}"; echo
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1; not submitting. Set DRY_RUN=0 CONFIRM_PATH_A_LAUNCH=1 to submit live."
  exit 0
fi

# Submit live.
JOB_ID="$("${SBATCH_CMD[@]}")"
echo "Submitted: $JOB_ID"
echo "$JOB_ID" > "$STATE_DIR/training_job_id.txt"

# Write a small paths record for the verifier + sidecar submitter.
cat > "$STATE_DIR/training_path_a_paths.env" <<EOF
RUN_TAG=$RUN_TAG
TRAIN_RUN_DIR=$OUTPUT_DIR
EVAL_ROOT=$RUN_ROOT/eval_${RUN_TAG}
MEGATRON_CKPT_ROOT=$OUTPUT_DIR/checkpoints
INIT_CKPT=$INIT_CKPT
DATASET_RUN_DIR=$DATASET_RUN_DIR
GEOMETRY=path_a
TRAINING_JOB_ID=$JOB_ID
SUBMITTED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo
echo "Next steps:"
echo "  - monitor:    ssh clariden 'squeue -j $JOB_ID'"
echo "  - log:        ssh clariden 'tail -f $RUN_ROOT/04a_path_a_probe_i119-$JOB_ID.out'"
echo "  - on done, sidecar fan-out:"
echo "    bash $SUBPROJECT_DIR/scripts/submit_checkpoint_sidecars.sh \\"
echo "      --train-run-dir $OUTPUT_DIR \\"
echo "      --run-tag $RUN_TAG \\"
echo "      --iteration 119 \\"
echo "      --tokens 499122176 \\"
echo "      --checkpoint-label Vanilla-Path-A-0.5B"

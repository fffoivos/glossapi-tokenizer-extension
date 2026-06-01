#!/usr/bin/env bash
# Submit the Task-1 Vanilla 5B CPT run as checkpoint-bounded Slurm segments.
#
# Clariden normal jobs are capped at 12h. The observed smoke throughput makes a
# single 5B job too long, so this submits a dependency chain whose segment
# targets are all near 0.5B-token boundaries. Later segments can run while eval
# sidecars consume earlier checkpoints.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/04_cpt_training_regime_on_vanilla"
LEGACY_TRAIN_DIR="${LEGACY_TRAIN_DIR:-/iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training}"
TRAIN_SCRIPT="$LEGACY_TRAIN_DIR/bakeoff_train.sbatch"
TRAIN_CONFIG="$SUBPROJECT_DIR/scripts/train_config_04_vanilla.env"
INIT_CKPT="${INIT_CKPT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt}"
DATASET_RUN_DIR="${DATASET_RUN_DIR:-$RUN_ROOT/latest_dataset_5b}"
RUN_TAG="${RUN_TAG:-04_vanilla_goldfish_5b_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_TAG}"
STATE_DIR="${STATE_DIR:-$RUN_ROOT/${RUN_TAG}_submit_state}"
TRAINING_CHAIN_TSV="$STATE_DIR/training_chain.tsv"
COMMANDS_SH="$STATE_DIR/training_sbatch_commands.sh"

DRY_RUN="${DRY_RUN:-1}"
CONFIRM_5B_LAUNCH="${CONFIRM_5B_LAUNCH:-0}"
ACCOUNT="${ACCOUNT:-a0140}"
PARTITION="${PARTITION:-normal}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-36}"
SEGMENT_TIME_LIMIT="${SEGMENT_TIME_LIMIT:-12:00:00}"
SAVE_INTERVAL="${SAVE_INTERVAL:-119}"
EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"
LOSS_OBJECTIVE="${LOSS_OBJECTIVE:-goldfish}"
LR_SCHEDULE_STYLE="${LR_SCHEDULE_STYLE:-constant}"
LR_WARMUP_INIT="${LR_WARMUP_INIT:-1.1e-6}"
ADEMA_BETA3_WARMUP_STEPS="${ADEMA_BETA3_WARMUP_STEPS:-287}"
ADEMA_ALPHA_WARMUP_STEPS="${ADEMA_ALPHA_WARMUP_STEPS:-287}"
DATA_SEED="${DATA_SEED:-20260528}"
LAUNCH_MODE="${LAUNCH_MODE:-slurm}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
SUBMIT_CHECKPOINT_WATCHER="${SUBMIT_CHECKPOINT_WATCHER:-1}"
WATCHER_JOB_ID=""

GLOBAL_BATCH_TOKENS=4194304

usage() {
  cat <<'USAGE'
Usage:
  DRY_RUN=0 CONFIRM_5B_LAUNCH=1 submit_training_5b_chain.sh

Required before live launch:
  - final dataset build has completed;
  - DATASET_RUN_DIR/dataset_paths.env exists;
  - validation JSON reports ok: true.

The script writes:
  - training_chain.tsv
  - training_sbatch_commands.sh
  - training_5b_paths.env
under RUN_ROOT/<RUN_TAG>_submit_state.
USAGE
}

case "$DRY_RUN" in
  0|1) ;;
  *) echo "ERROR: DRY_RUN must be 0|1, got $DRY_RUN" >&2; exit 2 ;;
esac
case "$CONFIRM_5B_LAUNCH" in
  0|1) ;;
  *) echo "ERROR: CONFIRM_5B_LAUNCH must be 0|1, got $CONFIRM_5B_LAUNCH" >&2; exit 2 ;;
esac
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_5B_LAUNCH" != "1" ]; then
  echo "ERROR: live 5B launch requires CONFIRM_5B_LAUNCH=1" >&2
  exit 2
fi
if [ -e "$OUTPUT_DIR" ] && [ "$ALLOW_EXISTING_OUTPUT" != "1" ]; then
  echo "ERROR: output dir already exists: $OUTPUT_DIR" >&2
  echo "Set ALLOW_EXISTING_OUTPUT=1 only for an intentional resume/resubmit." >&2
  exit 3
fi

test -f "$TRAIN_SCRIPT"
test -f "$TRAIN_CONFIG"
test -d "$INIT_CKPT/release"
test -f "$DATASET_RUN_DIR/dataset_paths.env"
# shellcheck disable=SC1090
source "$DATASET_RUN_DIR/dataset_paths.env"
test -f "$DATA_PREFIX.bin"
test -f "$DATA_PREFIX.idx"
test -f "$VALIDATION_JSON"
python3 - "$VALIDATION_JSON" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
if not data.get("ok"):
    raise SystemExit(f"dataset validation is not ok: {path}: {data.get('errors')}")
PY

mkdir -p "$STATE_DIR"
: > "$COMMANDS_SH"
chmod +x "$COMMANDS_SH"
printf "run_tag\tsegment\ttarget_iter\ttarget_tokens\toutput_dir\tinit_ckpt\tresume_training\tdependency_job\tjob_id\n" > "$TRAINING_CHAIN_TSV"

log_cmd() {
  printf '%q ' "$@" >> "$COMMANDS_SH"
  printf '\n' >> "$COMMANDS_SH"
}

submit_or_dryrun() {
  local dry_id="$1"
  shift
  log_cmd "$@"
  if [ "$DRY_RUN" = "1" ]; then
    echo "$dry_id"
  else
    "$@"
  fi
}

target_tokens_for_iter() {
  local iter="$1"
  echo $(( iter * GLOBAL_BATCH_TOKENS ))
}

echo "=== submit_training_5b_chain.sh ==="
echo "DRY_RUN:              $DRY_RUN"
echo "RUN_TAG:              $RUN_TAG"
echo "OUTPUT_DIR:           $OUTPUT_DIR"
echo "STATE_DIR:            $STATE_DIR"
echo "TRAIN_SCRIPT:         $TRAIN_SCRIPT"
echo "TRAIN_CONFIG:         $TRAIN_CONFIG"
echo "INIT_CKPT:            $INIT_CKPT"
echo "DATASET_RUN_DIR:      $DATASET_RUN_DIR"
echo "DATA_PREFIX:          $DATA_PREFIX"
echo "VALIDATION_JSON:      $VALIDATION_JSON"
echo "ACCOUNT/PARTITION:    $ACCOUNT / $PARTITION"
echo "NODES/GPUS_PER_NODE:  $NODES / $GPUS_PER_NODE"
echo "SEGMENT_TIME_LIMIT:   $SEGMENT_TIME_LIMIT"
echo "SAVE_INTERVAL:        $SAVE_INTERVAL"
echo "EVAL_INTERVAL:        $EVAL_INTERVAL"
echo "SUBMIT_WATCHER:       $SUBMIT_CHECKPOINT_WATCHER"
echo

if [ "$DRY_RUN" = "0" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Segment targets are absolute Megatron iteration targets. Required report
# checkpoints are 119 (~0.5B), 238 (~1B), 477 (~2B), 834 (~3.5B), and 1192
# (~5B). SAVE_INTERVAL=119 writes the 119/238 checkpoints inside segment 1.
# Segment 1 must run past the 287-step LR warmup, because Megatron asserts
# lr_warmup_steps < train_iters for each submitted job. Target 300 keeps the
# first segment within Clariden's 12h walltime at observed throughput.
segments=(
  300
  477
  596
  715
  834
  953
  1072
  1192
)

dependency_job=""
init_ckpt="$INIT_CKPT"
resume_training=0
segment_idx=0
for target_iter in "${segments[@]}"; do
  segment_idx=$((segment_idx + 1))
  target_tokens="$(target_tokens_for_iter "$target_iter")"
  job_name="04van5b_i${target_iter}"
  dependency_args=()
  dependency_label="none"
  if [ -n "$dependency_job" ]; then
    dependency_args=(--dependency="afterok:$dependency_job")
    dependency_label="$dependency_job"
    init_ckpt="$OUTPUT_DIR/checkpoints"
    resume_training=1
  fi

  cmd=(
    sbatch --parsable
    "${dependency_args[@]}"
    --job-name="$job_name"
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes="$NODES"
    --ntasks-per-node="$GPUS_PER_NODE"
    --gpus-per-node="$GPUS_PER_NODE"
    --gres="gpu:$GPUS_PER_NODE"
    --cpus-per-task="$CPUS_PER_TASK"
    --time="$SEGMENT_TIME_LIMIT"
    --output="$RUN_ROOT/%x-%j.out"
    --error="$RUN_ROOT/%x-%j.err"
    --export=ALL,ARM=vanilla,INIT_CKPT="$init_ckpt",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR_OVERRIDE="$LEGACY_TRAIN_DIR",TRAIN_CONFIG_OVERRIDE="$TRAIN_CONFIG",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$SEGMENT_TIME_LIMIT",TRAIN_TOKENS="$target_tokens",SAVE_INTERVAL="$SAVE_INTERVAL",EVAL_INTERVAL="$EVAL_INTERVAL",LOSS_OBJECTIVE="$LOSS_OBJECTIVE",LR_SCHEDULE_STYLE="$LR_SCHEDULE_STYLE",LR_WARMUP_INIT="$LR_WARMUP_INIT",ADEMA_BETA3_WARMUP_STEPS="$ADEMA_BETA3_WARMUP_STEPS",ADEMA_ALPHA_WARMUP_STEPS="$ADEMA_ALPHA_WARMUP_STEPS",BASE_DATA_PREFIX="$DATA_PREFIX",EXT_DATA_PREFIX="$DATA_PREFIX",DATA_SEED="$DATA_SEED",DISABLE_SAVE=0,RESUME_TRAINING="$resume_training"
    "$TRAIN_SCRIPT"
  )

  job_id="$(submit_or_dryrun "DRYRUN_i${target_iter}" "${cmd[@]}")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$RUN_TAG" "$segment_idx" "$target_iter" "$target_tokens" "$OUTPUT_DIR" \
    "$init_ckpt" "$resume_training" "$dependency_label" "$job_id" >> "$TRAINING_CHAIN_TSV"
  echo "segment $segment_idx target_iter=$target_iter target_tokens=$target_tokens job=$job_id dependency=$dependency_label resume=$resume_training"
  dependency_job="$job_id"
done

cat > "$STATE_DIR/training_5b_paths.env" <<PATHS
RUN_TAG="$RUN_TAG"
OUTPUT_DIR="$OUTPUT_DIR"
STATE_DIR="$STATE_DIR"
TRAINING_CHAIN_TSV="$TRAINING_CHAIN_TSV"
COMMANDS_SH="$COMMANDS_SH"
DATASET_RUN_DIR="$DATASET_RUN_DIR"
DATA_PREFIX="$DATA_PREFIX"
VALIDATION_JSON="$VALIDATION_JSON"
FIRST_JOB_ID="$(awk -F '\t' 'NR==2 {print $9}' "$TRAINING_CHAIN_TSV")"
FINAL_JOB_ID="$dependency_job"
PATHS

if [ "$SUBMIT_CHECKPOINT_WATCHER" = "1" ]; then
  watcher_script="$SUBPROJECT_DIR/scripts/watch_and_submit_checkpoint_sidecars.sbatch"
  test -f "$watcher_script"
  watcher_job_id="$(submit_or_dryrun "DRYRUN_WATCHER" \
    sbatch --parsable \
    --account="$ACCOUNT" \
    --partition=xfer \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --time=24:00:00 \
    --job-name=04van5b_watch \
    --output="$RUN_ROOT/%x-%j.out" \
    --error="$RUN_ROOT/%x-%j.err" \
    --export=ALL,RUN_TAG="$RUN_TAG",TRAIN_RUN_DIR="$OUTPUT_DIR",WATCH_STATE_DIR="$RUN_ROOT/${RUN_TAG}_sidecar_watch" \
    "$watcher_script")"
  WATCHER_JOB_ID="$watcher_job_id"
  printf 'WATCHER_JOB_ID="%s"\n' "$WATCHER_JOB_ID" >> "$STATE_DIR/training_5b_paths.env"
fi

echo
echo "Training chain manifest: $TRAINING_CHAIN_TSV"
echo "Training sbatch commands: $COMMANDS_SH"
echo "Paths env: $STATE_DIR/training_5b_paths.env"
echo "Final dependency-chain job: $dependency_job"
if [ -n "$WATCHER_JOB_ID" ]; then
  echo "Checkpoint sidecar watcher job: $WATCHER_JOB_ID"
fi

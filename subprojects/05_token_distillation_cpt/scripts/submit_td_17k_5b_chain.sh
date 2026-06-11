#!/usr/bin/env bash
# Submit the Task-2 TD-17k layer-11 Path-A 5B diagnostic as checkpoint-bounded
# Slurm segments, with checkpoint eval sidecars submitted by a watcher.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
LEGACY_TRAIN_DIR="${LEGACY_TRAIN_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training}"
TRAIN_SCRIPT="$LEGACY_TRAIN_DIR/bakeoff_train.sbatch"
TRAIN_CONFIG="${TRAIN_CONFIG:-$SUBPROJECT_DIR/scripts/train_config_td_path_a.env}"
INIT_CKPT="${INIT_CKPT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/td_layer11/megatron_tp2_r17patched_path_a}"
TD_HF_TOKENIZER_DIR="${TD_HF_TOKENIZER_DIR:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/td_layer11/hf_roundtrip_path_a}"
TD_R17_README="${TD_R17_README:-$INIT_CKPT/README.md}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt}"
RUN_TAG="${RUN_TAG:-05_td17k_l11_path_a_goldfish_5b_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_TAG}"
STATE_DIR="${STATE_DIR:-$RUN_ROOT/${RUN_TAG}_submit_state}"
TRAINING_CHAIN_TSV="$STATE_DIR/training_chain.tsv"
COMMANDS_SH="$STATE_DIR/training_sbatch_commands.sh"

DATASET_RUN_DIR="${DATASET_RUN_DIR:-}"
DATASET_PATHS_ENV="${DATASET_PATHS_ENV:-}"
if [ -z "$DATASET_PATHS_ENV" ] && [ -n "$DATASET_RUN_DIR" ]; then
  DATASET_PATHS_ENV="$DATASET_RUN_DIR/dataset_paths.env"
fi

DRY_RUN="${DRY_RUN:-1}"
CONFIRM_TD_LAUNCH="${CONFIRM_TD_LAUNCH:-0}"
ACCOUNT="${ACCOUNT:-a0140}"
PARTITION="${PARTITION:-normal}"
WATCHER_PARTITION="${WATCHER_PARTITION:-xfer}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-36}"
SEGMENT_TIME_LIMIT="${SEGMENT_TIME_LIMIT:-12:00:00}"
SAVE_INTERVAL="${SAVE_INTERVAL:-119}"
EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"
LOSS_OBJECTIVE="${LOSS_OBJECTIVE:-goldfish}"
LR_SCHEDULE_STYLE="${LR_SCHEDULE_STYLE:-constant}"
LR_WARMUP_INIT="${LR_WARMUP_INIT:-1.1e-6}"
LR_WARMUP_TOKENS="${LR_WARMUP_TOKENS:-1200000000}"
ADEMA_BETA3_WARMUP_STEPS="${ADEMA_BETA3_WARMUP_STEPS:-287}"
ADEMA_ALPHA_WARMUP_STEPS="${ADEMA_ALPHA_WARMUP_STEPS:-287}"
DATA_SEED="${DATA_SEED:-20260528}"
LAUNCH_MODE="${LAUNCH_MODE:-slurm}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
SUBMIT_CHECKPOINT_WATCHER="${SUBMIT_CHECKPOINT_WATCHER:-1}"
GOLDFISH_HASH_REPORT="${GOLDFISH_HASH_REPORT:-}"
DECONTAM_AUDIT_JSON="${DECONTAM_AUDIT_JSON:-}"
WATCHER_JOB_ID=""

GLOBAL_BATCH_TOKENS=4194304

usage() {
  cat <<'USAGE'
Usage:
  EXT_DATA_PREFIX=/capstor/.../hplt_b1_ext_text_document \
  DRY_RUN=0 CONFIRM_TD_LAUNCH=1 submit_td_17k_5b_chain.sh

Required before live launch:
  - INIT_CKPT is the TD layer-11 Path-A R17-patched Megatron checkpoint;
  - TD_HF_TOKENIZER_DIR contains config.json + tokenizer.json for conversion;
  - EXT_DATA_PREFIX points to extension-tokenized .bin/.idx data;
  - VALIDATION_JSON reports ok: true;
  - GOLDFISH_HASH_REPORT exists;
  - DECONTAM_AUDIT_JSON exists.

The script writes:
  - training_chain.tsv
  - training_sbatch_commands.sh
  - training_td17k_paths.env
under RUN_ROOT/<RUN_TAG>_submit_state.
USAGE
}

case "$DRY_RUN" in
  0|1) ;;
  *) echo "ERROR: DRY_RUN must be 0|1, got $DRY_RUN" >&2; exit 2 ;;
esac
case "$CONFIRM_TD_LAUNCH" in
  0|1) ;;
  *) echo "ERROR: CONFIRM_TD_LAUNCH must be 0|1, got $CONFIRM_TD_LAUNCH" >&2; exit 2 ;;
esac
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_TD_LAUNCH" != "1" ]; then
  echo "ERROR: live TD launch requires CONFIRM_TD_LAUNCH=1" >&2
  exit 2
fi
if [ -e "$OUTPUT_DIR" ] && [ "$ALLOW_EXISTING_OUTPUT" != "1" ]; then
  echo "ERROR: output dir already exists: $OUTPUT_DIR" >&2
  echo "Set ALLOW_EXISTING_OUTPUT=1 only for an intentional resume/resubmit." >&2
  exit 3
fi

test -f "$TRAIN_SCRIPT"
test -f "$TRAIN_CONFIG"

USER_BASE_DATA_PREFIX="${BASE_DATA_PREFIX:-}"
USER_EXT_DATA_PREFIX="${EXT_DATA_PREFIX:-}"
USER_VALIDATION_JSON="${VALIDATION_JSON:-}"
# shellcheck disable=SC1090
source "$TRAIN_CONFIG"
if [ -n "$DATASET_PATHS_ENV" ]; then
  test -f "$DATASET_PATHS_ENV"
  # shellcheck disable=SC1090
  source "$DATASET_PATHS_ENV"
fi
if [ -n "$USER_BASE_DATA_PREFIX" ]; then
  BASE_DATA_PREFIX="$USER_BASE_DATA_PREFIX"
fi
if [ -n "$USER_EXT_DATA_PREFIX" ]; then
  EXT_DATA_PREFIX="$USER_EXT_DATA_PREFIX"
fi
if [ -n "$USER_VALIDATION_JSON" ]; then
  VALIDATION_JSON="$USER_VALIDATION_JSON"
fi

: "${EXT_DATA_PREFIX:?EXT_DATA_PREFIX is required for TD; do not reuse base DATA_PREFIX implicitly}"

live_required_path() {
  local label="$1"
  local path="$2"
  if [ "$DRY_RUN" = "0" ] && [ ! -e "$path" ]; then
    echo "ERROR: missing required live-launch artifact for $label: $path" >&2
    exit 3
  fi
}

live_required_file() {
  local label="$1"
  local path="$2"
  if [ "$DRY_RUN" = "0" ] && [ ! -s "$path" ]; then
    echo "ERROR: missing required live-launch file for $label: $path" >&2
    exit 3
  fi
}

live_required_path "TD init checkpoint release" "$INIT_CKPT/release"
live_required_file "TD HF tokenizer config" "$TD_HF_TOKENIZER_DIR/config.json"
live_required_file "TD HF tokenizer tokenizer.json" "$TD_HF_TOKENIZER_DIR/tokenizer.json"
live_required_file "extended data bin" "$EXT_DATA_PREFIX.bin"
live_required_file "extended data idx" "$EXT_DATA_PREFIX.idx"

if [ -n "${VALIDATION_JSON:-}" ]; then
  live_required_file "dataset validation JSON" "$VALIDATION_JSON"
  if [ -s "$VALIDATION_JSON" ]; then
    python3 - "$VALIDATION_JSON" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
if not data.get("ok"):
    raise SystemExit(f"dataset validation is not ok: {path}: {data.get('errors')}")
PY
  fi
elif [ "$DRY_RUN" = "0" ]; then
  echo "ERROR: VALIDATION_JSON is required for live TD launch" >&2
  exit 3
fi

live_required_file "Path-A R17 README" "$TD_R17_README"
if [ "$DRY_RUN" = "0" ] && ! grep -Eiq 'max[[:space:]_-]+abs[[:space:]_-]+diff[^0-9]*0(\.0+)?' "$TD_R17_README"; then
  echo "ERROR: TD_R17_README does not mention max abs diff 0.0: $TD_R17_README" >&2
  exit 3
fi
if [ "$DRY_RUN" = "0" ]; then
  : "${GOLDFISH_HASH_REPORT:?GOLDFISH_HASH_REPORT is required for live TD launch}"
  : "${DECONTAM_AUDIT_JSON:?DECONTAM_AUDIT_JSON is required for live TD launch}"
fi
if [ -n "$GOLDFISH_HASH_REPORT" ]; then
  live_required_file "Goldfish hash-uniformity report" "$GOLDFISH_HASH_REPORT"
fi
if [ -n "$DECONTAM_AUDIT_JSON" ]; then
  live_required_file "5B decontamination audit" "$DECONTAM_AUDIT_JSON"
fi

if (( LR_WARMUP_TOKENS > 0 )); then
  first_target_tokens=$(( 300 * GLOBAL_BATCH_TOKENS ))
  if (( first_target_tokens < LR_WARMUP_TOKENS )); then
    echo "ERROR: first segment target $first_target_tokens is below LR_WARMUP_TOKENS=$LR_WARMUP_TOKENS" >&2
    exit 2
  fi
fi

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

echo "=== submit_td_17k_5b_chain.sh ==="
echo "DRY_RUN:              $DRY_RUN"
echo "RUN_TAG:              $RUN_TAG"
echo "OUTPUT_DIR:           $OUTPUT_DIR"
echo "STATE_DIR:            $STATE_DIR"
echo "TRAIN_SCRIPT:         $TRAIN_SCRIPT"
echo "TRAIN_CONFIG:         $TRAIN_CONFIG"
echo "INIT_CKPT:            $INIT_CKPT"
echo "TD_HF_TOKENIZER_DIR:  $TD_HF_TOKENIZER_DIR"
echo "TD_R17_README:        $TD_R17_README"
echo "DATASET_PATHS_ENV:    ${DATASET_PATHS_ENV:-none}"
echo "BASE_DATA_PREFIX:     ${BASE_DATA_PREFIX:-}"
echo "EXT_DATA_PREFIX:      $EXT_DATA_PREFIX"
echo "VALIDATION_JSON:      ${VALIDATION_JSON:-}"
echo "GOLDFISH_HASH_REPORT: ${GOLDFISH_HASH_REPORT:-}"
echo "DECONTAM_AUDIT_JSON:  ${DECONTAM_AUDIT_JSON:-}"
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
  job_name="05td5b_i${target_iter}"
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
    --export=ALL,ARM=td,INIT_CKPT="$init_ckpt",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR_OVERRIDE="$LEGACY_TRAIN_DIR",TRAIN_CONFIG_OVERRIDE="$TRAIN_CONFIG",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$SEGMENT_TIME_LIMIT",TRAIN_TOKENS="$target_tokens",SAVE_INTERVAL="$SAVE_INTERVAL",EVAL_INTERVAL="$EVAL_INTERVAL",LOSS_OBJECTIVE="$LOSS_OBJECTIVE",LR_SCHEDULE_STYLE="$LR_SCHEDULE_STYLE",LR_WARMUP_INIT="$LR_WARMUP_INIT",ADEMA_BETA3_WARMUP_STEPS="$ADEMA_BETA3_WARMUP_STEPS",ADEMA_ALPHA_WARMUP_STEPS="$ADEMA_ALPHA_WARMUP_STEPS",BASE_DATA_PREFIX="$BASE_DATA_PREFIX",EXT_DATA_PREFIX="$EXT_DATA_PREFIX",EXT_TOKENIZER_DIR="$EXT_TOKENIZER_DIR",DATA_SEED="$DATA_SEED",MAX_POSITION_EMBEDDINGS="$MAX_POSITION_EMBEDDINGS",ROTARY_BASE="$ROTARY_BASE",USE_ROPE_SCALING="$USE_ROPE_SCALING",ROPE_SCALING_FACTOR="$ROPE_SCALING_FACTOR",MAKE_VOCAB_SIZE_DIVISIBLE_BY="$MAKE_VOCAB_SIZE_DIVISIBLE_BY",DISABLE_SAVE=0,RESUME_TRAINING="$resume_training"
    "$TRAIN_SCRIPT"
  )

  job_id="$(submit_or_dryrun "DRYRUN_i${target_iter}" "${cmd[@]}")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$RUN_TAG" "$segment_idx" "$target_iter" "$target_tokens" "$OUTPUT_DIR" \
    "$init_ckpt" "$resume_training" "$dependency_label" "$job_id" >> "$TRAINING_CHAIN_TSV"
  echo "segment $segment_idx target_iter=$target_iter target_tokens=$target_tokens job=$job_id dependency=$dependency_label resume=$resume_training"
  dependency_job="$job_id"
done

cat > "$STATE_DIR/training_td17k_paths.env" <<PATHS
RUN_TAG="$RUN_TAG"
OUTPUT_DIR="$OUTPUT_DIR"
STATE_DIR="$STATE_DIR"
TRAINING_CHAIN_TSV="$TRAINING_CHAIN_TSV"
COMMANDS_SH="$COMMANDS_SH"
DATASET_PATHS_ENV="${DATASET_PATHS_ENV:-}"
BASE_DATA_PREFIX="${BASE_DATA_PREFIX:-}"
EXT_DATA_PREFIX="$EXT_DATA_PREFIX"
VALIDATION_JSON="${VALIDATION_JSON:-}"
INIT_CKPT="$INIT_CKPT"
TD_HF_TOKENIZER_DIR="$TD_HF_TOKENIZER_DIR"
TD_R17_README="$TD_R17_README"
GOLDFISH_HASH_REPORT="${GOLDFISH_HASH_REPORT:-}"
DECONTAM_AUDIT_JSON="${DECONTAM_AUDIT_JSON:-}"
FIRST_JOB_ID="$(awk -F '\t' 'NR==2 {print $9}' "$TRAINING_CHAIN_TSV")"
FINAL_JOB_ID="$dependency_job"
PATHS

if [ "$SUBMIT_CHECKPOINT_WATCHER" = "1" ]; then
  watcher_script="$SUBPROJECT_DIR/scripts/watch_and_submit_td_checkpoint_sidecars.sbatch"
  test -f "$watcher_script"
  watcher_job_id="$(submit_or_dryrun "DRYRUN_WATCHER" \
    sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$WATCHER_PARTITION" \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --time=24:00:00 \
    --job-name=05td5b_watch \
    --output="$RUN_ROOT/%x-%j.out" \
    --error="$RUN_ROOT/%x-%j.err" \
    --export=ALL,RUN_ROOT="$RUN_ROOT",RUN_TAG="$RUN_TAG",TRAIN_RUN_DIR="$OUTPUT_DIR",WATCH_STATE_DIR="$RUN_ROOT/${RUN_TAG}_sidecar_watch",EVAL_ARM=td,CHECKPOINT_LABEL_PREFIX=TD-17k-L11,HF_TOKENIZER_DIR="$TD_HF_TOKENIZER_DIR",WATCHER_PARTITION="$WATCHER_PARTITION" \
    "$watcher_script")"
  WATCHER_JOB_ID="$watcher_job_id"
  printf 'WATCHER_JOB_ID="%s"\n' "$WATCHER_JOB_ID" >> "$STATE_DIR/training_td17k_paths.env"
fi

echo
echo "Training chain manifest: $TRAINING_CHAIN_TSV"
echo "Training sbatch commands: $COMMANDS_SH"
echo "Paths env: $STATE_DIR/training_td17k_paths.env"
echo "Final dependency-chain job: $dependency_job"
if [ -n "$WATCHER_JOB_ID" ]; then
  echo "Checkpoint sidecar watcher job: $WATCHER_JOB_ID"
fi

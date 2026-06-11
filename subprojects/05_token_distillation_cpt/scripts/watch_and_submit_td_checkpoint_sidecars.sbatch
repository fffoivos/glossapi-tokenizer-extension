#!/usr/bin/env bash
# CPU-only watcher that submits eval sidecars when required checkpoints appear.

#SBATCH --job-name=05td5b_watch
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "05 TD checkpoint sidecar watcher"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
EVAL_ARM="${EVAL_ARM:-td}"
CHECKPOINT_LABEL_PREFIX="${CHECKPOINT_LABEL_PREFIX:-TD-17k-L11}"
HF_TOKENIZER_DIR="${HF_TOKENIZER_DIR:-}"
WATCHER_PARTITION="${WATCHER_PARTITION:-${SLURM_JOB_PARTITION:-xfer}}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-$RUN_ROOT/$RUN_TAG}"
WATCH_STATE_DIR="${WATCH_STATE_DIR:-$RUN_ROOT/${RUN_TAG}_sidecar_watch}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
MAX_WATCH_SECONDS="${MAX_WATCH_SECONDS:-82800}"
RESUBMIT_IF_INCOMPLETE="${RESUBMIT_IF_INCOMPLETE:-1}"
REQUIRE_CODE_MATH_HELDOUTS="${REQUIRE_CODE_MATH_HELDOUTS:-1}"
CODE_HELDOUT_JSONL="${CODE_HELDOUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_code_heldout_200_20260528.jsonl}"
MATH_HELDOUT_JSONL="${MATH_HELDOUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_math_heldout_200_20260528.jsonl}"
SUBMIT_NATIVE="${SUBMIT_NATIVE:-1}"
SUBMIT_GREEK_NLP="${SUBMIT_GREEK_NLP:-1}"
SUBMIT_BPB="${SUBMIT_BPB:-1}"
SUBMIT_RETENTION="${SUBMIT_RETENTION:-1}"
SUBMIT_CHECKSUM="${SUBMIT_CHECKSUM:-1}"
NATIVE_BENCHMARKS="${NATIVE_BENCHMARKS:-greekmmlu,ilsp_medical_mcqa,ilsp_mcqa_asep,plutus_qa}"

SUBMIT_SCRIPT="$SUBPROJECT_DIR/scripts/submit_td_checkpoint_sidecars.sh"
mkdir -p "$WATCH_STATE_DIR"

cat > "$WATCH_STATE_DIR/watcher_${SLURM_JOB_ID:-manual}.env" <<META
RUN_TAG="$RUN_TAG"
TRAIN_RUN_DIR="$TRAIN_RUN_DIR"
WATCH_STATE_DIR="$WATCH_STATE_DIR"
SLEEP_SECONDS="$SLEEP_SECONDS"
MAX_WATCH_SECONDS="$MAX_WATCH_SECONDS"
RESUBMIT_IF_INCOMPLETE="$RESUBMIT_IF_INCOMPLETE"
REQUIRE_CODE_MATH_HELDOUTS="$REQUIRE_CODE_MATH_HELDOUTS"
CODE_HELDOUT_JSONL="$CODE_HELDOUT_JSONL"
MATH_HELDOUT_JSONL="$MATH_HELDOUT_JSONL"
EVAL_ARM="$EVAL_ARM"
CHECKPOINT_LABEL_PREFIX="$CHECKPOINT_LABEL_PREFIX"
HF_TOKENIZER_DIR="$HF_TOKENIZER_DIR"
WATCHER_PARTITION="$WATCHER_PARTITION"
CHECKPOINTS_FILE="${CHECKPOINTS_FILE:-}"
SUBMIT_NATIVE="$SUBMIT_NATIVE"
SUBMIT_GREEK_NLP="$SUBMIT_GREEK_NLP"
SUBMIT_BPB="$SUBMIT_BPB"
SUBMIT_RETENTION="$SUBMIT_RETENTION"
SUBMIT_CHECKSUM="$SUBMIT_CHECKSUM"
NATIVE_BENCHMARKS="$NATIVE_BENCHMARKS"
START_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
META

echo "=== watch_and_submit_td_checkpoint_sidecars.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "RUN_TAG: $RUN_TAG"
echo "EVAL_ARM: $EVAL_ARM"
echo "CHECKPOINT_LABEL_PREFIX: $CHECKPOINT_LABEL_PREFIX"
echo "HF_TOKENIZER_DIR: $HF_TOKENIZER_DIR"
echo "WATCHER_PARTITION: $WATCHER_PARTITION"
echo "TRAIN_RUN_DIR: $TRAIN_RUN_DIR"
echo "WATCH_STATE_DIR: $WATCH_STATE_DIR"
echo "SUBMIT_SCRIPT: $SUBMIT_SCRIPT"
echo "SLEEP_SECONDS: $SLEEP_SECONDS"
echo "MAX_WATCH_SECONDS: $MAX_WATCH_SECONDS"
echo "REQUIRE_CODE_MATH_HELDOUTS: $REQUIRE_CODE_MATH_HELDOUTS"
echo "CODE_HELDOUT_JSONL: $CODE_HELDOUT_JSONL"
echo "MATH_HELDOUT_JSONL: $MATH_HELDOUT_JSONL"
echo "CHECKPOINTS_FILE: ${CHECKPOINTS_FILE:-<default>}"
echo "NATIVE_BENCHMARKS: $NATIVE_BENCHMARKS"
echo "SUBMIT_NATIVE: $SUBMIT_NATIVE"
echo "SUBMIT_GREEK_NLP: $SUBMIT_GREEK_NLP"
echo "SUBMIT_BPB: $SUBMIT_BPB"
echo "SUBMIT_RETENTION: $SUBMIT_RETENTION"
echo "SUBMIT_CHECKSUM: $SUBMIT_CHECKSUM"
echo

test -x "$SUBMIT_SCRIPT"

# Overridable via CHECKPOINTS_FILE ("iter consumed_tokens label" per line) so the
# 13.5B 2-arm experiment can feed its cadence; defaults to the TD-5B targets.
if [ -n "${CHECKPOINTS_FILE:-}" ] && [ -f "$CHECKPOINTS_FILE" ]; then
  mapfile -t checkpoints < "$CHECKPOINTS_FILE"
else
  checkpoints=(
    "119 499122176 ${CHECKPOINT_LABEL_PREFIX}-0.5B"
    "238 998244352 ${CHECKPOINT_LABEL_PREFIX}-1B"
    "477 2000683008 ${CHECKPOINT_LABEL_PREFIX}-2B"
    "834 3498049536 ${CHECKPOINT_LABEL_PREFIX}-3.5B"
    "1192 4999610368 ${CHECKPOINT_LABEL_PREFIX}-5B"
  )
fi

start_epoch="$(date +%s)"

all_submitted() {
  local item iter tokens label state_file
  for item in "${checkpoints[@]}"; do
    read -r iter tokens label <<< "$item"
    state_file="$WATCH_STATE_DIR/iter_${iter}.submitted"
    if [ ! -f "$state_file" ]; then
      return 1
    fi
  done
  return 0
}

code_math_heldouts_ready() {
  if [ "$REQUIRE_CODE_MATH_HELDOUTS" != "1" ]; then
    return 0
  fi
  [ -s "$CODE_HELDOUT_JSONL" ] && [ -s "$MATH_HELDOUT_JSONL" ]
}

while true; do
  for item in "${checkpoints[@]}"; do
    read -r iter tokens label <<< "$item"
    iter_pad="$(printf "%07d" "$iter")"
    ckpt_dir="$TRAIN_RUN_DIR/checkpoints/iter_$iter_pad"
    state_file="$WATCH_STATE_DIR/iter_${iter}.submitted"
    if [ -f "$state_file" ]; then
      continue
    fi
    if [ -d "$ckpt_dir" ] && [ -f "$ckpt_dir/.metadata" ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] checkpoint ready: iter=$iter label=$label"
      if ! code_math_heldouts_ready; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] delaying sidecars for iter=$iter until code/math heldouts exist"
        echo "  code: $CODE_HELDOUT_JSONL"
        echo "  math: $MATH_HELDOUT_JSONL"
        continue
      fi
      CODE_HELDOUT_JSONL="$CODE_HELDOUT_JSONL" \
      MATH_HELDOUT_JSONL="$MATH_HELDOUT_JSONL" \
      HF_TOKENIZER_DIR="$HF_TOKENIZER_DIR" \
      NATIVE_BENCHMARKS="$NATIVE_BENCHMARKS" \
      SUBMIT_NATIVE="$SUBMIT_NATIVE" \
      SUBMIT_GREEK_NLP="$SUBMIT_GREEK_NLP" \
      SUBMIT_BPB="$SUBMIT_BPB" \
      SUBMIT_RETENTION="$SUBMIT_RETENTION" \
      SUBMIT_CHECKSUM="$SUBMIT_CHECKSUM" \
      "$SUBMIT_SCRIPT" \
        --train-run-dir "$TRAIN_RUN_DIR" \
        --run-tag "$RUN_TAG" \
        --arm "$EVAL_ARM" \
        --iteration "$iter" \
        --tokens "$tokens" \
        --checkpoint-label "$label" \
        2>&1 | tee "$WATCH_STATE_DIR/iter_${iter}_submit.log"
      touch "$state_file"
    fi
  done

  if all_submitted; then
    echo "all required checkpoint sidecars submitted"
    date -u
    exit 0
  fi

  now="$(date +%s)"
  elapsed=$(( now - start_epoch ))
  if [ "$elapsed" -ge "$MAX_WATCH_SECONDS" ]; then
    echo "watch window expired after ${elapsed}s"
    if [ "$RESUBMIT_IF_INCOMPLETE" = "1" ]; then
      next_job="$(sbatch --parsable \
        --account=a0140 \
        --partition="$WATCHER_PARTITION" \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=4G \
        --time=24:00:00 \
        --job-name=05${EVAL_ARM}5b_watch \
        --output="$RUN_ROOT/%x-%j.out" \
        --error="$RUN_ROOT/%x-%j.err" \
        --export=ALL,RUN_TAG="$RUN_TAG",TRAIN_RUN_DIR="$TRAIN_RUN_DIR",WATCH_STATE_DIR="$WATCH_STATE_DIR",SLEEP_SECONDS="$SLEEP_SECONDS",MAX_WATCH_SECONDS="$MAX_WATCH_SECONDS",RESUBMIT_IF_INCOMPLETE="$RESUBMIT_IF_INCOMPLETE",REQUIRE_CODE_MATH_HELDOUTS="$REQUIRE_CODE_MATH_HELDOUTS",CODE_HELDOUT_JSONL="$CODE_HELDOUT_JSONL",MATH_HELDOUT_JSONL="$MATH_HELDOUT_JSONL",EVAL_ARM="$EVAL_ARM",CHECKPOINT_LABEL_PREFIX="$CHECKPOINT_LABEL_PREFIX",HF_TOKENIZER_DIR="$HF_TOKENIZER_DIR",WATCHER_PARTITION="$WATCHER_PARTITION",CHECKPOINTS_FILE="${CHECKPOINTS_FILE:-}",NATIVE_BENCHMARKS="$NATIVE_BENCHMARKS",SUBMIT_NATIVE="$SUBMIT_NATIVE",SUBMIT_GREEK_NLP="$SUBMIT_GREEK_NLP",SUBMIT_BPB="$SUBMIT_BPB",SUBMIT_RETENTION="$SUBMIT_RETENTION",SUBMIT_CHECKSUM="$SUBMIT_CHECKSUM" \
        "$SUBPROJECT_DIR/scripts/watch_and_submit_td_checkpoint_sidecars.sbatch")"
      echo "resubmitted watcher: $next_job"
    fi
    exit 0
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for checkpoints; elapsed=${elapsed}s"
  sleep "$SLEEP_SECONDS"
done

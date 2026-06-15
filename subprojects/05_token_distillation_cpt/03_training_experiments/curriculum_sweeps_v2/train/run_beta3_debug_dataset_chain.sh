#!/usr/bin/env bash
# Serial beta3 dataset builder for Clariden debug QOS.
#
# debug-qos currently allows only one running job and two submitted jobs per user.
# This driver keeps at most one running + one dependent pending job in debug,
# adopting already-submitted jobs when START_AFTER_JOB/SUBMITTED_JOBS are set.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

STAGE="${STAGE:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_beta3}"
export STAGE
source paths.env

RECIPES_DIR="${RECIPES_DIR:-$V2/dataset/recipes_beta3}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIX_SHARDS="${MIX_SHARDS:-32}"
HPLT_TGT="${HPLT_TGT:-16000000000}"
GLOSSAPI_TGT="${GLOSSAPI_TGT:-6900000000}"
REPLAY_TGT="${REPLAY_TGT:-6500000000}"
STATE_FILE="${STATE_FILE:-$STAGE/beta3_debug_dataset_chain_state.tsv}"
GPU_POLL_FILE="${GPU_POLL_FILE:-$STAGE/beta3_gpu_poll.tsv}"

# Remaining work after optional adopted jobs. Format: stage:task.
# stage is mix/stageA/stageB/split; task is 0/1/2 for array stages and x for split.
REMAINING_STEPS="${REMAINING_STEPS:-mix:0 mix:1 mix:2 stageA:0 stageA:1 stageA:2 stageB:0 stageB:1 stageB:2 split:x}"
START_AFTER_JOB="${START_AFTER_JOB:-}"
SUBMITTED_JOBS="${SUBMITTED_JOBS:-}"

mkdir -p "$STAGE" "$MEGOUT"
touch "$STATE_FILE" "$GPU_POLL_FILE"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

debug_job_count() {
  squeue -h -u "$USER" -p debug -o '%i' | wc -l | tr -d ' '
}

submitted_csv() {
  printf '%s\n' "$SUBMITTED_JOBS" | tr ' ' ',' | sed 's/^,*//; s/,*$//'
}

check_failures() {
  local ids
  ids="$(submitted_csv)"
  [ -n "$ids" ] || return 0
  local bad
  bad="$(sacct -n -X -j "$ids" --format=JobIDRaw,State,ExitCode 2>/dev/null \
    | awk '$2 ~ /FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED/ {print}' || true)"
  if [ -n "$bad" ]; then
    log "ERROR: failed debug dataset job(s):"
    printf '%s\n' "$bad"
    exit 1
  fi
}

poll_gpu_snapshot() {
  {
    printf '%s\n' "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sinfo -p normal,low,debug -h -o '%P|%t|%D|%C' | sort
  } | tee -a "$GPU_POLL_FILE"
}

submit_step() {
  local step="$1"
  local stage="${step%%:*}"
  local task="${step##*:}"
  local dep=()
  [ -n "$START_AFTER_JOB" ] && dep=(--dependency="afterok:$START_AFTER_JOB")

  local jid
  case "$stage" in
    mix)
      jid="$(sbatch --parsable --partition=debug --time="$TIME_LIMIT" "${dep[@]}" --array="$task" \
        --export=ALL,STAGE="$STAGE",RECIPES_DIR="$RECIPES_DIR",HPLT_TGT="$HPLT_TGT",GLOSSAPI_TGT="$GLOSSAPI_TGT",REPLAY_TGT="$REPLAY_TGT",MIX_SHARDS="$MIX_SHARDS" \
        dataset/mix_phase_binaries.sbatch)"
      ;;
    stageA)
      jid="$(sbatch --parsable --partition=debug --time="$TIME_LIMIT" "${dep[@]}" --array="$task" \
        --export=ALL,STAGE="$STAGE" dataset/stageA_clean_decontam_binary.sbatch)"
      ;;
    stageB)
      jid="$(sbatch --parsable --partition=debug --time="$TIME_LIMIT" "${dep[@]}" --array="$task" \
        --export=ALL,STAGE="$STAGE" dataset/stageB_anon_preprocess_binary.sbatch)"
      ;;
    split)
      jid="$(sbatch --parsable --partition=debug --time="$TIME_LIMIT" "${dep[@]}" \
        --export=ALL,STAGE="$STAGE" dataset/split_replay_final_and_tokenize.sbatch)"
      ;;
    *)
      log "ERROR: unknown step: $step"
      exit 2
      ;;
  esac
  START_AFTER_JOB="$jid"
  SUBMITTED_JOBS="${SUBMITTED_JOBS:+$SUBMITTED_JOBS }$jid"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$step" "$jid" "${dep[*]:-none}" >> "$STATE_FILE"
  log "submitted $step as $jid dep=${dep[*]:-none}"
}

log "beta3 debug dataset chain starting"
log "stage=$STAGE recipes=$RECIPES_DIR state=$STATE_FILE"
log "adopted submitted jobs: ${SUBMITTED_JOBS:-none}; start_after=${START_AFTER_JOB:-none}"
poll_gpu_snapshot

read -r -a steps <<< "$REMAINING_STEPS"
idx=0
while (( idx < ${#steps[@]} )); do
  check_failures
  count="$(debug_job_count)"
  log "debug submitted jobs=$count next=${steps[$idx]}"
  if (( count < 2 )); then
    submit_step "${steps[$idx]}"
    idx=$((idx + 1))
  else
    poll_gpu_snapshot
    sleep "$POLL_SECONDS"
  fi
done

log "all dataset jobs submitted; waiting for completion"
while true; do
  check_failures
  ids="$(submitted_csv)"
  queued="$(squeue -h -j "$ids" -o '%i %T %M %R' 2>/dev/null || true)"
  if [ -z "$queued" ]; then
    break
  fi
  printf '%s\n' "$queued"
  poll_gpu_snapshot
  sleep "$POLL_SECONDS"
done

check_failures
log "dataset debug chain completed; running output verification"
run_build_py -u dataset/verify_curriculum_outputs.py --stage "$STAGE"

log "dataset verification complete; polling GPU availability for launch planning"
while true; do
  poll_gpu_snapshot
  squeue -u "$USER" -p normal,low,debug -o '%.18i %.9P %.30j %.8T %.10M %.6D %.8C %.10m %R' | sed -n '1,80p'
  sleep "$POLL_SECONDS"
done

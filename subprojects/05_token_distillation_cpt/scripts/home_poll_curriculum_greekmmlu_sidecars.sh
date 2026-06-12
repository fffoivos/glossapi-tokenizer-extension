#!/usr/bin/env bash
# Poll Clariden from home and submit GreekMMLU-only sidecars for curriculum checkpoints.
#
# This is a fallback for periods when the CPU-only xfer partition is unavailable.
# It keeps the watcher loop off Clariden compute nodes and only uses Slurm for the
# actual conversion/native GreekMMLU sidecars.

set -euo pipefail

SSH_TARGET="${SSH_TARGET:-clariden-ln001}"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-/tmp/clariden-ln001-curriculum-%r@%h:%p}"
V2_REMOTE="${V2_REMOTE:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2}"
CADENCE_REMOTE="${CADENCE_REMOTE:-$V2_REMOTE/eval/cadence_curriculum.tsv}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"
MAX_SECONDS="${MAX_SECONDS:-86400}"

if [ "$#" -gt 0 ]; then
  RUN_TAGS=("$@")
elif [ -n "${RUN_TAGS:-}" ]; then
  # shellcheck disable=SC2206
  RUN_TAGS=($RUN_TAGS)
else
  echo "ERROR: pass run tags as args or set RUN_TAGS." >&2
  exit 2
fi

LOG_DIR="${LOG_DIR:-$PWD/logs}"
mkdir -p "$LOG_DIR"
RUN_LOG="${RUN_LOG:-$LOG_DIR/home_greekmmlu_sidecar_watch_$(date -u +%Y%m%dT%H%M%SZ).log}"

ssh_cmd=(
  ssh
  -S "$SSH_CONTROL_PATH"
  -o ControlMaster=auto
  -o BatchMode=yes
  -o ConnectTimeout=20
  "$SSH_TARGET"
)

run_once() {
  local tags_joined
  tags_joined="${RUN_TAGS[*]}"
  "${ssh_cmd[@]}" \
    "RUN_TAGS='$tags_joined' V2='$V2_REMOTE' CADENCE='$CADENCE_REMOTE' bash -s" <<'REMOTE'
set -euo pipefail
source "$V2/paths.env"
SUBMIT="$SUB/scripts/submit_td_checkpoint_sidecars.sh"
test -x "$SUBMIT"
test -f "$CADENCE"

submitted_now=0
already_done=0
waiting=0

for tag in $RUN_TAGS; do
  train_dir="$RUN_ROOT/$tag"
  state_dir="$RUN_ROOT/${tag}_sidecar_watch"
  mkdir -p "$state_dir"
  while read -r iter tokens label; do
    [ -n "${iter:-}" ] || continue
    case "$iter" in \#*) continue ;; esac
    iter_pad="$(printf "%07d" "$iter")"
    state_file="$state_dir/iter_${iter}.submitted"
    ckpt_meta="$train_dir/checkpoints/iter_${iter_pad}/.metadata"
    if [ -f "$state_file" ]; then
      already_done=$((already_done + 1))
      continue
    fi
    if [ ! -f "$ckpt_meta" ]; then
      waiting=$((waiting + 1))
      continue
    fi

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submit GreekMMLU sidecars: tag=$tag iter=$iter label=$label"
    (
      export HF_TOKENIZER_DIR="$EXT_HF_TOK"
      export NATIVE_BENCHMARKS=greekmmlu
      export SUBMIT_NATIVE=1
      export SUBMIT_GREEK_NLP=0
      export SUBMIT_BPB=0
      export SUBMIT_RETENTION=0
      export SUBMIT_CHECKSUM=0
      export DEFAULT_CODE_HELDOUT_JSONL=/nonexistent/disable_code_bpb.jsonl
      export DEFAULT_MATH_HELDOUT_JSONL=/nonexistent/disable_math_bpb.jsonl
      export CODE_HELDOUT_JSONL=
      export MATH_HELDOUT_JSONL=
      "$SUBMIT" \
        --train-run-dir "$train_dir" \
        --run-tag "$tag" \
        --arm td \
        --iteration "$iter" \
        --tokens "$tokens" \
        --checkpoint-label "$label"
    ) 2>&1 | tee "$state_dir/iter_${iter}_submit.log"
    touch "$state_file"
    submitted_now=$((submitted_now + 1))
  done < "$CADENCE"
done

total_required=$(( $(wc -l < "$CADENCE") * $(printf "%s\n" $RUN_TAGS | wc -l) ))
total_done=0
for tag in $RUN_TAGS; do
  for state in "$RUN_ROOT/${tag}_sidecar_watch"/iter_*.submitted; do
    [ -f "$state" ] || continue
    total_done=$((total_done + 1))
  done
done

echo "summary submitted_now=$submitted_now already_seen_this_pass=$already_done waiting_this_pass=$waiting total_done=$total_done total_required=$total_required"
if [ "$total_done" -ge "$total_required" ]; then
  echo "all required sidecars submitted"
fi
REMOTE
}

start_epoch="$(date +%s)"
echo "=== home_poll_curriculum_greekmmlu_sidecars ===" | tee -a "$RUN_LOG"
date -u | tee -a "$RUN_LOG"
printf "ssh_target=%s\nv2_remote=%s\ncadence=%s\nsleep=%s\nmax_seconds=%s\n" \
  "$SSH_TARGET" "$V2_REMOTE" "$CADENCE_REMOTE" "$SLEEP_SECONDS" "$MAX_SECONDS" | tee -a "$RUN_LOG"
printf "run_tags=%s\n" "${RUN_TAGS[*]}" | tee -a "$RUN_LOG"

while true; do
  now="$(date +%s)"
  elapsed=$((now - start_epoch))
  if [ "$elapsed" -ge "$MAX_SECONDS" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] max watch window reached: ${elapsed}s" | tee -a "$RUN_LOG"
    exit 0
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] poll elapsed=${elapsed}s" | tee -a "$RUN_LOG"
  poll_out="$(run_once 2>&1)" || {
    status=$?
    printf "%s\n" "$poll_out" | tee -a "$RUN_LOG"
    echo "poll failed with status $status" | tee -a "$RUN_LOG"
    sleep "$SLEEP_SECONDS"
    continue
  }
  printf "%s\n" "$poll_out" | tee -a "$RUN_LOG"
  if printf "%s\n" "$poll_out" | grep -q "all required sidecars submitted"; then
    exit 0
  fi
  sleep "$SLEEP_SECONDS"
done

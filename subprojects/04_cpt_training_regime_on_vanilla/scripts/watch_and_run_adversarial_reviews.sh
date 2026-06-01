#!/usr/bin/env bash
# Home-side watcher that runs Codex adversarial critiques after eval sidecars.
#
# Run this on home, not on Clariden. It polls Clariden for per-checkpoint
# sidecar handoff evidence, waits until the hardened verifier says the
# checkpoint handoff is ready, then invokes run_checkpoint_adversarial_review.sh
# locally.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt}"
REMOTE_HOST="${REMOTE_HOST:-clariden}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-$RUN_ROOT/$RUN_TAG}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/eval_${RUN_TAG}}"
STATE_DIR="${STATE_DIR:-$REPO_ROOT/subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/${RUN_TAG}_watch_state}"
SLEEP_SECONDS="${SLEEP_SECONDS:-600}"
ONCE="${ONCE:-0}"

mkdir -p "$STATE_DIR"

checkpoints=(
  "119 499122176 Vanilla-0.5B"
  "238 998244352 Vanilla-1B"
  "477 2000683008 Vanilla-2B"
  "834 3498049536 Vanilla-3.5B"
  "1192 4999610368 Vanilla-5B"
)

echo "=== watch_and_run_adversarial_reviews.sh ==="
date -u
echo "RUN_TAG: $RUN_TAG"
echo "TRAIN_RUN_DIR: $TRAIN_RUN_DIR"
echo "EVAL_ROOT: $EVAL_ROOT"
echo "STATE_DIR: $STATE_DIR"
echo "REMOTE_HOST: $REMOTE_HOST"
echo

all_done() {
  local item iter tokens label done_file
  for item in "${checkpoints[@]}"; do
    read -r iter tokens label <<< "$item"
    done_file="$STATE_DIR/iter_${iter}.review_done"
    if [ ! -f "$done_file" ]; then
      return 1
    fi
  done
  return 0
}

summary_field() {
  local json_path="$1"
  local field="$2"
  python3 - "$json_path" "$field" <<'PY'
import json
import sys

path, field = sys.argv[1:3]
with open(path, encoding="utf-8") as f:
    state = json.load(f)
value = state["summary"][field]
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

while true; do
  for item in "${checkpoints[@]}"; do
    read -r iter tokens label <<< "$item"
    iter_pad="$(printf "%07d" "$iter")"
    remote_iter_root="$EVAL_ROOT/iter_$iter_pad"
    hf_checkpoint_dir="$EVAL_ROOT/iter_${iter_pad}_hf"
    local_done="$STATE_DIR/iter_${iter}.review_done"
    verify_json="$STATE_DIR/iter_${iter}.pre_review_verify.json"
    if [ -f "$local_done" ]; then
      continue
    fi

    if ! "$SCRIPT_DIR/verify_checkpoint_sidecars.py" \
      --iteration "$iter" \
      --run-tag "$RUN_TAG" \
      --remote-host "$REMOTE_HOST" \
      --output "$verify_json"; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] WARN iter=$iter verifier failed before review"
      continue
    fi

    handoff_ready="$(summary_field "$verify_json" handoff_ready)"
    active_jobs="$(summary_field "$verify_json" active_sidecar_jobs)"
    outputs_ready="$(summary_field "$verify_json" expected_outputs_ready)"
    checksum_ready="$(summary_field "$verify_json" checksum_manifest_ready)"
    if [ "$handoff_ready" != "true" ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] iter=$iter waiting for handoff_ready; active_sidecar_jobs=$active_jobs expected_outputs_ready=$outputs_ready checksum_manifest_ready=$checksum_ready"
      continue
    fi

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running adversarial review for iter=$iter label=$label"
    "$SCRIPT_DIR/run_checkpoint_adversarial_review.sh" \
      --checkpoint-label "$label" \
      --iteration "$iter" \
      --tokens "$tokens" \
      --train-run-dir "$TRAIN_RUN_DIR" \
      --eval-root "$EVAL_ROOT" \
      --hf-checkpoint-dir "$hf_checkpoint_dir"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$local_done"
  done

  if all_done; then
    echo "all adversarial reviews complete"
    date -u
    exit 0
  fi
  if [ "$ONCE" = "1" ]; then
    echo "ONCE=1; exiting after one polling pass"
    exit 0
  fi
  sleep "$SLEEP_SECONDS"
done

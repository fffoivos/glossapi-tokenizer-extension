#!/usr/bin/env bash
# Home-side watcher that records checkpoint sidecar handoff verification.
#
# This is read-only: it calls verify_checkpoint_sidecars.py for each planned
# checkpoint and records when checkpoint metadata, expected sidecar outputs,
# completed Slurm jobs, and checksum manifest are present. It does not submit,
# cancel, or modify Slurm jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBPROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-04_vanilla_goldfish_5b_20260528T112539Z}"
REMOTE_HOST="${REMOTE_HOST:-clariden}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
ONCE="${ONCE:-0}"

STATE_DIR="${STATE_DIR:-$SUBPROJECT_DIR/monitor_logs/${RUN_TAG}_sidecar_verify_state}"
REPORT_DIR="${REPORT_DIR:-$SUBPROJECT_DIR/reports}"

mkdir -p "$STATE_DIR" "$REPORT_DIR"

checkpoints=(
  "119 Vanilla-0.5B"
  "238 Vanilla-1B"
  "477 Vanilla-2B"
  "834 Vanilla-3.5B"
  "1192 Vanilla-5B"
)

echo "=== watch_checkpoint_sidecar_verification.sh ==="
date -u
echo "RUN_TAG: $RUN_TAG"
echo "REMOTE_HOST: $REMOTE_HOST"
echo "STATE_DIR: $STATE_DIR"
echo "REPORT_DIR: $REPORT_DIR"
echo "SLEEP_SECONDS: $SLEEP_SECONDS"
echo

all_done() {
  local item iter label
  for item in "${checkpoints[@]}"; do
    read -r iter label <<< "$item"
    if [ ! -f "$STATE_DIR/iter_${iter}.handoff_done" ]; then
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
    read -r iter label <<< "$item"
    iter_pad="$(printf "%07d" "$iter")"
    done_file="$STATE_DIR/iter_${iter}.handoff_done"
    latest_json="$REPORT_DIR/iter_${iter_pad}_checkpoint_sidecar_verify_latest.json"
    pass_json="$REPORT_DIR/iter_${iter_pad}_checkpoint_sidecar_handoff_pass.json"

    if [ -f "$done_file" ]; then
      continue
    fi

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] checking iter=$iter label=$label"
    if ! "$SCRIPT_DIR/verify_checkpoint_sidecars.py" \
      --iteration "$iter" \
      --run-tag "$RUN_TAG" \
      --remote-host "$REMOTE_HOST" \
      --output "$latest_json"; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] WARN iter=$iter verifier failed"
      continue
    fi

    checkpoint_ready="$(summary_field "$latest_json" checkpoint_metadata_ready)"
    sidecars_submitted="$(summary_field "$latest_json" sidecars_submitted)"
    manifest_complete="$(summary_field "$latest_json" manifest_has_all_expected_kinds)"
    outputs_ready="$(summary_field "$latest_json" expected_outputs_ready)"
    active_jobs="$(summary_field "$latest_json" active_sidecar_jobs)"
    slurm_done="$(summary_field "$latest_json" slurm_jobs_completed)"
    checksum_ready="$(summary_field "$latest_json" checksum_manifest_ready)"
    handoff_ready="$(summary_field "$latest_json" handoff_ready)"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] iter=$iter checkpoint_metadata_ready=$checkpoint_ready sidecars_submitted=$sidecars_submitted manifest_has_all_expected_kinds=$manifest_complete expected_outputs_ready=$outputs_ready active_sidecar_jobs=$active_jobs slurm_jobs_completed=$slurm_done checksum_manifest_ready=$checksum_ready handoff_ready=$handoff_ready"

    if [ "$handoff_ready" = "true" ]; then
      cp "$latest_json" "$pass_json"
      date -u +%Y-%m-%dT%H:%M:%SZ > "$done_file"
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] iter=$iter handoff verified; wrote $pass_json"
    fi
  done

  if all_done; then
    echo "all checkpoint handoffs verified"
    date -u
    exit 0
  fi
  if [ "$ONCE" = "1" ]; then
    echo "ONCE=1; exiting after one polling pass"
    exit 0
  fi
  sleep "$SLEEP_SECONDS"
done

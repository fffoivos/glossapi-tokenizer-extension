#!/usr/bin/env bash
# Lightweight status logger for the 2026-05-25 5B TD-vs-Vanilla continuation.
# This is coordination-only: it runs on home and only queries Clariden status.

set -uo pipefail

LOG_DIR="${LOG_DIR:-/home/foivos/runs/codex_monitors/5b_td_vs_vanilla_20260525}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/monitor.log}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
MAX_POLLS="${MAX_POLLS:-288}"
TRACKED_JOBS="${TRACKED_JOBS:-2382982,2382983,2382984,2382985,2382986,2382998,2382999,2383000,2383001,2383002,2383003}"
STATE_FILE="${STATE_FILE:-/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_sidecar_eval_incremental/eval_sidecar_incremental_state.tsv}"

mkdir -p "$LOG_DIR"

for _ in $(seq 1 "$MAX_POLLS"); do
    {
        echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        ssh clariden "squeue -u \"\$USER\" -o \"%.18i %.9P %.36j %.8T %.10M %.10l %.6D %R\""
        ssh clariden 'for f in /capstor/scratch/cscs/fffoivos/runs/bakeoff/5b_vanilla_1013-2382982.out /capstor/scratch/cscs/fffoivos/runs/bakeoff/5b_td_layer11_1013-2382984.out /capstor/scratch/cscs/fffoivos/runs/bakeoff/5b_vanilla_1192-2382983.out /capstor/scratch/cscs/fffoivos/runs/bakeoff/5b_td_layer11_1192-2382985.out; do [ -f "$f" ] || continue; echo "--- $(basename "$f")"; grep -nE "iteration +[0-9]+/ +(1013|1192)|saving checkpoint|successfully saved|exiting|done" "$f" | tail -3 || true; done'
        ssh clariden "state='$STATE_FILE'; if [ -f \"\$state\" ]; then echo \"sidecar_rows=\$(( \$(wc -l < \"\$state\") - 1 ))\"; tail -8 \"\$state\"; fi"
    } >> "$LOG_FILE" 2>&1

    active_count="$(ssh clariden "squeue -h -j '$TRACKED_JOBS' | wc -l" 2>>"$LOG_FILE" || echo unknown)"
    echo "tracked_active_count=$active_count" >> "$LOG_FILE"
    if [ "$active_count" = "0" ]; then
        echo "all tracked jobs left queue at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_FILE"
        exit 0
    fi
    sleep "$INTERVAL_SECONDS"
done

#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/continuation_3p5b_20260524T143012Z_home_watcher"
LOG_FILE="$LOG_DIR/watcher.log"
REMOTE_EVAL_DIR="/iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval"
RUN_TAG="continuation_3p5b_20260524T143012Z"
TRAINING_CHAIN_TSV="/capstor/scratch/cscs/fffoivos/runs/bakeoff/continuation_3p5b_20260524T143012Z_submit_state/training_chain.tsv"
TRAINING_CHAIN_OVERLAY_TSV="/capstor/scratch/cscs/fffoivos/runs/bakeoff/continuation_3p5b_20260524T143012Z_submit_state/training_chain_repair_20260524T1836Z.tsv"
STATE_DIR="/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_sidecar_eval_incremental"
STATE_FILE="$STATE_DIR/eval_sidecar_incremental_state.tsv"
OUT_ROOT="/capstor/scratch/cscs/fffoivos/runs/eval"

mkdir -p "$LOG_DIR"

while true; do
    {
        echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) repaired ==="
        ssh clariden "cd '$REMOTE_EVAL_DIR' && RUN_TAG='$RUN_TAG' TRAINING_CHAIN_TSV='$TRAINING_CHAIN_TSV' TRAINING_CHAIN_OVERLAY_TSV='$TRAINING_CHAIN_OVERLAY_TSV' OUT_ROOT='$OUT_ROOT' STATE_DIR='$STATE_DIR' MAX_SUBMITTED_JOBS=14 LOOP=0 python3 submit_3p5b_eval_sidecars_incremental.py"
        submit_status=$?
        ssh clariden "state='$STATE_FILE'; if [ -f \"\$state\" ]; then lines=\$(wc -l < \"\$state\"); echo submitted_count=\$((lines - 1)); if [ \"\$lines\" -ge 28 ]; then exit 42; fi; else echo submitted_count=0; fi"
        done_status=$?
        echo "submit_exit=$submit_status done_check=$done_status"
    } >> "$LOG_FILE" 2>&1
    if [ "$done_status" -eq 42 ]; then
        exit 0
    fi
    sleep 120
done

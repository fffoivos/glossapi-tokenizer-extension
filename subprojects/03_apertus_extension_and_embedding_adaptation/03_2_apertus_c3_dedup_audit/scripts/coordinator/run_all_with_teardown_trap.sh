#!/usr/bin/env bash
# Coordinator orchestrator: run the entire dedup pipeline end-to-end, with a
# `trap 99_teardown.sh EXIT` so workers ALWAYS get torn down — whether the
# run completes, errors out, or is SIGINT'd.
#
# Per review r4: previous instructions in READY_TO_SPIN_UP.md listed 9 manual
# steps; if poll loop exited abnormally, workers would keep burning. This
# wrapper makes teardown unconditional.
#
# Usage:  bash run_all_with_teardown_trap.sh
#
# Required env at invocation:  HF_TOKEN
# Required state: pre_flight already executed and PASSED;
#                 partition + workers spun up via separate steps before
#                 invoking this driver (or pass --include-spinup; not default
#                 to keep the cost-event step explicit).
set -uo pipefail

SUB="/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_2_apertus_c3_dedup_audit"
RUN_ID="${RUN_ID:-$(cat "$SUB/manifests/CURRENT_RUN_ID")}"
MANI="$SUB/manifests/run_$RUN_ID"
PY=/home/foivos/.venvs/glossapi-merge-docling/bin/python3
COORD="$SUB/scripts/coordinator"

cd "$SUB"
LOG="$MANI/run_all.log"
echo "=== run_all start $(date -u +%Y-%m-%dT%H:%M:%SZ) RUN_ID=$RUN_ID ===" | tee -a "$LOG"

# Teardown trap — runs on any exit path (success, error, SIGINT, SIGTERM).
teardown() {
  local rc=$?
  echo "=== run_all exit handler firing (rc=$rc) $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  bash "$COORD/99_teardown.sh" 2>&1 | tee -a "$LOG" || true
  echo "=== run_all exit handler done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  exit "$rc"
}
trap teardown EXIT INT TERM

INCLUDE_SPINUP=0
for arg in "$@"; do
  case "$arg" in
    --include-spinup) INCLUDE_SPINUP=1 ;;
  esac
done

set -e

if [ "$INCLUDE_SPINUP" = "1" ]; then
  echo "[run_all] step 01 partition" | tee -a "$LOG"
  "$PY" "$COORD/01_bytes_balanced_partition.py" 2>&1 | tee -a "$LOG"
  echo "[run_all] step 02 spin_up (COST EVENT)" | tee -a "$LOG"
  bash "$COORD/02_spin_up_workers.sh" 2>&1 | tee -a "$LOG"
fi

echo "[run_all] step 03 dispatch" | tee -a "$LOG"
bash "$COORD/03_dispatch_bootstrap.sh" 2>&1 | tee -a "$LOG"

echo "[run_all] step 04 poll_and_collect (foreground)" | tee -a "$LOG"
"$PY" "$COORD/04_poll_and_collect.py" 2>&1 | tee -a "$LOG"

echo "[run_all] step 05 concat_per_source" | tee -a "$LOG"
"$PY" "$COORD/05_concat_per_source.py" 2>&1 | tee -a "$LOG"

echo "[run_all] step 06 exact_overlap_join" | tee -a "$LOG"
"$PY" "$COORD/06_exact_overlap_join.py" 2>&1 | tee -a "$LOG"

echo "[run_all] step 07 minhash_overlap_lsh" | tee -a "$LOG"
"$PY" "$COORD/07_minhash_overlap_lsh.py" 2>&1 | tee -a "$LOG"

echo "[run_all] step 08 holdout_contamination_check (non-fatal if no holdout list)" | tee -a "$LOG"
"$PY" "$COORD/08_holdout_contamination_check.py" 2>&1 | tee -a "$LOG" || true

echo "[run_all] step 09 build_summary_report" | tee -a "$LOG"
"$PY" "$COORD/09_build_summary_report.py" 2>&1 | tee -a "$LOG"

echo "=== run_all main pipeline complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
# trap fires here, calling 99_teardown.sh. exit rc 0 will propagate.

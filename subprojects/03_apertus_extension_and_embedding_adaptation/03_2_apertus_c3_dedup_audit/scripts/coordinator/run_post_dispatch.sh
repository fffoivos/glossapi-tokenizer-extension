#!/usr/bin/env bash
# Runs steps 04 → 09 (joins + report) then 99 (teardown).
# Use after 03_dispatch_bootstrap.sh has fired and workers are running.
# Teardown trap fires on any exit path so workers don't linger on error.
set -uo pipefail

SUB="/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_2_apertus_c3_dedup_audit"
RUN_ID="${RUN_ID:-$(cat "$SUB/manifests/CURRENT_RUN_ID")}"
MANI="$SUB/manifests/run_$RUN_ID"
PY=/home/foivos/.venvs/glossapi-merge-docling/bin/python3
COORD="$SUB/scripts/coordinator"
LOG="$MANI/run_post_dispatch.log"

cd "$SUB"
echo "=== run_post_dispatch start $(date -u +%Y-%m-%dT%H:%M:%SZ) RUN_ID=$RUN_ID ===" | tee -a "$LOG"

teardown() {
  local rc=$?
  echo "=== run_post_dispatch exit handler firing (rc=$rc) $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  bash "$COORD/99_teardown.sh" 2>&1 | tee -a "$LOG" || true
  echo "=== run_post_dispatch exit handler done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  exit "$rc"
}
trap teardown EXIT INT TERM

set -e

echo "[post] step 04 poll_and_collect" | tee -a "$LOG"
"$PY" "$COORD/04_poll_and_collect.py" 2>&1 | tee -a "$LOG"

echo "[post] step 05 concat_per_source" | tee -a "$LOG"
"$PY" "$COORD/05_concat_per_source.py" 2>&1 | tee -a "$LOG"

echo "[post] step 06 exact_overlap_join" | tee -a "$LOG"
"$PY" "$COORD/06_exact_overlap_join.py" 2>&1 | tee -a "$LOG"

echo "[post] step 07 minhash_overlap_lsh" | tee -a "$LOG"
"$PY" "$COORD/07_minhash_overlap_lsh.py" 2>&1 | tee -a "$LOG"

echo "[post] step 08 holdout_contamination_check (non-fatal if no holdout list)" | tee -a "$LOG"
"$PY" "$COORD/08_holdout_contamination_check.py" 2>&1 | tee -a "$LOG" || true

echo "[post] step 09 build_summary_report" | tee -a "$LOG"
"$PY" "$COORD/09_build_summary_report.py" 2>&1 | tee -a "$LOG"

echo "=== run_post_dispatch main pipeline complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

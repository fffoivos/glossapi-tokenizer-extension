#!/usr/bin/env bash
# Close one exact accelerated signature array before merging its manifest.
set -eEuo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"
: "${PRODUCTION_ATTEMPT_ROOT:?PRODUCTION_ATTEMPT_ROOT is required}"
: "${ARRAY_JOB_ID:?ARRAY_JOB_ID is required}"
: "${ARRAY_SPEC:?ARRAY_SPEC is required}"
: "${ATTEMPT_ID:?ATTEMPT_ID is required}"

[[ "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$ARRAY_SPEC" =~ ^[0-9]+-[0-9]+%[0-9]+$ ]]
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9._-]{5,63}$ ]]

coord_root="${COORD_ROOT:-$(dirname "$RUN_ROOT")/.${RUN_ROOT##*/}.coord}"
python="$coord_root/runtime/venv/bin/python"
acceleration="$PIPELINE_ROOT/scripts/agent1_v5_dedup_acceleration.py"
capture="$PIPELINE_ROOT/slurm/agent1_v5_eiger/capture_signature_array_evidence.sh"
stage="$PIPELINE_ROOT/slurm/agent1_v5_eiger/stage.sh"
scheduler_evidence="$PRODUCTION_ATTEMPT_ROOT/scheduler-evidence.json"
execution_receipt="$PRODUCTION_ATTEMPT_ROOT/execution.json"
submission="$PRODUCTION_ATTEMPT_ROOT/submission.json"
authorization="$PRODUCTION_ATTEMPT_ROOT/release_authorization.json"
chunk_plan="$PRODUCTION_ATTEMPT_ROOT/chunks.json"
metrics_root="$PRODUCTION_ATTEMPT_ROOT/metrics"
replacement_recovery="$(jq -r '.pending_replacement.recovery_receipt_path // empty' "$chunk_plan" 2>/dev/null || true)"

for path in "$acceleration" "$capture" "$stage" "$submission" \
  "$authorization" "$chunk_plan"; do
  [[ -e "$path" ]] || { echo "required closure input is missing: $path" >&2; exit 2; }
done
uenv run pytorch/v2.6.0:v1 --view=default -- test -x "$python" || {
  echo "required closure Python is unavailable inside uenv: $python" >&2
  exit 2
}
if [[ -n "$replacement_recovery" ]]; then
  [[ -f "$replacement_recovery" ]] || { echo "pending replacement recovery is missing" >&2; exit 2; }
  predecessor_execution="$PRODUCTION_ATTEMPT_ROOT/predecessor-execution.json"
  if [[ ! -e "$predecessor_execution" ]]; then
    uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
      "$python" "$acceleration" validate-pending-replacement-predecessor \
      --run-root "$RUN_ROOT" \
      --recovery-receipt "$replacement_recovery" \
      --output "$predecessor_execution"
  fi
  jq -e '.status == "passed" and .rank_count > 0' "$predecessor_execution" >/dev/null
fi
if [[ -e "$scheduler_evidence" ]]; then
  jq -e --arg job "$ARRAY_JOB_ID" --arg spec "$ARRAY_SPEC" --arg attempt "$ATTEMPT_ID" \
    '.status == "passed" and .array_job_id == $job and .array_spec == $spec
     and .attempt_id == $attempt and .expected_state == "COMPLETED"
     and .expected_exit_code == "0:0"' "$scheduler_evidence" >/dev/null
else
  "$capture" "$ARRAY_JOB_ID" "$ARRAY_SPEC" "$ATTEMPT_ID" COMPLETED 0:0 \
    "$scheduler_evidence"
fi

if [[ -e "$execution_receipt" ]]; then
  chunk_sha="$(sha256sum "$chunk_plan" | awk '{print $1}')"
  expected_ranks="$(jq '[.chunks[].ranks[]] | length' "$chunk_plan")"
  jq -e --arg job "$ARRAY_JOB_ID" --arg attempt "$ATTEMPT_ID" \
    --arg run "$RUN_ROOT" --arg chunk_sha "$chunk_sha" --argjson ranks "$expected_ranks" \
    '.status == "passed" and .array_job_id == $job and .attempt_id == $attempt
     and .run_root == $run and .chunk_plan_sha256 == $chunk_sha
     and .rank_count == $ranks' "$execution_receipt" >/dev/null
else
  uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
    "$python" "$acceleration" validate-attempt-execution \
    --run-root "$RUN_ROOT" \
    --submission-receipt "$submission" \
    --release-authorization "$authorization" \
    --chunk-plan "$chunk_plan" \
    --scheduler-evidence "$scheduler_evidence" \
    --metrics-root "$metrics_root" \
    --output "$execution_receipt"
fi

# merge-signatures performs the final 431-receipt / 13,792-output hash closure.
if [[ ! -e "$RUN_ROOT/signature_manifest.json" ]]; then
  uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
    STAGE=merge-signatures \
    PIPELINE_ROOT="$PIPELINE_ROOT" \
    RUN_ROOT="$RUN_ROOT" \
    VENV_ROOT="$coord_root/runtime/venv" \
    "$stage"
fi

jq -e '.status == "passed" and .task_count == 431 and .bucket_count == 32' \
  "$RUN_ROOT/signature_manifest.json" >/dev/null
printf 'signature execution and manifest closure passed\n'

#!/usr/bin/env bash
# Run one immutable normal-partition signature chunk with bounded local workers.
# The production path accepts work only from a nonce-bound held array receipt.
set -eEuo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"
: "${CHUNK_PLAN:?CHUNK_PLAN is required}"
: "${CHUNK_PLAN_SHA256:?CHUNK_PLAN_SHA256 is required}"
: "${FULL_INPUT_AUDIT:?FULL_INPUT_AUDIT is required}"
: "${WORKERS:?WORKERS is required}"
: "${SUBMISSION_NONCE:?SUBMISSION_NONCE is required}"

benchmark_mode="${BENCHMARK_MODE:-0}"
chunk_index="${SLURM_ARRAY_TASK_ID:-0}"
case "$WORKERS" in 1|2|3|4|5) ;; *) echo "invalid WORKERS=$WORKERS" >&2; exit 2 ;; esac
case "$benchmark_mode" in 0|1) ;; *) echo "invalid BENCHMARK_MODE=$benchmark_mode" >&2; exit 2 ;; esac

[[ "$(sha256sum "$CHUNK_PLAN" | awk '{print $1}')" == "$CHUNK_PLAN_SHA256" ]]
[[ -f "$FULL_INPUT_AUDIT" ]]

config="$PIPELINE_ROOT/configs/agent1_v5_eiger_pipeline.json"
dedup="$PIPELINE_ROOT/scripts/agent1_v5_datatrove.py"
acceleration="$PIPELINE_ROOT/scripts/agent1_v5_dedup_acceleration.py"
contract="$RUN_ROOT/run_contract.json"
manifest="$RUN_ROOT/release-pre-dedup/manifests/combined_manifest.json"
runtime="$RUN_ROOT/datatrove_runtime.json"
coord="$(dirname "$RUN_ROOT")/.${RUN_ROOT##*/}.coord"
python="$coord/runtime/venv/bin/python"
uenv_python=(
  uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME
  "$python"
)
attempt_id="${ATTEMPT_ID:-benchmark}"
recovery_receipt_sha256="${RECOVERY_RECEIPT_SHA256:-benchmark}"
metrics_root="${METRICS_ROOT:-$RUN_ROOT/60-dedup/minhash-signatures/accelerated-metrics}"
logs_root="${LOGS_ROOT:-$RUN_ROOT/60-dedup/minhash-signatures/accelerated-rank-logs}"
stop_sentinel="${STOP_SENTINEL:-$RUN_ROOT/dedup_acceleration.stop}"
mkdir -p "$metrics_root" "$logs_root"

if [[ "$benchmark_mode" == 0 ]]; then
  : "${SUBMISSION_RECEIPT:?SUBMISSION_RECEIPT is required for production work}"
  : "${RELEASE_AUTHORIZATION:?RELEASE_AUTHORIZATION is required for production work}"
  : "${SLURM_ARRAY_JOB_ID:?array job ID is required for production work}"
  : "${ATTEMPT_ID:?ATTEMPT_ID is required for production work}"
  : "${RECOVERY_RECEIPT_SHA256:?RECOVERY_RECEIPT_SHA256 is required for production work}"
  "${uenv_python[@]}" "$acceleration" validate-worker-authorization \
    --submission-receipt "$SUBMISSION_RECEIPT" \
    --release-authorization "$RELEASE_AUTHORIZATION" \
    --array-job-id "$SLURM_ARRAY_JOB_ID" \
    --submission-nonce "$SUBMISSION_NONCE" \
    --chunk-plan-sha256 "$CHUNK_PLAN_SHA256" \
    --runner "${BASH_SOURCE[0]}" \
    --workers "$WORKERS" \
    --attempt-id "$ATTEMPT_ID" \
    --recovery-receipt-sha256 "$RECOVERY_RECEIPT_SHA256"
fi

if [[ "$benchmark_mode" == 1 ]]; then
  mapfile -t ranks < <(
    jq -er --argjson index "$chunk_index" --argjson workers "$WORKERS" \
      'select(.schema_version == "agent1_v5_dedup_acceleration_benchmark_plan_v1"
              and .status == "passed")
       | (.phases[] | select(.index == $index and .workers == $workers) | .ranks[])' \
      "$CHUNK_PLAN"
  )
else
  mapfile -t ranks < <(
    jq -er --argjson index "$chunk_index" --argjson workers "$WORKERS" \
      'select(.schema_version == "agent1_v5_dedup_acceleration_chunk_plan_v1"
              and .status == "passed" and .selected_workers == $workers)
       | (.chunks[] | select(.index == $index) | .ranks[])' "$CHUNK_PLAN"
  )
fi
(( ${#ranks[@]} > 0 ))

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

drain_requested=0
trap 'drain_requested=1' USR1
drain_epoch="${SLURM_JOB_END_TIME:-0}"
if [[ "$drain_epoch" =~ ^[0-9]+$ ]] && (( drain_epoch > 0 )); then
  drain_epoch=$((drain_epoch - 900))
fi

should_drain() {
  [[ -e "$stop_sentinel" ]] && return 0
  (( drain_requested == 1 )) && return 0
  (( drain_epoch > 0 && $(date +%s) >= drain_epoch )) && return 0
  return 1
}

run_rank() {
  local rank="$1" start end status metric token time_report input_bytes input_rows out_log err_log
  start="$(date +%s)"
  read -r input_bytes input_rows < <(
    jq -er --argjson rank "$rank" '.files[] | select(.rank == $rank) | "\(.bytes) \(.rows)"' "$manifest"
  )
  token="${SUBMISSION_NONCE}:${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}:${chunk_index}:${rank}"
  time_report="$(mktemp "$metrics_root/.time-${rank}.XXXXXX")"
  out_log="$logs_root/chunk-$(printf '%04d' "$chunk_index")-rank-$(printf '%06d' "$rank").out"
  err_log="$logs_root/chunk-$(printf '%04d' "$chunk_index")-rank-$(printf '%06d' "$rank").err"
  if /usr/bin/time -v -o "$time_report" \
      uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
      "$python" "$dedup" accelerated-signature-task \
      --config "$config" --contract "$contract" --combined-manifest "$manifest" \
      --runtime-receipt "$runtime" --full-input-audit "$FULL_INPUT_AUDIT" \
      --task-index "$rank" --claim-token "$token" >"$out_log" 2>"$err_log"; then
    status=passed
  else
    status=failed
  fi
  end="$(date +%s)"
  metric="$metrics_root/chunk-$(printf '%04d' "$chunk_index")-rank-$(printf '%06d' "$rank").json"
  python3 - "$metric" "$rank" "$start" "$end" "$status" "$chunk_index" "$WORKERS" \
    "$input_bytes" "$input_rows" "$time_report" "$benchmark_mode" "$CHUNK_PLAN_SHA256" "$out_log" "$err_log" \
    "$attempt_id" "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}" "$SUBMISSION_NONCE" <<'PY'
import json, os, re, sys, tempfile
(
    path, rank, start, end, status, chunk, workers, input_bytes, input_rows,
    time_report, benchmark_mode, plan_sha, out_log, err_log, attempt_id,
    array_job_id, submission_nonce,
) = sys.argv[1:]
values = {}
for raw in open(time_report, encoding="utf-8", errors="replace"):
    match = re.match(r"\s*([^:]+):\s*(.*)$", raw)
    if match:
        values[match.group(1).strip()] = match.group(2).strip()
def integer(label):
    value = values.get(label, "0").replace(",", "")
    return int(value) if value.isdigit() else 0
payload = {
    "schema_version": "agent1_v5_accelerated_signature_metric_v1",
    "status": status,
    "rank": int(rank),
    "chunk_index": int(chunk),
    "phase_index": int(chunk) if benchmark_mode == "1" else None,
    "workers": int(workers),
    "started_epoch": int(start),
    "finished_epoch": int(end),
    "elapsed_seconds": int(end) - int(start),
    "input_bytes": int(input_bytes),
    "input_rows": int(input_rows),
    "max_rss_kib": integer("Maximum resident set size (kbytes)"),
    "user_cpu_seconds": values.get("User time (seconds)", "0"),
    "system_cpu_seconds": values.get("System time (seconds)", "0"),
    "filesystem_input_kib": integer("File system inputs"),
    "filesystem_output_kib": integer("File system outputs"),
    "benchmark_plan_sha256": plan_sha if benchmark_mode == "1" else None,
    "chunk_plan_sha256": plan_sha if benchmark_mode == "0" else None,
    "attempt_id": attempt_id,
    "array_job_id": array_job_id,
    "submission_nonce": submission_nonce,
    "stdout": out_log,
    "stderr": err_log,
}
fd, temporary = tempfile.mkstemp(prefix=".metric-", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
  rm -f "$time_report"
  [[ "$status" == passed ]]
}

if should_drain; then
  echo "stop sentinel or drain window present before chunk $chunk_index" >&2
  exit 75
fi

pids=()
for rank in "${ranks[@]}"; do
  if should_drain; then
    echo "drain requested before rank $rank" >&2
    break
  fi
  run_rank "$rank" &
  pids+=("$!")
  if (( ${#pids[@]} >= WORKERS )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

if should_drain; then
  exit 75
fi

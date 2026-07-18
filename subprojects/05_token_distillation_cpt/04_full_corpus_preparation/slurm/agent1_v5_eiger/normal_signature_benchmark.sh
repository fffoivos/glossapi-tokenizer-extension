#!/usr/bin/env bash
# Execute the immutable 1→2→4→5 signature benchmark on one normal node.
set -eEuo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"
: "${BENCHMARK_PLAN:?BENCHMARK_PLAN is required}"
: "${BENCHMARK_PLAN_SHA256:?BENCHMARK_PLAN_SHA256 is required}"
: "${FULL_INPUT_AUDIT:?FULL_INPUT_AUDIT is required}"
: "${BENCHMARK_NONCE:?BENCHMARK_NONCE is required}"

[[ "$(sha256sum "$BENCHMARK_PLAN" | awk '{print $1}')" == "$BENCHMARK_PLAN_SHA256" ]]
[[ -f "$FULL_INPUT_AUDIT" ]]

runner="$PIPELINE_ROOT/slurm/agent1_v5_eiger/normal_signature_runner.sh"
metrics_root="$RUN_ROOT/60-dedup/minhash-signatures/accelerated-metrics"
sample_root="$RUN_ROOT/60-dedup/minhash-signatures/benchmark-samples"
sample_file="$sample_root/${SLURM_JOB_ID:-local}-${BENCHMARK_NONCE}.ndjson"
phase_file="$sample_root/${SLURM_JOB_ID:-local}-${BENCHMARK_NONCE}.phase"
observations="$RUN_ROOT/dedup_acceleration_benchmark_observations.json"
stop_sentinel="${STOP_SENTINEL:-$RUN_ROOT/dedup_acceleration.stop}"
mkdir -p "$sample_root"
[[ ! -e "$observations" ]] || { echo "immutable observations already exist: $observations" >&2; exit 2; }

sample_once() {
  local epoch phase raw status
  epoch="$(date +%s)"
  phase="$(cat "$phase_file" 2>/dev/null || printf '%s' -1)"
  if raw="$(sstat -n -P -j "${SLURM_JOB_ID}.batch" --format=JobID,AveRSS,MaxRSS,AveDiskRead,AveDiskWrite,TotalCPU 2>&1)"; then
    status=passed
  else
    status=failed
  fi
  jq -cn --argjson epoch "$epoch" --argjson phase "$phase" --arg status "$status" --arg raw "$raw" \
    '{epoch:$epoch,phase_index:$phase,status:$status,raw:$raw}' >> "$sample_file"
}

sampler() {
  while true; do
    sample_once
    sleep 300
  done
}

sampler_pid=""
cleanup() {
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

printf '%s\n' -1 > "$phase_file"
sample_once
sampler &
sampler_pid=$!

for phase in 0 1 2 3; do
  [[ ! -e "$stop_sentinel" ]] || { echo "stop sentinel present before benchmark phase $phase" >&2; exit 75; }
  workers="$(jq -er --argjson phase "$phase" '.phases[] | select(.index == $phase) | .workers' "$BENCHMARK_PLAN")"
  printf '%s\n' "$phase" > "$phase_file"
  sample_once
  BENCHMARK_MODE=1 \
    SLURM_ARRAY_TASK_ID="$phase" \
    RUN_ROOT="$RUN_ROOT" PIPELINE_ROOT="$PIPELINE_ROOT" \
    CHUNK_PLAN="$BENCHMARK_PLAN" CHUNK_PLAN_SHA256="$BENCHMARK_PLAN_SHA256" \
    FULL_INPUT_AUDIT="$FULL_INPUT_AUDIT" WORKERS="$workers" \
    SUBMISSION_NONCE="$BENCHMARK_NONCE" STOP_SENTINEL="$stop_sentinel" \
    "$runner"
  sample_once
done

cleanup
sampler_pid=""

python3 - "$BENCHMARK_PLAN" "$BENCHMARK_PLAN_SHA256" "$metrics_root" "$sample_file" "$observations" <<'PY'
import json, math, os, re, statistics, sys, tempfile
plan_path, plan_sha, metrics_root, sample_path, output = sys.argv[1:]
plan = json.load(open(plan_path, encoding="utf-8"))
samples = [json.loads(line) for line in open(sample_path, encoding="utf-8") if line.strip()]
units = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
def size(raw):
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?)\s*", raw or "")
    if not match:
        return None
    return float(match.group(1)) * units.get(match.group(2), 1)
def disk_values(sample):
    if sample.get("status") != "passed":
        return None
    row = sample.get("raw", "").strip().splitlines()
    if len(row) != 1:
        return None
    fields = row[0].split("|")
    if len(fields) < 5:
        return None
    return size(fields[3]), size(fields[4])
phases = []
for phase in plan["phases"]:
    index = int(phase["index"])
    metrics = []
    for rank in phase["ranks"]:
        path = os.path.join(metrics_root, f"chunk-{index:04d}-rank-{int(rank):06d}.json")
        metrics.append(json.load(open(path, encoding="utf-8")))
    local_samples = [sample for sample in samples if int(sample.get("phase_index", -1)) == index]
    parsed = [disk_values(sample) for sample in local_samples]
    errors = []
    if len(local_samples) < 2:
        errors.append("fewer than two five-minute resource samples")
    if any(value is None or value[0] is None or value[1] is None for value in parsed):
        errors.append("unparseable sstat disk sample")
    # AveDiskRead/AveDiskWrite are byte counts, not rates.  A five-minute
    # rate is therefore computed only from adjacent sampler points 240–330 s
    # apart.  This avoids treating a cumulative accounting value as B/s.
    ordered = sorted(
        (
            (sample, value) for sample, value in zip(local_samples, parsed)
            if value is not None and value[0] is not None and value[1] is not None
        ),
        key=lambda item: int(item[0]["epoch"]),
    )
    read_rates, write_rates = [], []
    for (before, before_values), (after, after_values) in zip(ordered, ordered[1:]):
        elapsed = int(after["epoch"]) - int(before["epoch"])
        if 240 <= elapsed <= 330:
            read_rates.append(max(0.0, after_values[0] - before_values[0]) / elapsed)
            write_rates.append(max(0.0, after_values[1] - before_values[1]) / elapsed)
    if not read_rates or not write_rates:
        errors.append("no adjacent five-minute sstat interval")
    wall = max(metric["finished_epoch"] for metric in metrics) - min(metric["started_epoch"] for metric in metrics)
    if wall <= 0:
        errors.append("non-positive phase wall time")
        wall = 1
    cpu = 0.0
    for metric in metrics:
        try:
            cpu += float(metric.get("user_cpu_seconds", 0)) + float(metric.get("system_cpu_seconds", 0))
        except (TypeError, ValueError):
            errors.append("unparseable worker CPU time")
    phases.append({
        "phase_index": index,
        "sample_count": len(local_samples),
        "aggregate_rss_bytes": sum(int(metric.get("max_rss_kib", 0)) * 1024 for metric in metrics),
        "read_peak_5m_bps": max(read_rates, default=math.inf),
        "write_peak_5m_bps": max(write_rates, default=math.inf),
        "max_cpu_cores": cpu / wall,
        "warnings": [],
        "errors": errors,
    })
payload = {
    "schema_version": "agent1_v5_dedup_benchmark_observations_v1",
    "status": "passed",
    "benchmark_plan_sha256": plan_sha,
    "sample_file": sample_path,
    "phases": phases,
}
fd, temporary = tempfile.mkstemp(prefix=".observations-", dir=os.path.dirname(output))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
PY

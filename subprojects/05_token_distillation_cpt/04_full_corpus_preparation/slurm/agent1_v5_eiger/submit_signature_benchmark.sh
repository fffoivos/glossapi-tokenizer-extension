#!/usr/bin/env bash
# Submit a nonce-bound held benchmark and release it only after identity proof.
set -eEuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT COORD_ROOT PIPELINE_ROOT" >&2
  exit 2
fi
run=$(realpath "$1")
coord=$(realpath "$2")
pipeline=$(realpath "$3")
artifact_prefix="${BENCHMARK_ARTIFACT_PREFIX:-dedup_acceleration_benchmark}"
benchmark_label="${BENCHMARK_LABEL:-primary}"
[[ "$artifact_prefix" =~ ^[A-Za-z0-9._-]+$ && "$benchmark_label" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid benchmark artifact prefix or label" >&2; exit 2; }
plan="${BENCHMARK_PLAN:-$run/dedup_acceleration_benchmark_plan.json}"
audit="$run/dedup_full_input_audit.json"
cutover="$run/dedup_acceleration_cutover.json"
runner="$pipeline/slurm/agent1_v5_eiger/normal_signature_benchmark.sh"
observations="${BENCHMARK_OBSERVATIONS:-$run/${artifact_prefix}_observations.json}"
submission="$run/${artifact_prefix}_submission.json"
evidence="$run/${artifact_prefix}_held_job_evidence.json"
release="$run/${artifact_prefix}_release_observation.json"
job_name="a1v5-signature-benchmark"
[[ "$benchmark_label" == primary ]] || job_name+="-$benchmark_label"
[[ -f "$plan" && -f "$audit" && -f "$cutover" && -x "$runner" ]] || { echo "benchmark inputs are incomplete" >&2; exit 1; }
[[ ! -e "$submission" && ! -e "$evidence" && ! -e "$release" ]] || { echo "benchmark submission artifacts already exist" >&2; exit 1; }
plan_sha=$(sha256sum "$plan" | awk '{print $1}')
audit_sha=$(sha256sum "$audit" | awk '{print $1}')
cutover_sha=$(sha256sum "$cutover" | awk '{print $1}')
runner_sha=$(sha256sum "$runner" | awk '{print $1}')
jq -e --arg plan "$plan_sha" --arg audit "$audit_sha" --arg cutover "$cutover_sha" '
  .schema_version == "agent1_v5_dedup_acceleration_benchmark_plan_v1" and .status == "passed"
  and .full_input_audit_sha256 == $audit and .cutover_receipt_sha256 == $cutover
  and (.phases | length == 4)' "$plan" >/dev/null
nonce=${BENCHMARK_NONCE_OVERRIDE:-$(tr -d '-' < /proc/sys/kernel/random/uuid)}
[[ "$nonce" =~ ^[A-Za-z0-9._-]{12,128}$ ]] || { echo "invalid benchmark nonce" >&2; exit 2; }

write_immutable() {
  local path="$1" payload="$2"
  python3 - "$path" "$payload" <<'PY'
import json, os, sys, tempfile
path, payload = sys.argv[1:]
if os.path.exists(path):
    raise SystemExit("refusing to overwrite " + path)
fd, temporary = tempfile.mkstemp(prefix=".benchmark-", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(json.loads(payload), handle, sort_keys=True); handle.write("\n")
os.link(temporary, path); os.unlink(temporary)
PY
}

verify_held() {
  local job="$1" raw
  raw=$(scontrol show job -o "$job")
  [[ "$raw" == *"UserId=fffoivos("* && "$raw" == *"Account=a0140 "* && "$raw" == *"Partition=normal "* ]] || return 1
  [[ "$raw" == *"JobName=$job_name "* && "$raw" == *"Comment=agent1-v5-dedup-benchmark:${nonce}"* ]] || return 1
  [[ "$raw" == *"JobState=PENDING "* && "$raw" == *"Reason=JobHeldUser "* ]] || return 1
  [[ "$raw" == *"StdOut=${coord}/slurm/"* && "$raw" == *"StdErr=${coord}/slurm/"* ]] || return 1
  printf '%s\n' "$raw"
}

job=""
armed=0
cleanup() {
  if (( armed == 1 )) && [[ -n "$job" ]] && verify_held "$job" >/dev/null 2>&1; then
    scancel "$job" || true
  fi
}
trap cleanup EXIT INT TERM ERR
job=$(sbatch --parsable --hold --uenv-passthrough=ignore --account=a0140 --partition=normal \
  --comment="agent1-v5-dedup-benchmark:${nonce}" --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --time=06:00:00 \
  --job-name="$job_name" --output="$coord/slurm/%x-%j.out" --error="$coord/slurm/%x-%j.err" \
  --export=ALL,RUN_ROOT="$run",PIPELINE_ROOT="$pipeline",BENCHMARK_PLAN="$plan",BENCHMARK_PLAN_SHA256="$plan_sha",FULL_INPUT_AUDIT="$audit",BENCHMARK_NONCE="$nonce",BENCHMARK_OBSERVATIONS="$observations",STOP_SENTINEL="$run/dedup_acceleration.stop" \
  "$runner")
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || { echo "unparseable benchmark job ID" >&2; exit 1; }
armed=1
raw=$(verify_held "$job") || { echo "held benchmark identity validation failed" >&2; exit 1; }
payload=$(jq -cn --arg job "$job" --arg nonce "$nonce" --arg plan "$plan_sha" --arg audit "$audit_sha" --arg cutover "$cutover_sha" --arg runner "$runner_sha" --arg raw "$raw" --arg run "$run" --arg coord "$coord" --arg pipeline "$pipeline" --arg job_name "$job_name" --arg observations "$observations" --arg benchmark_label "$benchmark_label" '{schema_version:"agent1_v5_dedup_acceleration_benchmark_submission_v1",status:"passed",array_job_id:$job,benchmark_nonce:$nonce,benchmark_label:$benchmark_label,run_root:$run,coord_root:$coord,pipeline_root:$pipeline,benchmark_plan_sha256:$plan,benchmark_observations_path:$observations,full_input_audit_sha256:$audit,cutover_receipt_sha256:$cutover,runner_sha256:$runner,owner:"fffoivos",account:"a0140",partition:"normal",job_name:$job_name,state:"PENDING",reason:"JobHeldUser",scontrol_raw:$raw}')
write_immutable "$evidence" "$payload"
write_immutable "$submission" "$payload"
scontrol release "$job"
armed=0
write_immutable "$release" "$(jq -cn --arg job "$job" --arg nonce "$nonce" '{schema_version:"agent1_v5_dedup_acceleration_benchmark_release_observation_v1",status:"passed",array_job_id:$job,benchmark_nonce:$nonce,release_requested:true}')"
printf 'submitted and released benchmark job %s\n' "$job"

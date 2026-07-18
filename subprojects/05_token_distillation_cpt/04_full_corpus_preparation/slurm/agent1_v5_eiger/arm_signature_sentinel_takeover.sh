#!/usr/bin/env bash
# Atomically arm one checksum-bound serial-chain handoff.  This does not cancel
# or modify a running Slurm job; the currently running legacy helper will read
# the active path only when it submits its successor.
set -eEuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 RUN_ROOT COORD_ROOT ACCELERATED_PIPELINE_ROOT LEGACY_PIPELINE_ROOT EXPECTED_RUNNING_RANK" >&2
  exit 2
fi
run=$(realpath "$1")
coord=$(realpath "$2")
accelerated=$(realpath "$3")
legacy_pipeline=$(realpath "$4")
expected_rank="$5"
(( expected_rank >= 0 && expected_rank < 430 )) || { echo "invalid expected running rank" >&2; exit 2; }
stop_after=$((expected_rank + 1))
active="/capstor/scratch/cscs/fffoivos/agent1-v5-code/signature-chain/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/scripts/run_signature_task_chain.sh"
guarded="$accelerated/scripts/run_signature_task_chain_guarded.sh"
tool_source="$accelerated/scripts/agent1_v5_signature_takeover.py"
tool_active="$(dirname -- "$active")/$(basename -- "$tool_source")"
request="$run/dedup_acceleration_takeover_request.json"
arm="$run/dedup_acceleration_takeover_arm.json"

run_takeover_tool() {
  uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
    "$coord/runtime/venv/bin/python" "$@"
}

[[ -f "$guarded" && -f "$tool_source" && -x "$guarded" ]] || { echo "accelerated handoff files are missing" >&2; exit 1; }
[[ ! -e "$request" && ! -e "$arm" && ! -e "$tool_active" ]] || { echo "takeover artifacts already exist; refusing re-arm" >&2; exit 1; }
# A predecessor can remain in COMPLETING very briefly after it has already
# submitted the next serial rank.  It is not an active chain job, so require
# exactly one *running* chain rank and separately reject queued successors.
mapfile -t jobs < <(squeue -h -u fffoivos -p debug -t RUNNING -o '%i|%j|%T|%a')
(( ${#jobs[@]} == 1 )) || { echo "expected exactly one running debug job before arming" >&2; exit 1; }
IFS='|' read -r job_id job_name job_state account <<<"${jobs[0]}"
[[ "$job_name" == "a1v5-signature-chain-r${expected_rank}" && "$job_state" == RUNNING && "$account" == a0140 ]] || {
  echo "serial boundary moved; expected running rank ${expected_rank}, found ${job_name}/${job_state}" >&2; exit 1;
}
mapfile -t queued_chain < <(squeue -h -u fffoivos -p debug -t PENDING -o '%j' | grep -E '^a1v5-signature-chain-r[0-9]+$' || true)
(( ${#queued_chain[@]} == 0 )) || { echo "queued serial successor exists: ${queued_chain[*]}" >&2; exit 1; }
raw="$(scontrol show job -o "$job_id")"
[[ "$raw" == *"UserId=fffoivos("* && "$raw" == *"Account=a0140 "* && "$raw" == *"Partition=debug "* && "$raw" == *"QOS=normal "* ]] || {
  echo "running serial job identity drift" >&2; exit 1;
}
original_sha="$(sha256sum "$active" | awk '{print $1}')"
expected_original_sha="af43fd90829ea36b9db58f7b43e66853bdfe1bc5b471369ccf5292dcaf6c25ca"
[[ "$original_sha" == "$expected_original_sha" ]] || { echo "legacy helper checksum drift" >&2; exit 1; }
backup="${active}.original-${original_sha}"
[[ ! -e "$backup" ]] || { echo "checksum-qualified legacy backup already exists" >&2; exit 1; }

run_takeover_tool "$tool_source" create-request --run-root "$run" --coord-root "$coord" \
  --legacy-pipeline-root "$legacy_pipeline" --active-helper "$active" --original-helper "$active" \
  --guarded-helper "$guarded" --takeover-tool "$tool_source" --stop-after-rank "$stop_after" --output "$request"

cp -p "$active" "$backup"
[[ "$(sha256sum "$backup" | awk '{print $1}')" == "$original_sha" ]] || { echo "legacy helper backup checksum mismatch" >&2; exit 1; }
tool_tmp="${tool_active}.partial-$$"
helper_tmp="${active}.partial-$$"
trap 'rm -f "$tool_tmp" "$helper_tmp"' EXIT
install -m 0755 "$tool_source" "$tool_tmp"
[[ "$(sha256sum "$tool_tmp" | awk '{print $1}')" == "$(jq -r .takeover_tool_sha256 "$request")" ]] || { echo "takeover tool checksum mismatch" >&2; exit 1; }
mv -f "$tool_tmp" "$tool_active"
install -m 0755 "$guarded" "$helper_tmp"
[[ "$(sha256sum "$helper_tmp" | awk '{print $1}')" == "$(jq -r .guarded_helper_sha256 "$request")" ]] || { echo "guarded helper checksum mismatch" >&2; exit 1; }
mv -f "$helper_tmp" "$active"
run_takeover_tool "$tool_active" write-arm --request "$request" --run-root "$run" --coord-root "$coord" \
  --legacy-pipeline-root "$legacy_pipeline" --active-helper "$active" --takeover-tool "$tool_active" \
  --stop-after-rank "$stop_after" --output "$arm"
printf 'armed sentinel handoff: rank %s -> rank %s stops\n' "$expected_rank" "$stop_after"

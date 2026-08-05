#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_STAGE_ROOT:?set frozen full-8B data root}"
: "${FULL8_BENCHMARK_ROOT:?set new benchmark root}"
: "${FULL8_INITIAL_MEGATRON:?set verified TP2 initialization root}"
DRY_RUN=${DRY_RUN:-1}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2
    printf 'DRY_JOB_%s\n' "$(printf '%s\0' "$@" | cksum | awk '{print $1}')"
  else
    sbatch --parsable "$@"
  fi
}

if [[ "$DRY_RUN" == 0 ]]; then
  [[ "${CONFIRM_GPU_BENCHMARK:-}" == "APERTUS8B_DP32_DP64" ]] || {
    echo "set CONFIRM_GPU_BENCHMARK=APERTUS8B_DP32_DP64" >&2; exit 2;
  }
  [[ ! -e "$FULL8_BENCHMARK_ROOT" ]] || { echo "benchmark root exists" >&2; exit 2; }
  mkdir -p "$FULL8_BENCHMARK_ROOT/logs" "$FULL8_BENCHMARK_ROOT/control_dp32" "$FULL8_BENCHMARK_ROOT/candidate_dp64"
  uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/freeze_benchmark_contract.py" \
    --schedule-manifest "$FULL8_STAGE_ROOT/schedules/schedule_manifest.json" \
    --goldfish-implementation "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/training/scheduled_packed_dataset.py" \
    --output "$FULL8_BENCHMARK_ROOT/benchmark_contract.json"
fi

sbatch_file="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch"
common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_BENCHMARK_MODE=1"
initial_common="$common,FULL8_BENCHMARK_SAVE_ITERATIONS=160"
control=$(submit --nodes=16 --time=02:00:00 --job-name=full8b-dp32-control \
  --output="$FULL8_BENCHMARK_ROOT/logs/%x-%j.out" --error="$FULL8_BENCHMARK_ROOT/logs/%x-%j.err" \
  --export="$initial_common,FULL8_RUN_ROOT=$FULL8_BENCHMARK_ROOT/control_dp32,FULL8_EXECUTION_PROFILE=dp32_16node,FULL8_START_ITERATION=0,FULL8_END_ITERATION=288,FULL8_LOAD_CHECKPOINT=$FULL8_INITIAL_MEGATRON" \
  "$sbatch_file")
candidate=$(submit --nodes=32 --time=02:00:00 --job-name=full8b-dp64-candidate \
  --output="$FULL8_BENCHMARK_ROOT/logs/%x-%j.out" --error="$FULL8_BENCHMARK_ROOT/logs/%x-%j.err" \
  --export="$initial_common,FULL8_RUN_ROOT=$FULL8_BENCHMARK_ROOT/candidate_dp64,FULL8_EXECUTION_PROFILE=dp64_32node,FULL8_START_ITERATION=0,FULL8_END_ITERATION=288,FULL8_LOAD_CHECKPOINT=$FULL8_INITIAL_MEGATRON" \
  "$sbatch_file")

control_restart=$(submit --nodes=16 --time=00:20:00 --job-name=full8b-dp32-restart --dependency="afterok:$control" \
  --output="$FULL8_BENCHMARK_ROOT/logs/%x-%j.out" --error="$FULL8_BENCHMARK_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_BENCHMARK_SAVE_ITERATIONS=,FULL8_EXACT_LOAD_ITERATION=160,FULL8_RUN_ROOT=$FULL8_BENCHMARK_ROOT/control_dp32,FULL8_EXECUTION_PROFILE=dp32_16node,FULL8_START_ITERATION=160,FULL8_END_ITERATION=161,FULL8_LOAD_CHECKPOINT=$FULL8_BENCHMARK_ROOT/control_dp32/checkpoints" \
  "$sbatch_file")
candidate_restart=$(submit --nodes=32 --time=00:20:00 --job-name=full8b-dp64-restart --dependency="afterok:$candidate" \
  --output="$FULL8_BENCHMARK_ROOT/logs/%x-%j.out" --error="$FULL8_BENCHMARK_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_BENCHMARK_SAVE_ITERATIONS=,FULL8_EXACT_LOAD_ITERATION=160,FULL8_RUN_ROOT=$FULL8_BENCHMARK_ROOT/candidate_dp64,FULL8_EXECUTION_PROFILE=dp64_32node,FULL8_START_ITERATION=160,FULL8_END_ITERATION=161,FULL8_LOAD_CHECKPOINT=$FULL8_BENCHMARK_ROOT/candidate_dp64/checkpoints" \
  "$sbatch_file")
gate=$(submit --dependency="afterok:$control_restart:$candidate_restart" \
  --output="$FULL8_BENCHMARK_ROOT/logs/%x-%j.out" --error="$FULL8_BENCHMARK_ROOT/logs/%x-%j.err" \
  --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_BENCHMARK_ROOT=$FULL8_BENCHMARK_ROOT" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_parallelism_benchmark.sbatch")

graph=$(python3 - "$control" "$candidate" "$control_restart" "$candidate_restart" "$gate" <<'PY'
import json,sys
print(json.dumps({"control":sys.argv[1],"candidate":sys.argv[2],"control_restart":sys.argv[3],"candidate_restart":sys.argv[4],"promotion_gate":sys.argv[5]},separators=(",",":")))
PY
)
if [[ "$DRY_RUN" == 0 ]]; then
  python3 - "$FULL8_BENCHMARK_ROOT/submission.json" "$graph" <<'PY'
import datetime,json,os,sys,tempfile
out,raw=sys.argv[1:]; payload={"schema_version":"apertus_full_8b_parallelism_benchmark_submission_v1","status":"submitted","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"jobs":json.loads(raw)}
fd,tmp=tempfile.mkstemp(prefix=".submission.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
fi
printf '%s\n' "$graph"

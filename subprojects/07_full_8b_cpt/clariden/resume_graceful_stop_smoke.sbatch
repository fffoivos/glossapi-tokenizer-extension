#!/usr/bin/env bash
#SBATCH --job-name=full8b-resume-smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_STAGE_ROOT:?set frozen data stage}"
: "${FULL8_INITIAL_MEGATRON:?set canonical initialization}"
: "${FULL8_SMOKE_RUN_ROOT:?set smoke root}"
: "${FULL8_SELECTED_PROFILE:?set selected execution profile receipt}"
: "${FULL8_EXECUTION_PROFILE:?set selected profile id}"
: "${FULL8_SMOKE_NODES:?set selected profile node count}"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
interrupted="$FULL8_SMOKE_RUN_ROOT/segments/updates_0_288"
receipt="$interrupted/training_job_receipt.json"
marker="$interrupted/graceful_stop_requested"
tracker="$FULL8_SMOKE_RUN_ROOT/checkpoints/latest_checkpointed_iteration.txt"
[[ -f "$receipt" && -f "$marker" && -f "$tracker" ]] || { echo "graceful-stop artifacts are incomplete" >&2; exit 2; }
latest=$(cat "$tracker")
python3 - "$receipt" "$latest" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); latest=int(sys.argv[2])
if x.get("status")!="gracefully_stopped" or not 0 < latest <= int(x.get("last_logged_iteration",-1)): raise SystemExit("graceful-stop receipt/checkpoint drift")
PY
end=$((latest + 1))
restart=$(sbatch --uenv-passthrough=ignore --parsable --nodes="$FULL8_SMOKE_NODES" --time=00:20:00 \
  --output="$FULL8_SMOKE_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_SMOKE_RUN_ROOT/logs/%x-%j.err" \
  --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_RUN_ROOT=$FULL8_SMOKE_RUN_ROOT,FULL8_START_ITERATION=$latest,FULL8_END_ITERATION=$end,FULL8_LOAD_CHECKPOINT=$FULL8_SMOKE_RUN_ROOT/checkpoints,FULL8_BENCHMARK_MODE=1,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_SELECTED_PROFILE=$FULL8_SELECTED_PROFILE,FULL8_EXECUTION_PROFILE=$FULL8_EXECUTION_PROFILE,FULL8_SMOKE_NODES=$FULL8_SMOKE_NODES" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch")
finalizer=$(sbatch --uenv-passthrough=ignore --parsable --dependency="afterok:$restart" \
  --output="$FULL8_SMOKE_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_SMOKE_RUN_ROOT/logs/%x-%j.err" \
  --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_SMOKE_RUN_ROOT=$FULL8_SMOKE_RUN_ROOT,FULL8_INTERRUPTED_ROOT=$interrupted,FULL8_RESTART_ROOT=$FULL8_SMOKE_RUN_ROOT/segments/updates_${latest}_${end}" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_graceful_stop_smoke.sbatch")
python3 - "$FULL8_SMOKE_RUN_ROOT/resume_submission.json" "$latest" "$restart" "$finalizer" <<'PY'
import datetime,json,sys
out,latest,restart,finalizer=sys.argv[1:]
open(out,"w").write(json.dumps({"schema_version":"apertus_full_8b_graceful_resume_submission_v1","status":"submitted","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"checkpoint_iteration":int(latest),"restart_job":restart,"finalizer_job":finalizer},indent=2,sort_keys=True)+"\n")
PY

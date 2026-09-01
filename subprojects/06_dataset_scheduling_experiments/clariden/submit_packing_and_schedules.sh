#!/usr/bin/env bash
set -euo pipefail
: "${SCHEDULE_PACKING_BUNDLE:?set immutable packing code bundle}"
: "${SCHEDULE_RUN_ID:?set immutable run id}"
: "${POOL_FINALIZER_JOB:?set pool finalizer job id}"
: "${CONFIRM_DATA_PACKING:?set to 1 for live submission}"
[[ "$CONFIRM_DATA_PACKING" == 1 ]] || { echo "live packing requires CONFIRM_DATA_PACKING=1" >&2; exit 2; }
SOURCE_REPO_ROOT=${SOURCE_REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/train-apertus-with-glossapi-cpt25b-86c1b8fe}
SCHEDULE_STAGE_ROOT=${SCHEDULE_STAGE_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/dataset_scheduling_0p5b/$SCHEDULE_RUN_ID}
SCHEDULE_RUN_ROOT=${SCHEDULE_RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments/data/$SCHEDULE_RUN_ID}
PACKING_PLAN_PATH=${PACKING_PLAN_PATH:-$SCHEDULE_STAGE_ROOT/packing_plan.json}
PACKED_CORPUS_RECEIPT_PATH=${PACKED_CORPUS_RECEIPT_PATH:-$SCHEDULE_STAGE_ROOT/packed_corpus_receipt.json}
SCHEDULE_OUTPUT_DIR=${SCHEDULE_OUTPUT_DIR:-$SCHEDULE_STAGE_ROOT/schedules}
base_export="ALL,SCHEDULE_PACKING_BUNDLE=$SCHEDULE_PACKING_BUNDLE,SOURCE_REPO_ROOT=$SOURCE_REPO_ROOT,SCHEDULE_STAGE_ROOT=$SCHEDULE_STAGE_ROOT,PACKING_PLAN_PATH=$PACKING_PLAN_PATH,PACKED_CORPUS_RECEIPT_PATH=$PACKED_CORPUS_RECEIPT_PATH,SCHEDULE_OUTPUT_DIR=$SCHEDULE_OUTPUT_DIR"
plan_job=$(sbatch --parsable --dependency="afterok:$POOL_FINALIZER_JOB" --export="$base_export" \
  --output="$SCHEDULE_RUN_ROOT/%x-%j.out" --error="$SCHEDULE_RUN_ROOT/%x-%j.err" \
  "$SCHEDULE_PACKING_BUNDLE/clariden/build_packing_plan.sbatch")
controller_export="$base_export,SCHEDULE_RUN_ROOT=$SCHEDULE_RUN_ROOT,MAX_PARALLEL_PACKING=${MAX_PARALLEL_PACKING:-24}"
controller_job=$(sbatch --parsable --dependency="afterok:$plan_job" --export="$controller_export" \
  --output="$SCHEDULE_RUN_ROOT/%x-%j.out" --error="$SCHEDULE_RUN_ROOT/%x-%j.err" \
  "$SCHEDULE_PACKING_BUNDLE/clariden/submit_dynamic_packing.sbatch")
python3 - "$SCHEDULE_STAGE_ROOT/submissions/packing_submission_controller.json" "$plan_job" "$controller_job" <<'PY'
import json,sys
out,plan,controller=sys.argv[1:]
payload={"schema_version":"apertus_mini_packing_controller_submission_v1","jobs":{"packing_plan":plan,"dynamic_controller":controller}}
open(out,"w",encoding="utf-8").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
printf 'packing_plan=%s\ndynamic_controller=%s\n' "$plan_job" "$controller_job"

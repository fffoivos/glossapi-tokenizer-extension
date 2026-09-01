#!/usr/bin/env bash
#SBATCH --job-name=mini_mix_packsubmit
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail
: "${SCHEDULE_PACKING_BUNDLE:?set immutable packing code bundle}"
: "${SOURCE_REPO_ROOT:?set frozen corpus builder checkout}"
: "${SCHEDULE_STAGE_ROOT:?set Mini corpus stage}"
: "${SCHEDULE_RUN_ROOT:?set run log root}"
MAX_PARALLEL_PACKING=${MAX_PARALLEL_PACKING:-24}
PACKING_PLAN_PATH=${PACKING_PLAN_PATH:-$SCHEDULE_STAGE_ROOT/packing_plan.json}
PACKED_CORPUS_RECEIPT_PATH=${PACKED_CORPUS_RECEIPT_PATH:-$SCHEDULE_STAGE_ROOT/packed_corpus_receipt.json}
SCHEDULE_OUTPUT_DIR=${SCHEDULE_OUTPUT_DIR:-$SCHEDULE_STAGE_ROOT/schedules}

task_count=$(python3 - "$PACKING_PLAN_PATH" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1],encoding="utf-8"))["tasks"]))
PY
)
if ! [[ "$task_count" =~ ^[1-9][0-9]*$ ]] || (( task_count > 512 )); then
  echo "invalid dynamic packing task count: $task_count" >&2
  exit 2
fi
last_task=$((task_count - 1))
base_export="ALL,SCHEDULE_PACKING_BUNDLE=$SCHEDULE_PACKING_BUNDLE,SOURCE_REPO_ROOT=$SOURCE_REPO_ROOT,SCHEDULE_STAGE_ROOT=$SCHEDULE_STAGE_ROOT,PACKING_PLAN_PATH=$PACKING_PLAN_PATH,PACKED_CORPUS_RECEIPT_PATH=$PACKED_CORPUS_RECEIPT_PATH,SCHEDULE_OUTPUT_DIR=$SCHEDULE_OUTPUT_DIR"
pack_job=$(sbatch --parsable --array="0-${last_task}%${MAX_PARALLEL_PACKING}" --export="$base_export" \
  --output="$SCHEDULE_RUN_ROOT/%x-%A_%a.out" --error="$SCHEDULE_RUN_ROOT/%x-%A_%a.err" \
  "$SCHEDULE_PACKING_BUNDLE/clariden/pack_catalog_bucket.sbatch")
final_job=$(sbatch --parsable --dependency="afterok:$pack_job" --export="$base_export" \
  --output="$SCHEDULE_RUN_ROOT/%x-%j.out" --error="$SCHEDULE_RUN_ROOT/%x-%j.err" \
  "$SCHEDULE_PACKING_BUNDLE/clariden/finalize_packed_corpus.sbatch")
schedule_job=$(sbatch --parsable --dependency="afterok:$final_job" --export="$base_export" \
  --output="$SCHEDULE_RUN_ROOT/%x-%j.out" --error="$SCHEDULE_RUN_ROOT/%x-%j.err" \
  "$SCHEDULE_PACKING_BUNDLE/clariden/build_five_schedules.sbatch")
python3 - "$SCHEDULE_STAGE_ROOT/submissions/packing_and_schedules_dynamic.json" \
  "$task_count" "$pack_job" "$final_job" "$schedule_job" <<'PY'
import json,sys
out,count,pack,final,schedules=sys.argv[1:]
payload={"schema_version":"apertus_mini_dynamic_packing_schedule_submission_v1","packing_task_count":int(count),"jobs":{"packing_array":pack,"packed_finalizer":final,"five_schedules":schedules}}
open(out,"w",encoding="utf-8").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
printf 'packing_tasks=%s\npacking_array=%s\npacked_finalizer=%s\nfive_schedules=%s\n' \
  "$task_count" "$pack_job" "$final_job" "$schedule_job"

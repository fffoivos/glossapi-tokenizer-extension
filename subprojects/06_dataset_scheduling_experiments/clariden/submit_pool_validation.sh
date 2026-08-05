#!/usr/bin/env bash
set -euo pipefail

: "${SCHEDULE_VALIDATION_BUNDLE:?set immutable validation code bundle}"
: "${SCHEDULE_RUN_ID:?set immutable run id}"
: "${TRAIN_ARRAY_A:?set first completed-or-pending training array job id}"
: "${TRAIN_ARRAY_B:?set second completed-or-pending training array job id}"
: "${HELDOUT_ARRAY:?set completed-or-pending heldout array job id}"
: "${CONFIRM_DATA_VALIDATION:?set to 1 for live submission}"
[[ "$CONFIRM_DATA_VALIDATION" == 1 ]] || { echo "live validation requires CONFIRM_DATA_VALIDATION=1" >&2; exit 2; }

SOURCE_REPO_ROOT=${SOURCE_REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/train-apertus-with-glossapi-cpt25b-86c1b8fe}
SCHEDULE_STAGE_ROOT=${SCHEDULE_STAGE_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/dataset_scheduling_0p5b/$SCHEDULE_RUN_ID}
SCHEDULE_RUN_ROOT=${SCHEDULE_RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments/data/$SCHEDULE_RUN_ID}
MAX_PARALLEL_VALIDATION=${MAX_PARALLEL_VALIDATION:-12}
dependency="afterok:$TRAIN_ARRAY_A:$TRAIN_ARRAY_B:$HELDOUT_ARRAY"
base_export="ALL,SCHEDULE_VALIDATION_BUNDLE=$SCHEDULE_VALIDATION_BUNDLE,SOURCE_REPO_ROOT=$SOURCE_REPO_ROOT,SCHEDULE_STAGE_ROOT=$SCHEDULE_STAGE_ROOT"

validation_job=$(sbatch --parsable --dependency="$dependency" \
  --array="0-512%$MAX_PARALLEL_VALIDATION" --export="$base_export" \
  --output="$SCHEDULE_RUN_ROOT/%x-%A_%a.out" --error="$SCHEDULE_RUN_ROOT/%x-%A_%a.err" \
  "$SCHEDULE_VALIDATION_BUNDLE/clariden/validate_partition_group.sbatch")
final_job=$(sbatch --parsable --dependency="afterok:$validation_job" --export="$base_export" \
  --output="$SCHEDULE_RUN_ROOT/%x-%j.out" --error="$SCHEDULE_RUN_ROOT/%x-%j.err" \
  "$SCHEDULE_VALIDATION_BUNDLE/clariden/finalize_pool_corpus.sbatch")

python3 - "$SCHEDULE_STAGE_ROOT/submissions/pool_validation.json" "$validation_job" "$final_job" <<'PY'
import json,sys
out,validation,final=sys.argv[1:]
payload={
 "schema_version":"apertus_mini_pool_validation_submission_v1",
 "expected_partition_groups":513,
 "jobs":{"partition_validation_array":validation,"pool_finalizer":final},
}
open(out,"w",encoding="utf-8").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
printf 'partition_validation=%s\npool_finalizer=%s\n' "$validation_job" "$final_job"

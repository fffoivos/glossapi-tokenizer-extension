#!/usr/bin/env bash
set -euo pipefail

: "${SCHEDULE_CODE_BUNDLE:?set remote immutable schedule code bundle}"
: "${SCHEDULE_RUN_ID:?set immutable run id}"
: "${CONFIRM_DATA_PREPARATION:?set to 1 for live submission}"
[[ "$CONFIRM_DATA_PREPARATION" == 1 ]] || { echo "live preparation requires CONFIRM_DATA_PREPARATION=1" >&2; exit 2; }

SOURCE_REPO_ROOT=${SOURCE_REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/train-apertus-with-glossapi-cpt25b-86c1b8fe}
SOURCE_STAGE_ROOT=${SOURCE_STAGE_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/cpt25b_midtraining/20260731T124000Z-cpt25b-v1}
SCHEDULE_STAGE_ROOT=${SCHEDULE_STAGE_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/dataset_scheduling_0p5b/$SCHEDULE_RUN_ID}
SCHEDULE_RUN_ROOT=${SCHEDULE_RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments/data/$SCHEDULE_RUN_ID}
MINI_SOURCE_DIR=${MINI_SOURCE_DIR:-/iopsstor/scratch/cscs/fffoivos/tokenizers/Apertus-v1.1-0.5B-1b727617}
MINI_OVERLAY_DIR=${MINI_OVERLAY_DIR:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_v1_1_0p5b_greek_overlay_fcd33ec}
MAX_PARALLEL_TRAIN=${MAX_PARALLEL_TRAIN:-12}
MAX_PARALLEL_HELDOUT=${MAX_PARALLEL_HELDOUT:-6}

mkdir -p "$SCHEDULE_STAGE_ROOT/submissions" "$SCHEDULE_RUN_ROOT"
base_export="ALL,SCHEDULE_CODE_BUNDLE=$SCHEDULE_CODE_BUNDLE,SOURCE_REPO_ROOT=$SOURCE_REPO_ROOT,SOURCE_STAGE_ROOT=$SOURCE_STAGE_ROOT,SCHEDULE_STAGE_ROOT=$SCHEDULE_STAGE_ROOT,MINI_SOURCE_DIR=$MINI_SOURCE_DIR,MINI_OVERLAY_DIR=$MINI_OVERLAY_DIR"
prepare_job=$(sbatch --parsable --export="$base_export" \
  --output="$SCHEDULE_RUN_ROOT/%x-%j.out" --error="$SCHEDULE_RUN_ROOT/%x-%j.err" \
  "$SCHEDULE_CODE_BUNDLE/clariden/prepare_data_mixes.sbatch")
array_a=$(sbatch --parsable --dependency="afterok:$prepare_job" \
  --array="0-1000%$MAX_PARALLEL_TRAIN" --export="$base_export,TASK_OFFSET=0" \
  --output="$SCHEDULE_RUN_ROOT/%x-%A_%a.out" --error="$SCHEDULE_RUN_ROOT/%x-%A_%a.err" \
  "$SCHEDULE_CODE_BUNDLE/clariden/build_train_shards.sbatch")
array_b=$(sbatch --parsable --dependency="afterok:$prepare_job" \
  --array="0-455%$MAX_PARALLEL_TRAIN" --export="$base_export,TASK_OFFSET=1001" \
  --output="$SCHEDULE_RUN_ROOT/%x-%A_%a.out" --error="$SCHEDULE_RUN_ROOT/%x-%A_%a.err" \
  "$SCHEDULE_CODE_BUNDLE/clariden/build_train_shards.sbatch")
heldout_job=$(sbatch --parsable --dependency="afterok:$prepare_job" \
  --array="0-11%$MAX_PARALLEL_HELDOUT" --export="$base_export" \
  --output="$SCHEDULE_RUN_ROOT/%x-%A_%a.out" --error="$SCHEDULE_RUN_ROOT/%x-%A_%a.err" \
  "$SCHEDULE_CODE_BUNDLE/clariden/build_heldout_shards.sbatch")

python3 - "$SCHEDULE_STAGE_ROOT/submissions/initial_build.json" "$SCHEDULE_RUN_ID" "$prepare_job" "$array_a" "$array_b" "$heldout_job" <<'PY'
import json,sys
out,run_id,prepare,array_a,array_b,heldout=sys.argv[1:]
payload={
 "schema_version":"apertus_mini_schedule_initial_build_submission_v1",
 "run_id":run_id,
 "expected_training_tasks":1457,
 "expected_heldout_tasks":12,
 "jobs":{"prepare":prepare,"training_arrays":[array_a,array_b],"heldout_array":heldout},
}
open(out,"w",encoding="utf-8").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
printf 'run_id=%s\nstage=%s\nprepare=%s\ntraining_arrays=%s,%s\nheldout=%s\n' \
  "$SCHEDULE_RUN_ID" "$SCHEDULE_STAGE_ROOT" "$prepare_job" "$array_a" "$array_b" "$heldout_job"

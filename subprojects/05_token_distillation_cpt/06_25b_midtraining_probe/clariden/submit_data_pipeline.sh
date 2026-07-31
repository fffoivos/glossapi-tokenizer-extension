#!/usr/bin/env bash
# Submit receipt-bound preparation jobs. This never submits the 64-GPU run.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"
MODE=${1:-status}
DRY_RUN=${DRY_RUN:-1}
CONFIRM_PREPARATION=${CONFIRM_PREPARATION:-0}
MAX_ARRAY_SIZE=1001
MAX_PARALLEL_TRAIN=${MAX_PARALLEL_TRAIN:-12}
MAX_PARALLEL_HELDOUT=${MAX_PARALLEL_HELDOUT:-6}
case "$DRY_RUN" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
if [[ "$DRY_RUN" == 0 && "$CONFIRM_PREPARATION" != 1 ]]; then
  echo "ERROR: live preparation requires CONFIRM_PREPARATION=1" >&2
  exit 2
fi
if [[ "$DRY_RUN" == 0 ]]; then
  mkdir -p "$RUN_ROOT" "$STAGE_ROOT/submissions"
fi

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY:' >&2; printf ' %q' sbatch --parsable "$@" >&2; printf '\n' >&2
    echo "DRY${RANDOM}"
  else
    sbatch --parsable "$@"
  fi
}

base_export="ALL,CPT_CLARIDEN_DIR=$HERE,REPO_ROOT=$REPO_ROOT,CPT_RUN_ID=$CPT_RUN_ID"
hf_export="$base_export,HF_TOKEN"
common=(--account="$CPT_ACCOUNT" --export="$base_export")

case "$MODE" in
  prereqs)
    if [[ "$DRY_RUN" == 0 ]]; then : "${HF_TOKEN:?inject HF_TOKEN for gated inputs}"; fi
    runtime_job=$(submit "${common[@]}" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/bootstrap_runtime.sbatch")
    dataset_job=$(submit --account="$CPT_ACCOUNT" --export="$hf_export" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/materialize_hf_v2.sbatch")
    base_init_job=$(submit --account="$CPT_ACCOUNT" --export="$hf_export" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/materialize_base_init.sbatch")
    replay_job=$(submit --account="$CPT_ACCOUNT" --dependency="afterok:$runtime_job" --export="$hf_export,RESTAGE_REPLACE=1" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/acquire_replay.sbatch")
    query_job=$(submit "${common[@]}" --dependency="afterok:$runtime_job" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/freeze_greekmmlu.sbatch")
    old_greek_job=$(submit "${common[@]}" --dependency="afterok:$replay_job" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/build_old_greek.sbatch")
    freeze_job=$(submit "${common[@]}" --dependency="afterok:$dataset_job:$old_greek_job:$query_job" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/freeze_inputs.sbatch")
    heldout_job=$(submit "${common[@]}" --dependency="afterok:$freeze_job" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/build_heldouts.sbatch")
    init_job=$(submit "${common[@]}" --dependency="afterok:$base_init_job" \
      --export="$base_export,BASE_TD_INIT_HF=$BASE_INIT_HF,TOKENIZER_DIR=$TOKENIZER_DIR,TD_COVERAGE_JSONL=$TOKENIZER_PROBE_ROOT/coverage_0512/td_coverage_prepass.jsonl,TD_SNIPPETS_JSONL=$TOKENIZER_PROBE_ROOT/coverage_0512/td_snippet_index/snippets.jsonl,TD_TOKEN_IDS_FILE=$TOKENIZER_PROBE_ROOT/train_token_ids_0512.txt,CALIBRATION_JSONL=$TOKENIZER_PROBE_ROOT/probe_data/output_calibration.jsonl,OUTPUT_ROOT=$INIT_ROOT" \
      --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$CPT_DIR/initialization/build_production_init.sbatch")
    roundtrip_job=$(submit "${common[@]}" --dependency="afterok:$init_job" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/roundtrip_production_init.sbatch")
    if [[ "$DRY_RUN" == 0 ]]; then python3 - "$STAGE_ROOT/submissions/prereqs.json" "$runtime_job" "$dataset_job" "$base_init_job" "$replay_job" "$query_job" "$old_greek_job" "$freeze_job" "$heldout_job" "$init_job" "$roundtrip_job" <<'PY'
import json,sys
names=("runtime","dataset","base_init","replay","greekmmlu","old_greek","freeze_inputs","heldouts","production_init","roundtrip")
open(sys.argv[1],"w",encoding="utf-8").write(json.dumps({"schema_version":"greek_cpt25b_prereq_submission_v1","jobs":dict(zip(names,sys.argv[2:]))},indent=2,sort_keys=True)+"\n")
PY
    fi
    printf 'runtime=%s\ndataset=%s\nbase_init=%s\nreplay=%s\ngreekmmlu=%s\nold_greek=%s\nfreeze_inputs=%s\nheldouts=%s\nproduction_init=%s\nroundtrip=%s\n' \
      "$runtime_job" "$dataset_job" "$base_init_job" "$replay_job" "$query_job" "$old_greek_job" "$freeze_job" "$heldout_job" "$init_job" "$roundtrip_job"
    ;;
  after-freeze)
    test -s "$INPUT_RECEIPT" || { echo "ERROR: input receipt not ready: $INPUT_RECEIPT" >&2; exit 3; }
    prereqs="$STAGE_ROOT/submissions/prereqs.json"
    test -s "$prereqs"
    read -r task_count heldout_count heldout_job < <(cpt_python - "$INPUT_RECEIPT" "$RECIPE" "$prereqs" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8")); c=json.load(open(sys.argv[2],encoding="utf-8")); p=json.load(open(sys.argv[3],encoding="utf-8"))
print(len(r["tasks"]),sum(len(c["heldouts"][k]) for k in ("new_greek","foreign_replay","old_greek_replay")),p["jobs"]["heldouts"])
PY
)
    (( task_count > 0 && heldout_count > 0 ))
    train_jobs=()
    offset=0
    while (( offset < task_count )); do
      remaining=$((task_count-offset)); chunk=$((remaining<MAX_ARRAY_SIZE ? remaining : MAX_ARRAY_SIZE))
      job=$(submit --account="$CPT_ACCOUNT" --dependency="afterok:$heldout_job" \
        --array="0-$((chunk-1))%$MAX_PARALLEL_TRAIN" --export="$base_export,TASK_OFFSET=$offset" \
        --output="$RUN_ROOT/%x-%A_%a.out" --error="$RUN_ROOT/%x-%A_%a.err" "$HERE/build_train_shards.sbatch")
      train_jobs+=("$job"); offset=$((offset+chunk))
    done
    val_job=$(submit --account="$CPT_ACCOUNT" --dependency="afterok:$heldout_job" \
      --array="0-$((heldout_count-1))%$MAX_PARALLEL_HELDOUT" --export="$base_export" \
      --output="$RUN_ROOT/%x-%A_%a.out" --error="$RUN_ROOT/%x-%A_%a.err" "$HERE/build_heldout_shards.sbatch")
    dependency=$(IFS=:; echo "${train_jobs[*]}:$val_job")
    final_job=$(submit "${common[@]}" --dependency="afterok:$dependency" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/finalize_bridge.sbatch")
    if [[ "$DRY_RUN" == 0 ]]; then python3 - "$STAGE_ROOT/submissions/binaries.json" "$task_count" "$heldout_count" "$val_job" "$final_job" "${train_jobs[@]}" <<'PY'
import json,sys
out,tasks,heldouts,val,final,*train=sys.argv[1:]
open(out,"w",encoding="utf-8").write(json.dumps({"schema_version":"greek_cpt25b_binary_submission_v1","training_tasks":int(tasks),"heldout_tasks":int(heldouts),"jobs":{"train_arrays":train,"heldout_array":val,"finalize":final}},indent=2,sort_keys=True)+"\n")
PY
    fi
    printf 'training_tasks=%s\ntrain_arrays=%s\nheldout_array=%s\nfinalize=%s\n' "$task_count" "${train_jobs[*]}" "$val_job" "$final_job"
    ;;
  assets)
    test -s "$BRIDGE_MANIFEST"
    test -s "$ROUNDTRIP_ROOT/work/verification.json"
    job=$(submit "${common[@]}" --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" "$HERE/freeze_training_assets.sbatch")
    echo "assets=$job"
    ;;
  status)
    for path in "$RUNTIME_RECEIPT" "$DATASET_MANIFEST" "$BASE_INIT_HF/materialization_receipt.json" "$REPLAY_RECEIPT" "$OLD_GREEK_RECEIPT" "$DECONTAM_BINDING" "$INPUT_RECEIPT" "$HELDOUT_MANIFEST" "$INIT_ROOT/production_init_verification.json" "$ROUNDTRIP_ROOT/work/verification.json" "$BRIDGE_MANIFEST" "$TRAINING_ASSETS_RECEIPT"; do
      if [[ -s "$path" ]]; then printf 'READY %s\n' "$path"; else printf 'MISSING %s\n' "$path"; fi
    done
    squeue -u fffoivos -o '%.18i %.32j %.10T %.10M %.20R' | grep -E 'JOBID|cpt25b' || true
    ;;
  *) echo "usage: $0 {prereqs|after-freeze|assets|status}" >&2; exit 2 ;;
esac

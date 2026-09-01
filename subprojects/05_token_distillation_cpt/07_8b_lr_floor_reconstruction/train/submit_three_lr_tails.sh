#!/usr/bin/env bash
# Submit one shared prefix and the T10/T20/T30 tails. Clariden only.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LR13_CODE_ROOT=$(cd "$HERE/.." && pwd)
export LR13_CODE_ROOT
source "$LR13_CODE_ROOT/clariden/paths.env"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_GPU_LAUNCH="${CONFIRM_GPU_LAUNCH:-}"
case "$DRY_RUN" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
cluster=$(scontrol show config 2>/dev/null | sed -n 's/^ClusterName *= *//p' | head -1)
if [[ "$cluster" != clariden ]]; then
  echo "ERROR: GPU launch is Clariden-only; current Slurm cluster is '${cluster:-unknown}'" >&2
  exit 3
fi
if [[ "$DRY_RUN" == 0 && "$CONFIRM_GPU_LAUNCH" != APERTUS8B_LR_FLOOR_3WAY ]]; then
  echo "ERROR: live launch requires CONFIRM_GPU_LAUNCH=APERTUS8B_LR_FLOOR_3WAY" >&2
  exit 4
fi
test -s "$LR13_DATASET_MANIFEST"
test -s "$LR13_TRAINING_DATA_ENV"
test -s "$LR13_ASSETS_RECEIPT"

read -r LR13_SOURCE_REPO MEGATRON_LM_SWISSAI_DIR LR13_INIT_CHECKPOINT < <(
  python3 - "$LR13_ASSETS_RECEIPT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
print(r["source_repository"]["root"],r["megatron"]["root"],r["initialization"]["checkpoint"]["root"])
PY
)
export LR13_SOURCE_REPO MEGATRON_LM_SWISSAI_DIR

mkdir_live() { [[ "$DRY_RUN" == 1 ]] || mkdir -p "$1"; }
submit() {
  local dry_id="$1"
  shift
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY:' >&2; printf ' %q' sbatch --parsable "$@" >&2; printf '\n' >&2
    echo "$dry_id"
  else
    sbatch --parsable "$@"
  fi
}
shared="$LR13_RUN_ROOT/shared_prefix"
receipts="$LR13_RUN_ROOT/checkpoint_receipts"
submissions="$LR13_RUN_ROOT/submissions"
mkdir_live "$shared"; mkdir_live "$receipts"; mkdir_live "$submissions"
common="ALL,LR13_CODE_ROOT=$LR13_CODE_ROOT,LR13_ASSETS_RECEIPT=$LR13_ASSETS_RECEIPT,LR13_TRAINING_DATA_ENV=$LR13_TRAINING_DATA_ENV,LR13_SOURCE_REPO=$LR13_SOURCE_REPO,MEGATRON_LM_SWISSAI_DIR=$MEGATRON_LM_SWISSAI_DIR"
freeze_bundle="$submissions/freeze_bundle"
if [[ "$DRY_RUN" == 0 ]]; then
  install -d -m 0755 "$freeze_bundle/clariden" "$freeze_bundle/train"
  install -m 0555 "$LR13_CODE_ROOT/clariden/freeze_checkpoint.sbatch" \
    "$freeze_bundle/clariden/freeze_checkpoint.sbatch"
  install -m 0555 "$LR13_CODE_ROOT/train/freeze_resume_checkpoint.py" \
    "$freeze_bundle/train/freeze_resume_checkpoint.py"
fi
freeze_common="ALL,LR13_CODE_ROOT=$freeze_bundle,LR13_ASSETS_RECEIPT=$LR13_ASSETS_RECEIPT"

p1=$(submit DRY_PHASE1 --account=a0140 --partition=normal --nodes=16 --time=08:00:00 \
  --job-name="${LR13_RUN_ID}_p1" --output="$shared/%x-%j.out" --error="$shared/%x-%j.err" \
  --export="$common,LR13_START_ITERATION=0,LR13_END_ITERATION=2253,LR13_PHASE=1,LR13_LR_FLOOR_PERCENT=10,LR13_SAVE_INTERVAL=119,LR13_LOAD_CHECKPOINT=$LR13_INIT_CHECKPOINT,LR13_RESUME_RECEIPT=,LR13_OUTPUT_DIR=$shared" \
  "$LR13_CODE_ROOT/clariden/train_segment.sbatch")
p1_receipt="$receipts/iteration_2253.json"
f1=$(submit DRY_FREEZE2253 --account=a0140 --partition=xfer --dependency="afterok:$p1" \
  --job-name="${LR13_RUN_ID}_freeze2253" --output="$shared/%x-%j.out" --error="$shared/%x-%j.err" \
  --export="$freeze_common,LR13_CHECKPOINT_DIR=$shared/checkpoints,LR13_CHECKPOINT_ITERATION=2253,LR13_CHECKPOINT_RECEIPT=$p1_receipt" \
  "$freeze_bundle/clariden/freeze_checkpoint.sbatch")
p2=$(submit DRY_SHARED_STABLE --account=a0140 --partition=normal --nodes=16 --time=03:00:00 --dependency="afterok:$f1" \
  --job-name="${LR13_RUN_ID}_stable" --output="$shared/%x-%j.out" --error="$shared/%x-%j.err" \
  --export="$common,LR13_START_ITERATION=2253,LR13_END_ITERATION=2574,LR13_PHASE=2,LR13_LR_FLOOR_PERCENT=10,LR13_SAVE_INTERVAL=119,LR13_LOAD_CHECKPOINT=$shared/checkpoints,LR13_RESUME_RECEIPT=$p1_receipt,LR13_OUTPUT_DIR=$shared" \
  "$LR13_CODE_ROOT/clariden/train_segment.sbatch")
branch_receipt="$receipts/iteration_2574.json"
f2=$(submit DRY_FREEZE2574 --account=a0140 --partition=xfer --dependency="afterok:$p2" \
  --job-name="${LR13_RUN_ID}_freeze2574" --output="$shared/%x-%j.out" --error="$shared/%x-%j.err" \
  --export="$freeze_common,LR13_CHECKPOINT_DIR=$shared/checkpoints,LR13_CHECKPOINT_ITERATION=2574,LR13_CHECKPOINT_RECEIPT=$branch_receipt" \
  "$freeze_bundle/clariden/freeze_checkpoint.sbatch")

tail_jobs=(); final_jobs=()
for floor in 10 20 30; do
  branch="$LR13_RUN_ROOT/T${floor}"
  mkdir_live "$branch"
  job=$(submit "DRY_T${floor}" --account=a0140 --partition=normal --nodes=16 --time=03:00:00 --dependency="afterok:$f2" \
    --job-name="${LR13_RUN_ID}_T${floor}" --output="$branch/%x-%j.out" --error="$branch/%x-%j.err" \
    --export="$common,LR13_START_ITERATION=2574,LR13_END_ITERATION=3218,LR13_PHASE=2,LR13_LR_FLOOR_PERCENT=$floor,LR13_SAVE_INTERVAL=107,LR13_LOAD_CHECKPOINT=$shared/checkpoints,LR13_RESUME_RECEIPT=$branch_receipt,LR13_OUTPUT_DIR=$branch" \
    "$LR13_CODE_ROOT/clariden/train_segment.sbatch")
  final_receipt="$receipts/T${floor}_iteration_3218.json"
  frozen=$(submit "DRY_T${floor}_FREEZE" --account=a0140 --partition=xfer --dependency="afterok:$job" \
    --job-name="${LR13_RUN_ID}_T${floor}_freeze" --output="$branch/%x-%j.out" --error="$branch/%x-%j.err" \
    --export="$freeze_common,LR13_CHECKPOINT_DIR=$branch/checkpoints,LR13_CHECKPOINT_ITERATION=3218,LR13_CHECKPOINT_RECEIPT=$final_receipt" \
    "$freeze_bundle/clariden/freeze_checkpoint.sbatch")
  tail_jobs+=("$job"); final_jobs+=("$frozen")
done

printf 'phase1=%s\nfreeze2253=%s\nshared_stable=%s\nfreeze2574=%s\nT10=%s\nT20=%s\nT30=%s\n' \
  "$p1" "$f1" "$p2" "$f2" "${tail_jobs[0]}" "${tail_jobs[1]}" "${tail_jobs[2]}"
if [[ "$DRY_RUN" == 0 ]]; then
  python3 - "$submissions/launch_graph.json" "$p1" "$f1" "$p2" "$f2" "${tail_jobs[@]}" "${final_jobs[@]}" <<'PY'
import datetime,json,os,sys,tempfile
out,p1,f1,p2,f2,t10,t20,t30,z10,z20,z30=sys.argv[1:]
value={"schema_version":"apertus8b_lr_floor_submission_v1","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"jobs":{"phase1":p1,"freeze_2253":f1,"shared_stable":p2,"freeze_2574":f2,"tails":{"T10":t10,"T20":t20,"T30":t30},"freeze_final":{"T10":z10,"T20":z20,"T30":z30}}}
fd,tmp=tempfile.mkstemp(prefix=".launch.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w",encoding="utf-8") as h: json.dump(value,h,indent=2,sort_keys=True); h.write("\n"); h.flush(); os.fsync(h.fileno())
os.replace(tmp,out)
PY
fi

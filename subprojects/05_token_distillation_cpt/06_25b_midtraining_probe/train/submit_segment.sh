#!/usr/bin/env bash
# Submit exactly one receipt-gated segment. Default is dry-run only.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE_ROOT=$(cd "$HERE/.." && pwd)
: "${TRAINING_ASSETS_RECEIPT:?set TRAINING_ASSETS_RECEIPT}"
: "${SMOKE_VERIFICATION:?set SMOKE_VERIFICATION from the passed two-phase GPU smoke}"
: "${START_ITERATION:?set START_ITERATION to 0, 1785, or 3570}"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_GPU_LAUNCH="${CONFIRM_GPU_LAUNCH:-}"
RESUME_CHECKPOINT_RECEIPT="${RESUME_CHECKPOINT_RECEIPT:-}"
RUN_TAG="${RUN_TAG:-greek_cpt25b_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/25b_midtraining}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_TAG}"

case "$START_ITERATION" in
  0) CPT_PHASE=1; END_ITERATION=1785 ;;
  1785) CPT_PHASE=1; END_ITERATION=3570 ;;
  3570) CPT_PHASE=2; END_ITERATION=5960 ;;
  *) echo "ERROR: START_ITERATION must be 0, 1785, or 3570" >&2; exit 2 ;;
esac
case "$DRY_RUN" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
if [[ "$DRY_RUN" == 0 && "$CONFIRM_GPU_LAUNCH" != "GREEK_CPT25B_64GPU" ]]; then
  echo "ERROR: live launch requires CONFIRM_GPU_LAUNCH=GREEK_CPT25B_64GPU" >&2
  exit 2
fi

read -r REPO_ROOT INIT_CKPT MEGATRON_DIR TOKENIZER_DIR BRIDGE_DATA_ENV TRAIN_SCRIPT < <(
  python3 - "$TRAINING_ASSETS_RECEIPT" "$START_ITERATION" "$RESUME_CHECKPOINT_RECEIPT" "$SMOKE_VERIFICATION" <<'PY'
import json,sys
from pathlib import Path
import hashlib
assets_path=Path(sys.argv[1]).resolve(); assets=json.load(open(assets_path,encoding="utf-8")); start=int(sys.argv[2]); resume=sys.argv[3]; smoke_path=Path(sys.argv[4]).resolve()
smoke=json.load(open(smoke_path,encoding="utf-8"))
assets_sha=hashlib.sha256(assets_path.read_bytes()).hexdigest()
if smoke.get("schema_version") != "greek_cpt_two_phase_smoke_verification_v1" or smoke.get("status") != "passed": raise SystemExit("two-phase GPU smoke has not passed")
if Path(smoke.get("training_assets_receipt",{}).get("path","")).resolve() != assets_path or smoke.get("training_assets_receipt",{}).get("sha256") != assets_sha: raise SystemExit("GPU smoke is bound to different training assets")
load=assets["init_checkpoint"]["root"]
if start:
    if not resume: raise SystemExit("resume checkpoint receipt required")
    value=json.load(open(resume,encoding="utf-8"))
    if value.get("iteration") != start: raise SystemExit("resume iteration drift")
    load=value["checkpoint_tree"]["root"]
d=assets["dependencies"]
print(assets["repository"]["root"],load,assets["megatron"]["root"],assets["tokenizer"]["root"],d["training_data_env"]["path"],d["trainer"]["path"])
PY
)

if (( START_ITERATION == 0 )); then
  [[ -z "$RESUME_CHECKPOINT_RECEIPT" ]] || { echo "ERROR: initial segment must not set a resume receipt" >&2; exit 2; }
else
  test -s "$RESUME_CHECKPOINT_RECEIPT" || { echo "ERROR: resume receipt missing" >&2; exit 2; }
fi

SBATCH="$PROBE_ROOT/clariden/train_segment.sbatch"
submission_dir="$OUTPUT_DIR/segment_submissions"
submission_receipt="$submission_dir/${START_ITERATION}_${END_ITERATION}.json"
if [[ "$DRY_RUN" == 0 ]]; then
  if (( START_ITERATION == 0 )); then
    [[ ! -e "$OUTPUT_DIR" ]] || { echo "ERROR: initial output already exists: $OUTPUT_DIR" >&2; exit 3; }
    mkdir -p "$submission_dir"
  else
    test -d "$OUTPUT_DIR" || { echo "ERROR: resume output is absent: $OUTPUT_DIR" >&2; exit 3; }
    mkdir -p "$submission_dir"
  fi
  test ! -e "$submission_receipt" || { echo "ERROR: segment already has a submission receipt" >&2; exit 3; }
fi
cmd=(sbatch --parsable
  --job-name="${RUN_TAG}_i${START_ITERATION}_${END_ITERATION}"
  --output="$OUTPUT_DIR/%x-%j.out" --error="$OUTPUT_DIR/%x-%j.err"
  --export="ALL,TRAINING_ASSETS_RECEIPT=$TRAINING_ASSETS_RECEIPT,SMOKE_VERIFICATION=$SMOKE_VERIFICATION,START_ITERATION=$START_ITERATION,END_ITERATION=$END_ITERATION,CPT_PHASE=$CPT_PHASE,INIT_CKPT=$INIT_CKPT,RESUME_CHECKPOINT_RECEIPT=$RESUME_CHECKPOINT_RECEIPT,TRAIN_SCRIPT=$TRAIN_SCRIPT,PROBE_ROOT=$PROBE_ROOT,REPO_ROOT=$REPO_ROOT,MEGATRON_LM_SWISSAI_DIR=$MEGATRON_DIR,FULL_CPT_TOKENIZER_DIR=$TOKENIZER_DIR,BRIDGE_DATA_ENV=$BRIDGE_DATA_ENV,OUTPUT_DIR=$OUTPUT_DIR,SCRIPT_DIR_OVERRIDE=$(dirname "$TRAIN_SCRIPT"),ACCOUNT=a0140,PARTITION=normal,NODES=16,GPUS_PER_NODE=4,LAUNCH_MODE=torchrun,TIME_LIMIT=08:00:00,DISABLE_SAVE=0,SAVE_INTERVAL=119,EVAL_INTERVAL=25,EVAL_ITERS=1"
  "$SBATCH")
watcher="$PROBE_ROOT/eval/watch_greekmmlu_checkpoints.sbatch"

echo "segment: phase=$CPT_PHASE iterations=$START_ITERATION..$END_ITERATION"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY:'; printf ' %q' "${cmd[@]}"; printf '\n'
  printf 'DRY: sbatch --dependency=after:<train-job-id> %q\n' "$watcher"
  echo "No job submitted."
  exit 0
fi
job_id=$("${cmd[@]}")
watch_job=$(sbatch --parsable --account=a0140 --partition=xfer \
  --dependency="after:$job_id" --job-name="${RUN_TAG}_eval_i${START_ITERATION}_${END_ITERATION}" \
  --output="$OUTPUT_DIR/%x-%j.out" --error="$OUTPUT_DIR/%x-%j.err" \
  --export="ALL,REPO_ROOT=$REPO_ROOT,RUN_TAG=$RUN_TAG,TRAIN_RUN_DIR=$OUTPUT_DIR,TRAINING_ASSETS_RECEIPT=$TRAINING_ASSETS_RECEIPT,TRAIN_JOB_ID=$job_id,START_ITERATION=$START_ITERATION,END_ITERATION=$END_ITERATION" \
  "$watcher")
python3 - "$submission_receipt" "$job_id" "$watch_job" "$START_ITERATION" "$END_ITERATION" "$CPT_PHASE" <<'PY'
import datetime,json,os,sys,tempfile
out,job,watch,start,end,phase=sys.argv[1:]
value={"schema_version":"greek_cpt_segment_submission_v1","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"job_id":job,"evaluation_watcher_job_id":watch,"start_iteration":int(start),"end_iteration":int(end),"phase":int(phase)}
fd,tmp=tempfile.mkstemp(prefix=".segment.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
echo "submitted training job $job_id and evaluation watcher $watch_job"

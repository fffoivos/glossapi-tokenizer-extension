#!/usr/bin/env bash
# Submit the isolated 1-node, 4-GPU two-phase smoke. Never submits production.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE_ROOT=$(cd "$HERE/.." && pwd)
: "${TRAINING_ASSETS_RECEIPT:?set TRAINING_ASSETS_RECEIPT}"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_GPU_LAUNCH="${CONFIRM_GPU_LAUNCH:-}"
RUN_TAG="${RUN_TAG:-greek_cpt25b_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/25b_midtraining_smokes}"
SMOKE_ROOT="${SMOKE_ROOT:-$RUN_ROOT/$RUN_TAG}"

case "$DRY_RUN" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
if [[ "$DRY_RUN" == 0 && "$CONFIRM_GPU_LAUNCH" != "GREEK_CPT25B_SMOKE" ]]; then
  echo "ERROR: live smoke requires CONFIRM_GPU_LAUNCH=GREEK_CPT25B_SMOKE" >&2
  exit 2
fi

read -r REPO_ROOT INIT_CKPT MEGATRON_DIR TOKENIZER_DIR BRIDGE_DATA_ENV TRAIN_SCRIPT < <(
  python3 - "$TRAINING_ASSETS_RECEIPT" "$PROBE_ROOT" <<'PY'
import json,sys
from pathlib import Path
assets_path=Path(sys.argv[1]).resolve(); probe=Path(sys.argv[2]).resolve()
assets=json.load(open(assets_path,encoding="utf-8"))
if assets.get("schema_version") != "greek_cpt_training_assets_receipt_v1" or assets.get("status") != "frozen":
    raise SystemExit("training assets are not frozen")
repo=Path(assets["repository"]["root"]).resolve()
expected=(repo/"subprojects/05_token_distillation_cpt/06_25b_midtraining_probe").resolve()
if probe != expected:
    raise SystemExit(f"launcher is not in the frozen repository: {probe} != {expected}")
d=assets["dependencies"]
print(repo,assets["init_checkpoint"]["root"],assets["megatron"]["root"],assets["tokenizer"]["root"],d["training_data_env"]["path"],d["trainer"]["path"])
PY
)

PHASE1_RECEIPT="$SMOKE_ROOT/receipts/iteration_1.json"
PHASE2_RECEIPT="$SMOKE_ROOT/receipts/iteration_2.json"
SMOKE_VERIFICATION="$SMOKE_ROOT/smoke_verification.json"
PHASE1_LOG_PATTERN="$SMOKE_ROOT/logs/phase1-%j.out"
PHASE1_ERR_PATTERN="$SMOKE_ROOT/logs/phase1-%j.err"
PHASE2_LOG_PATTERN="$SMOKE_ROOT/logs/phase2-%j.out"
PHASE2_ERR_PATTERN="$SMOKE_ROOT/logs/phase2-%j.err"

if [[ "$DRY_RUN" == 0 ]]; then
  [[ ! -e "$SMOKE_ROOT" ]] || { echo "ERROR: smoke output already exists: $SMOKE_ROOT" >&2; exit 3; }
  mkdir -p "$SMOKE_ROOT/logs" "$SMOKE_ROOT/receipts"
fi

dry_counter=0
submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    dry_counter=$((dry_counter+1))
    printf 'DRY:' >&2; printf ' %q' sbatch --parsable "$@" >&2; printf '\n' >&2
    echo "DRY$dry_counter"
  else
    sbatch --parsable "$@"
  fi
}

SBATCH="$PROBE_ROOT/clariden/smoke_train_segment.sbatch"
common_export="TRAINING_ASSETS_RECEIPT=$TRAINING_ASSETS_RECEIPT,TRAIN_SCRIPT=$TRAIN_SCRIPT,PROBE_ROOT=$PROBE_ROOT,REPO_ROOT=$REPO_ROOT,MEGATRON_LM_SWISSAI_DIR=$MEGATRON_DIR,FULL_CPT_TOKENIZER_DIR=$TOKENIZER_DIR,BRIDGE_DATA_ENV=$BRIDGE_DATA_ENV,CPT_SMOKE=1,SCRIPT_DIR_OVERRIDE=$(dirname "$TRAIN_SCRIPT"),ACCOUNT=a0140,PARTITION=normal,NODES=1,GPUS_PER_NODE=4,LAUNCH_MODE=torchrun,TIME_LIMIT=01:30:00,DISABLE_SAVE=0,SAVE_INTERVAL=1,EVAL_INTERVAL=1,EVAL_ITERS=1"

phase1_job=$(submit --account=a0140 --partition=normal --nodes=1 --ntasks-per-node=1 \
  --gpus-per-node=4 --gres=gpu:4 --cpus-per-task=288 --mem=460G --time=01:30:00 \
  --job-name="${RUN_TAG}_p1" --output="$PHASE1_LOG_PATTERN" --error="$PHASE1_ERR_PATTERN" \
  --export="ALL,$common_export,START_ITERATION=0,END_ITERATION=1,CPT_PHASE=1,INIT_CKPT=$INIT_CKPT,RESUME_CHECKPOINT_RECEIPT=,SEGMENT_OUTPUT_DIR=$SMOKE_ROOT/phase1,SMOKE_CHECKPOINT_RECEIPT=$PHASE1_RECEIPT" \
  "$SBATCH")

phase2_job=$(submit --account=a0140 --partition=normal --nodes=1 --ntasks-per-node=1 \
  --gpus-per-node=4 --gres=gpu:4 --cpus-per-task=288 --mem=460G --time=01:30:00 \
  --dependency="afterok:$phase1_job" --job-name="${RUN_TAG}_p2" \
  --output="$PHASE2_LOG_PATTERN" --error="$PHASE2_ERR_PATTERN" \
  --export="ALL,$common_export,START_ITERATION=1,END_ITERATION=2,CPT_PHASE=2,INIT_CKPT=$SMOKE_ROOT/phase1/checkpoints,RESUME_CHECKPOINT_RECEIPT=$PHASE1_RECEIPT,SEGMENT_OUTPUT_DIR=$SMOKE_ROOT/phase2,SMOKE_CHECKPOINT_RECEIPT=$PHASE2_RECEIPT" \
  "$SBATCH")

phase1_log="$SMOKE_ROOT/logs/phase1-$phase1_job.out"
phase1_err="$SMOKE_ROOT/logs/phase1-$phase1_job.err"
phase2_log="$SMOKE_ROOT/logs/phase2-$phase2_job.out"
phase2_err="$SMOKE_ROOT/logs/phase2-$phase2_job.err"
verify_job=$(submit --account=a0140 --partition=normal --dependency="afterok:$phase2_job" \
  --job-name="${RUN_TAG}_verify" --output="$SMOKE_ROOT/logs/verify-%j.out" \
  --error="$SMOKE_ROOT/logs/verify-%j.err" \
  --export="ALL,PROBE_ROOT=$PROBE_ROOT,TRAINING_ASSETS_RECEIPT=$TRAINING_ASSETS_RECEIPT,SMOKE_ROOT=$SMOKE_ROOT,PHASE1_RECEIPT=$PHASE1_RECEIPT,PHASE2_RECEIPT=$PHASE2_RECEIPT,PHASE1_LOG=$phase1_log,PHASE1_ERR=$phase1_err,PHASE2_LOG=$phase2_log,PHASE2_ERR=$phase2_err,SMOKE_VERIFICATION=$SMOKE_VERIFICATION" \
  "$PROBE_ROOT/clariden/verify_smoke.sbatch")

if [[ "$DRY_RUN" == 0 ]]; then
  python3 - "$SMOKE_ROOT/submission.json" "$TRAINING_ASSETS_RECEIPT" "$phase1_job" "$phase2_job" "$verify_job" "$SMOKE_VERIFICATION" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
out,assets,p1,p2,verify,evidence=sys.argv[1:]
sha=lambda p: hashlib.sha256(open(p,"rb").read()).hexdigest()
value={"schema_version":"greek_cpt_two_phase_smoke_submission_v1","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"training_assets_receipt":{"path":os.path.realpath(assets),"sha256":sha(assets)},"jobs":{"phase1":p1,"phase2":p2,"verify":verify},"expected_verification":evidence}
fd,tmp=tempfile.mkstemp(prefix=".submission.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
fi

printf 'smoke_root=%s\nphase1=%s\nphase2=%s\nverify=%s\nverification=%s\n' \
  "$SMOKE_ROOT" "$phase1_job" "$phase2_job" "$verify_job" "$SMOKE_VERIFICATION"

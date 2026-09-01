#!/usr/bin/env bash
# Submit the self-advancing, receipt-gated campaign. Dry-run is the default.
set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}

: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
DRY_RUN=${DRY_RUN:-1}
CONFIRM_GPU_LAUNCH=${CONFIRM_GPU_LAUNCH:-}
RUN_TAG=${RUN_TAG:-mini_cpt5_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_PARENT=${RUN_PARENT:-/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments}
RUN_ROOT=${RUN_ROOT:-$RUN_PARENT/$RUN_TAG}
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
if [[ "$DRY_RUN" == 0 && "$CONFIRM_GPU_LAUNCH" != FIVE_ARM_MINI_CPT ]]; then
  echo "live launch requires CONFIRM_GPU_LAUNCH=FIVE_ARM_MINI_CPT" >&2; exit 2
fi

readarray -t values < <("$HOST_PYTHON" - "$CAMPAIGN_MANIFEST" "$RUN_PARENT" <<'PY'
import json,shutil,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve(); c=json.load(open(p)); parent=Path(sys.argv[2])
if c.get("schema_version") != "apertus_mini_campaign_manifest_v1" or c.get("status") != "frozen":
    raise SystemExit("campaign manifest is not frozen")
free=shutil.disk_usage(parent).free
minimum=int(c["runtime"]["minimum_starting_checkpoint_headroom_bytes"])
if free < minimum: raise SystemExit(f"insufficient starting checkpoint headroom: {free} < {minimum}")
print(c["assets"]["scientific_bundle"])
print(c["assets"]["checkpoint_plan"]["path"])
print(free)
PY
)
SCIENTIFIC_BUNDLE=${values[0]}
CHECKPOINT_PLAN=${values[1]}
FREE_BYTES=${values[2]}
train_sbatch="$SCIENTIFIC_BUNDLE/clariden/train_five_arm_segment.sbatch"
watch_sbatch="$SCIENTIFIC_BUNDLE/clariden/watch_checkpoint_evaluations.sbatch"
supervisor_sbatch="$SCIENTIFIC_BUNDLE/clariden/supervise_production_segment.sbatch"
initial_validation_sbatch="$SCIENTIFIC_BUNDLE/clariden/run_initial_validation.sbatch"
for path in "$train_sbatch" "$watch_sbatch" "$supervisor_sbatch" "$initial_validation_sbatch"; do
  [[ -f "$path" ]] || { echo "missing production script: $path" >&2; exit 2; }
done

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY RUN campaign=$CAMPAIGN_MANIFEST run_root=$RUN_ROOT free_bytes=$FREE_BYTES"
  echo "initial validation -> segment0 + watcher0 + supervisor0 -> receipt-gated segment1 -> training completion"
  echo "supervisors retry only Slurm infrastructure failures, at most twice, from the newest common five-arm checkpoint"
  exit 0
fi
[[ ! -e "$RUN_ROOT" ]] || { echo "refusing to reuse run root: $RUN_ROOT" >&2; exit 3; }
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/receipts"

common_export="ALL,CAMPAIGN_MANIFEST=$CAMPAIGN_MANIFEST,RUN_ROOT=$RUN_ROOT,SCIENTIFIC_BUNDLE=$SCIENTIFIC_BUNDLE,RUN_TAG=$RUN_TAG"
initial_validation=$(sbatch --parsable --job-name="${RUN_TAG}_initval" \
  --output="$RUN_ROOT/logs/%x-%j.out" --error="$RUN_ROOT/logs/%x-%j.err" \
  --export="$common_export,OUTPUT_ROOT=$RUN_ROOT/initial_validation" "$initial_validation_sbatch")
train0=$(sbatch --parsable --dependency="afterok:$initial_validation" --job-name="${RUN_TAG}_s0" \
  --output="$RUN_ROOT/logs/%x-%j.out" --error="$RUN_ROOT/logs/%x-%j.err" \
  --export="$common_export,SEGMENT_ID=0" "$train_sbatch")
watch0=$(sbatch --parsable --dependency="after:$train0" --job-name="${RUN_TAG}_watch0" \
  --output="$RUN_ROOT/logs/%x-%j.out" --error="$RUN_ROOT/logs/%x-%j.err" \
  --export="$common_export,SEGMENT_ID=0,SEGMENT_ATTEMPT=0,TRAIN_JOB_ID=$train0" "$watch_sbatch")
supervisor0=$(sbatch --parsable --dependency="after:$train0" --job-name="${RUN_TAG}_supervise0" \
  --output="$RUN_ROOT/logs/%x-%j.out" --error="$RUN_ROOT/logs/%x-%j.err" \
  --export="$common_export,SEGMENT_ID=0,SEGMENT_ATTEMPT=0,TRAIN_JOB_ID=$train0,WATCH_JOB_ID=$watch0" "$supervisor_sbatch")

"$HOST_PYTHON" - "$RUN_ROOT/submission_receipt.json" "$CAMPAIGN_MANIFEST" "$RUN_TAG" \
  "$initial_validation" "$train0" "$watch0" "$supervisor0" <<'PY'
import datetime,json,os,sys,tempfile
out,manifest,tag,*jobs=sys.argv[1:]
names=("initial_validation","train0","watch0","supervisor0")
payload={
 "schema_version":"apertus_mini_campaign_submission_v2",
 "status":"submitted",
 "submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "run_tag":tag,"campaign_manifest":manifest,"jobs":dict(zip(names,jobs)),
 "continuation":"segment supervisors append immutable events and submit subsequent attempts/segment",
}
fd,tmp=tempfile.mkstemp(prefix=".submission.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
print(json.dumps(payload,sort_keys=True))
PY

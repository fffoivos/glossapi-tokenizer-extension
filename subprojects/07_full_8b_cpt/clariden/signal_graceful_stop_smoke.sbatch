#!/usr/bin/env bash
#SBATCH --job-name=full8b-signal-smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_SMOKE_RUN_ROOT:?set smoke root}"
: "${FULL8_SMOKE_TRAIN_JOB:?set training job id}"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
log="$FULL8_SMOKE_RUN_ROOT/segments/updates_0_288/training.log"
for _ in $(seq 1 600); do
  latest=$(python3 - "$log" <<'PY'
import re,sys
latest=0
try:
 for line in open(sys.argv[1],errors="replace"):
  match=re.search(r"iteration\s+(\d+)\s*/",line)
  if match: latest=int(match.group(1))
except FileNotFoundError: pass
print(latest)
PY
)
  if ((latest >= 10)); then
    scancel --batch --signal=USR1 "$FULL8_SMOKE_TRAIN_JOB"
    python3 - "$FULL8_SMOKE_RUN_ROOT/signal_request.json" "$FULL8_SMOKE_TRAIN_JOB" "$latest" <<'PY'
import datetime,json,sys
out,job,latest=sys.argv[1:]
open(out,"w").write(json.dumps({"schema_version":"apertus_full_8b_signal_request_v1","status":"sent","sent_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"job_id":job,"signal":"USR1","last_iteration_before_signal":int(latest)},indent=2,sort_keys=True)+"\n")
PY
    exit 0
  fi
  sleep 2
done
echo "training did not reach the signal threshold" >&2
exit 2

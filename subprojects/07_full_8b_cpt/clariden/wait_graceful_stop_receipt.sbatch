#!/usr/bin/env bash
#SBATCH --job-name=full8b-stop-smoke-wait
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_GRACEFUL_STOP_SMOKE:?set expected graceful-stop receipt}"
for _ in $(seq 1 720); do
  if [[ -s "$FULL8_GRACEFUL_STOP_SMOKE" ]] && python3 - "$FULL8_GRACEFUL_STOP_SMOKE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get("schema_version")=="apertus_full_8b_graceful_stop_smoke_v1" and x.get("status")=="passed" and x.get("resume",{}).get("passed") is True else 1)
PY
  then
    exit 0
  fi
  sleep 60
done
echo "graceful-stop smoke receipt did not pass inside 12 hours" >&2
exit 2

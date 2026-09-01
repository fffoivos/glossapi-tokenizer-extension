#!/usr/bin/env bash
#SBATCH --job-name=full8b-wait-nested-proof
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_NESTED_SBATCH_PROOF:?set expected nested-submit proof}"
for _ in $(seq 1 120); do
  [[ -f "$FULL8_NESTED_SBATCH_PROOF" ]] && break
  sleep 5
done
[[ -f "$FULL8_NESTED_SBATCH_PROOF" ]] || { echo "nested-submit proof did not arrive" >&2; exit 2; }
python3 - "$FULL8_NESTED_SBATCH_PROOF" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get("schema_version")!="apertus_full_8b_nested_sbatch_proof_v1" or x.get("status")!="passed":
 raise SystemExit("nested-submit proof failed")
if x.get("nested_submit_flag")!="--uenv-passthrough=ignore" or not x.get("parent_job_id") or not x.get("child_job_id"):
 raise SystemExit("nested-submit proof is incomplete")
PY

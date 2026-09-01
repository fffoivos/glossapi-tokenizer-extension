#!/usr/bin/env bash
#SBATCH --job-name=h2g-producer-compat
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable current code root}"
: "${H2G_CODE_RECEIPT:?set immutable current code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_PRODUCER_BUNDLE_RECEIPTS:?set the exact space-separated producer bundle receipts}"
: "${H2G_PRODUCER_COMPATIBILITY_OUTPUT:?set a new immutable compatibility output path}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "producer compatibility freeze requires one debug node" >&2; exit 2;
}
verify="$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
args=()
current_seen=0
for receipt in $H2G_PRODUCER_BUNDLE_RECEIPTS; do
  [[ -s "$receipt" ]] || { echo "producer bundle receipt missing: $receipt" >&2; exit 2; }
  root=$(/usr/bin/python3.11 - "$receipt" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["root"])
PY
)
  /usr/bin/python3.11 "$verify" --root "$root" --receipt "$receipt" --kind scientific
  [[ "$receipt" != "$H2G_CODE_RECEIPT" ]] || current_seen=1
  args+=(--producer-receipt "$receipt")
done
[[ "$current_seen" == 1 ]] || { echo "producer receipts omit current bundle" >&2; exit 2; }
output="$H2G_PRODUCER_COMPATIBILITY_OUTPUT"
mkdir -p "$(dirname "$output")"
[[ ! -e "$output" ]] || { echo "immutable producer compatibility authority exists: $output" >&2; exit 2; }
export H2G_CODE_ROOT H2G_CODE_RECEIPT
/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_producer_bundle_compatibility.py" \
  "${args[@]}" --output "$output"

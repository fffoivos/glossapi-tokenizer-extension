#!/usr/bin/env bash
#SBATCH --job-name=h2g-adopt-data
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable experiment code root}"
: "${H2G_CODE_RECEIPT:?set immutable experiment code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${APERTUS_EFFICIENCY_ROOT:?set immutable canonical runner root}"
: "${APERTUS_EFFICIENCY_CODE_RECEIPT:?set immutable canonical runner receipt}"
: "${H2G_DATA_PYTHON:?set verified data runtime Python}"
: "${H2G_CANONICAL_DATA_OUTPUT_ROOT:=$H2G_STAGE_ROOT/canonical/pre_main_data_v2}"

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "canonical data adoption requires one debug node" >&2
  exit 2
}

verify_bundle="$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify_bundle" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
/usr/bin/python3.11 "$verify_bundle" \
  --root "$APERTUS_EFFICIENCY_ROOT" \
  --receipt "$APERTUS_EFFICIENCY_CODE_RECEIPT" --kind efficiency

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
output_root="$H2G_CANONICAL_DATA_OUTPUT_ROOT"
case "$output_root" in
  "$H2G_STAGE_ROOT"/canonical/*) ;;
  *) echo "refusing canonical data output outside stage root: $output_root" >&2; exit 2 ;;
esac
"$H2G_DATA_PYTHON" "$subproject/scripts/adopt_canonical_pre_main_data.py" \
  --stage-root "$H2G_STAGE_ROOT" \
  --canonical-root "$APERTUS_EFFICIENCY_ROOT" \
  --canonical-code-receipt "$APERTUS_EFFICIENCY_CODE_RECEIPT" \
  --hf-inventory "$H2G_STAGE_ROOT/receipts/hf_inventory_canonical_b940fd1.json" \
  --reader-smoke "$H2G_STAGE_ROOT/receipts/profile_benchmark_8b_dp32_16node.json" \
  --output-root "$output_root"

printf '%s\n' "$output_root/training_data_manifest_gate.json"

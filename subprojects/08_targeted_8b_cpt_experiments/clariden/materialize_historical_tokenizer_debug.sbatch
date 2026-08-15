#!/usr/bin/env bash
#SBATCH --job-name=h2g-tokenizer
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code-bundle receipt}"
: "${H2G_STAGE_ROOT:?set immutable experiment stage root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "historical tokenizer materialization requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

source_root="$H2G_CODE_ROOT/frozen_historical_tokenizer_148480"
output_root="$H2G_STAGE_ROOT/assets/tokenizer_148480"
receipt="$H2G_STAGE_ROOT/receipts/tokenizer_148480.json"
mkdir -p "$H2G_STAGE_ROOT/assets" "$H2G_STAGE_ROOT/receipts"
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/materialize_historical_tokenizer.py" \
  --source-root "$source_root" \
  --output-root "$output_root" \
  --output-receipt "$receipt"

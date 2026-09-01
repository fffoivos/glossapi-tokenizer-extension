#!/usr/bin/env bash
#SBATCH --job-name=h2g-cache-freeze
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_PHASE:?set phase 1, 2 or 3}"
: "${H2G_PHASE_DATA_PATH_SPEC:?set exact frozen phase data-path spec}"
: "${H2G_PHASE_CACHE_ROOT:?set completed GPTDataset cache root}"
: "${H2G_PHASE_BLEND_MANIFEST:?set frozen phase blend manifest}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "phase cache freezing requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
receipt="$H2G_STAGE_ROOT/receipts/phase_${H2G_PHASE}_blend_cache.json"
export H2G_CODE_ROOT H2G_CODE_RECEIPT
/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_phase_blend_cache.py" \
  --phase "$H2G_PHASE" --data-path-spec "$H2G_PHASE_DATA_PATH_SPEC" \
  --cache-root "$H2G_PHASE_CACHE_ROOT" --blend-manifest "$H2G_PHASE_BLEND_MANIFEST" \
  --output "$receipt"

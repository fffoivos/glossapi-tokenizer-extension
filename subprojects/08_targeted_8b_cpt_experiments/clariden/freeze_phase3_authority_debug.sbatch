#!/usr/bin/env bash
#SBATCH --job-name=h2g-phase3-authority
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "Phase-3 authority requires one debug node" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
out="$H2G_STAGE_ROOT/receipts/phase3_authority.json"
[[ ! -e "$out" ]] || { echo "immutable Phase-3 authority exists: $out" >&2; exit 2; }
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/freeze_phase3_authority.py" \
  --unseen-catalog-receipt "$H2G_STAGE_ROOT/receipts/phase3_unseen_catalog.json" \
  --openarchives-tokenized-receipt "$H2G_STAGE_ROOT/receipts/tokenized_phase3_openarchives.json" \
  --foreign-tokenized-receipt "$H2G_STAGE_ROOT/receipts/tokenized_phase3_foreign.json" \
  --old-greek-tokenized-receipt "$H2G_STAGE_ROOT/receipts/tokenized_phase3_old_greek.json" \
  --phase3-cache-receipt "$H2G_STAGE_ROOT/receipts/phase_3_blend_cache.json" \
  --output "$out"

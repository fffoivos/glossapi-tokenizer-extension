#!/usr/bin/env bash
#SBATCH --job-name=t8b-final-selection
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "selection finalization must run on debug" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
stage="$TARGET8_STAGE_ROOT/experiment_a"
receipts="$stage/receipts"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_selected_pools.py" \
  --academic-pool-receipt "$receipts/academic_pool_receipt.json" --poly-pool-receipt "$receipts/poly_pool_receipt.json" \
  --hplt-decontamination-manifest "$stage/hplt_candidates_decontaminated/decontamination_manifest.json" \
  --hplt-decontamination-audit "$stage/hplt_candidates_decontaminated/audit_receipt.json" \
  --hplt-validation-manifest "$stage/hplt_candidates_training_eligible/exclusion_manifest.json" \
  --hplt-validation-audit "$stage/hplt_candidates_training_eligible/audit_receipt.json" \
  --pool-corpus-receipt "$stage/inventory/pool_corpus_receipt.json" \
  --decontamination-output "$receipts/decontamination_summary.json" \
  --selected-pools-output "$receipts/selected_pools_receipt.json"

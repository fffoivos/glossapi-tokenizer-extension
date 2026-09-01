#!/usr/bin/env bash
#SBATCH --job-name=t8b-freeze-bins
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
: "${TARGET8_REPOSITORY_COMMIT:?set full repository commit}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
TOKENIZER_ROOT=${TARGET8_TOKENIZER_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}
PARENT_POOL=${TARGET8_PARENT_POOL_RECEIPT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260807T063000Z-d0-v3/inventory/pool_corpus_receipt.json}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "binary contract freeze must run on debug" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
out="$TARGET8_STAGE_ROOT/experiment_a/binary_contract"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_targeted_binary_inputs.py" \
  --academic-pool-receipt "$TARGET8_STAGE_ROOT/experiment_a/receipts/academic_pool_receipt.json" \
  --hplt-exclusion-manifest "$TARGET8_STAGE_ROOT/experiment_a/hplt_candidates_training_eligible/exclusion_manifest.json" \
  --hplt-exclusion-audit "$TARGET8_STAGE_ROOT/experiment_a/hplt_candidates_training_eligible/audit_receipt.json" \
  --poly-pool-receipt "$TARGET8_STAGE_ROOT/experiment_a/receipts/poly_pool_receipt.json" \
  --parent-pool-receipt "$PARENT_POOL" --tokenizer-root "$TOKENIZER_ROOT" \
  --bridge-builder "$TARGET8_CODE_ROOT/subprojects/05_token_distillation_cpt/05_training_dataset_bridge/scripts/build_binary_shard.py" \
  --bridge-common "$TARGET8_CODE_ROOT/subprojects/05_token_distillation_cpt/05_training_dataset_bridge/scripts/bridge_common.py" \
  --code-bundle-receipt "$TARGET8_CODE_RECEIPT" --repository-commit "$TARGET8_REPOSITORY_COMMIT" \
  --output-dir "$out"

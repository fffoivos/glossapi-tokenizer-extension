#!/usr/bin/env bash
#SBATCH --job-name=t8b-final-acad
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
TOKENIZER_JSON=${TARGET8_TOKENIZER_JSON:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992/tokenizer.json}
RELEASE_COUNTS=${TARGET8_RELEASE_COUNTS:-/capstor/scratch/cscs/fffoivos/hf_v2_anonymized_releases/20260811T121833Z/run/release/manifests/token_counts.json}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "academic token finalization must run on debug" >&2; exit 2; }

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
decontam="$TARGET8_STAGE_ROOT/experiment_a/academic_decontaminated"
eligible="$TARGET8_STAGE_ROOT/experiment_a/academic_training_eligible"
receipts="$TARGET8_STAGE_ROOT/experiment_a/receipts"
mkdir -p "$receipts"
[[ -s "$decontam/audit_receipt.json" && -s "$eligible/audit_receipt.json" ]] || { echo "academic audits missing" >&2; exit 2; }

uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/count_parquet_tokens.py" \
  --input "$decontam/dropped" --tokenizer-json "$TOKENIZER_JSON" --workers 32 \
  --schema-version targeted_8b_academic_greekmmlu_removed_tokens_v1 \
  --output "$receipts/academic_greekmmlu_removed_tokens.json"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/count_parquet_tokens.py" \
  --input "$eligible/excluded" --tokenizer-json "$TOKENIZER_JSON" --workers 32 \
  --schema-version targeted_8b_academic_validation_excluded_tokens_v1 \
  --output "$receipts/academic_validation_excluded_tokens.json"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_academic_pool.py" \
  --release-token-counts "$RELEASE_COUNTS" \
  --decontamination-manifest "$decontam/decontamination_manifest.json" \
  --decontamination-audit "$decontam/audit_receipt.json" \
  --decontamination-dropped-tokens "$receipts/academic_greekmmlu_removed_tokens.json" \
  --validation-exclusion-manifest "$eligible/exclusion_manifest.json" \
  --validation-exclusion-audit "$eligible/audit_receipt.json" \
  --validation-excluded-tokens "$receipts/academic_validation_excluded_tokens.json" \
  --output "$receipts/academic_pool_receipt.json"

#!/usr/bin/env bash
#SBATCH --job-name=t8b-count-poly
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
: "${TARGET8_POLY_TRAIN_PARQUET:?set exact existing receipt-bound poly_train.parquet}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
TOKENIZER_JSON=${TARGET8_TOKENIZER_JSON:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992/tokenizer.json}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "poly token count must run on debug" >&2; exit 2; }

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
output="$TARGET8_STAGE_ROOT/experiment_a/poly_train_token_receipt.json"
[[ ! -e "$output" ]] || { echo "immutable poly token receipt exists" >&2; exit 2; }
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/count_parquet_tokens.py" \
  --input "$TARGET8_POLY_TRAIN_PARQUET" --tokenizer-json "$TOKENIZER_JSON" \
  --expected-rows 14929 --workers 32 --schema-version poly_train_token_receipt_v1 \
  --output "$output"

#!/usr/bin/env bash
#SBATCH --job-name=t8b-count-release-poly
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
TOKENIZER_JSON=${TARGET8_TOKENIZER_JSON:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992/tokenizer.json}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "release polytonic token count must run on debug" >&2; exit 2; }

/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
stage="$TARGET8_STAGE_ROOT/experiment_a"
config="$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/experiment_a_recipe.json"
audit="$stage/polytonic_release_source_audit.json"
input="$stage/poly_source/data"
output="$stage/poly_release_token_receipt.json"
[[ -s "$audit" && -d "$input" ]] || { echo "release-polytonic audit or extraction input missing" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "immutable release-polytonic token receipt exists" >&2; exit 2; }
mapfile -t sources < <(jq -r '.modern_data.polytonic.source_datasets[]' "$config")
args=()
for source in "${sources[@]}"; do args+=(--source-dataset "$source"); done
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/count_parquet_tokens.py" \
  --input "$input" --tokenizer-json "$TOKENIZER_JSON" --workers 32 \
  --schema-version targeted_8b_release_polytonic_token_receipt_v1 \
  --release-source-selection-audit "$audit" "${args[@]}" --output "$output"

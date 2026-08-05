#!/usr/bin/env bash
#SBATCH --job-name=full8b-data-freeze
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_STAGE_ROOT:?set new immutable data stage root}"
SOURCE_ROOT=${FULL8_SOURCE_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/cpt25b_midtraining/20260731T124000Z-cpt25b-v1}
TOKENIZER_ROOT=${FULL8_TOKENIZER_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/freeze_source_inventory.py" \
  --source-root "$SOURCE_ROOT" --output-dir "$FULL8_STAGE_ROOT/inventory" \
  --tokenizer-root "$TOKENIZER_ROOT"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/dataset/build_packing_plan.py" \
  --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --output "$FULL8_STAGE_ROOT/inventory/packing_plan.json"

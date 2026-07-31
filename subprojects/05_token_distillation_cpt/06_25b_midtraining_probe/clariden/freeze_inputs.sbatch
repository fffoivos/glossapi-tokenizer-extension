#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_freeze
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -euo pipefail
HERE=${CPT_CLARIDEN_DIR:?submit through submit_data_pipeline.sh}
source "$HERE/paths.env"
cpt_require_runtime
test -s "$DATASET_MANIFEST"
test -s "$REPLAY_RECEIPT"
test -s "$OLD_GREEK_RECEIPT"
test -s "$DECONTAM_BINDING"
mkdir -p "$STAGE_ROOT"
cpt_python "$CPT_DIR/dataset/freeze_inputs.py" \
  --recipe "$RECIPE" --legacy-bridge-config "$LEGACY_CONFIG" \
  --dataset-root "$DATASET_ROOT" --dataset-manifest "$DATASET_MANIFEST" \
  --tokenizer-root "$TOKENIZER_DIR" --scratch-root "$SCRATCH_ROOT" \
  --replay-acquisition-receipt "$REPLAY_RECEIPT" \
  --old-greek-build-receipt "$OLD_GREEK_RECEIPT" \
  --decontamination-binding "$DECONTAM_BINDING" --repo-root "$REPO_ROOT" \
  --output "$INPUT_RECEIPT"

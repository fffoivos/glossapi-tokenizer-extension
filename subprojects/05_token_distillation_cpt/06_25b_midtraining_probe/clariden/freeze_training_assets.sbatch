#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_assets
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
HERE=${CPT_CLARIDEN_DIR:?submit through submit_data_pipeline.sh}
source "$HERE/paths.env"
cpt_require_runtime
test -s "$BRIDGE_MANIFEST"
test -s "$TRAINING_DATA_ENV"
test -s "$INIT_ROOT/production_init_verification.json"
test -s "$ROUNDTRIP_ROOT/work/verification.json"
cpt_python "$CPT_DIR/initialization/freeze_training_assets.py" \
  --init-checkpoint "$ROUNDTRIP_ROOT/megatron_tp2_r17patched" \
  --init-evidence "$INIT_ROOT/production_init_verification.json" \
  --roundtrip-verification "$ROUNDTRIP_ROOT/work/verification.json" \
  --megatron-dir "$MEGATRON_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
  --hf-conversion-template "$INIT_ROOT/production" \
  --bridge-manifest "$BRIDGE_MANIFEST" --training-data-env "$TRAINING_DATA_ENV" \
  --repo-root "$REPO_ROOT" --output "$TRAINING_ASSETS_RECEIPT"

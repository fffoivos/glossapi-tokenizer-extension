#!/usr/bin/env bash
#SBATCH --job-name=cpt5_assets
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${BRIDGE_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
bridge_require_paths
test -s "$BRIDGE_MANIFEST"
bridge_python "$TRAINING_ASSETS_SCRIPT" \
    --bridge-manifest "$BRIDGE_MANIFEST" \
    --config "$BRIDGE_CONFIG" \
    --repo-root "$REPO_ROOT" \
    --scratch-root "$SCRATCH_ROOT" \
    --megatron-dir "$MEGATRON_DIR" \
    --trainer "$TRAIN_SCRIPT" \
    --training-env "$TRAINING_ENV" \
    --output "$TRAINING_ASSETS_RECEIPT"

#!/usr/bin/env bash
#SBATCH --job-name=v2_stage_starcoder
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.err
# Stage an Apertus-family code replay/forgetting pool. Requires the user/HF
# token to have accepted bigcode/starcoderdata terms.
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$V2_DIR/paths.env" ] && [ -f "$V2_DIR/../paths.env" ]; then
  V2_DIR="$(cd "$V2_DIR/.." && pwd)"
fi
source "$V2_DIR/paths.env"
mkdir -p "$RUN_ROOT" "$SC/cpt_corpus/replay/starcoderdata_v2"
check_build_py
run_build_py -u "$V2/dataset/stage_starcoderdata_subset.py" \
  --output-root "$SC/cpt_corpus/replay/starcoderdata_v2" \
  --plan "${STARCODER_PLAN:-python:8,javascript:6,java:4,go:3,rust:2,cpp:2,typescript:3}"
echo "starcoder staging DONE $(date -u)"

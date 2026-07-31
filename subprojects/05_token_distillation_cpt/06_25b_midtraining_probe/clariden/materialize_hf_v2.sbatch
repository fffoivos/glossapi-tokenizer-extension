#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_hfv2
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
: "${REPO_ROOT:?set REPO_ROOT}"
: "${HF_TOKEN:?inject HF_TOKEN for the gated dataset}"
DONOR_ROOT="${DONOR_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/agent1-v5-clariden-debug-20260715T111552Z-30c72e9/release-deduplicated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/greek-cleaned-v2-3f97cec4}"
CACHE_DIR="${CACHE_DIR:-/capstor/scratch/cscs/fffoivos/hf_cache/greek-cleaned-v2}"
mkdir -p "$CACHE_DIR" "$(dirname "$OUTPUT_ROOT")"
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=120
uenv run pytorch/v2.9.1:v2 --view=default -- \
  python3 "$REPO_ROOT/subprojects/05_token_distillation_cpt/06_25b_midtraining_probe/clariden/materialize_hf_v2.py" \
    --donor-root "$DONOR_ROOT" --output-root "$OUTPUT_ROOT" --cache-dir "$CACHE_DIR"

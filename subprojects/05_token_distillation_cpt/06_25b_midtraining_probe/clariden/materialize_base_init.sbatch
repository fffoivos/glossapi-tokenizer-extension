#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_baseinit
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
set -euo pipefail
HERE=${CPT_CLARIDEN_DIR:?submit through submit_data_pipeline.sh}
source "$HERE/paths.env"
: "${HF_TOKEN:?inject HF_TOKEN}"
mkdir -p "$CAP_ROOT/hf_cache/apertus-tokenizer-extension" "$(dirname "$BASE_INIT_HF")"
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=120
uenv run "$CPT_UENV" --view=default -- python3 "$CPT_DIR/clariden/materialize_base_init.py" \
  --output-root "$BASE_INIT_HF" --cache-dir "$CAP_ROOT/hf_cache/apertus-tokenizer-extension"

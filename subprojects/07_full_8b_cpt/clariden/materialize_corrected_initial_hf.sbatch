#!/usr/bin/env bash
#SBATCH --job-name=full8b-init-hf-anchor
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_INITIAL_HF_SOURCE:?set canonical zero-drift HF round trip}"
: "${FULL8_INITIAL_HF_ROOT:?set new corrected HF model root}"
: "${FULL8_INITIAL_HF_RECEIPT:?set new corrected HF receipt}"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/materialize_corrected_initial_hf.py" \
  --source "$FULL8_INITIAL_HF_SOURCE" \
  --roundtrip-verification /capstor/scratch/cscs/fffoivos/models/greek-cpt25b-init-roundtrip/20260731T124000Z-cpt25b-v1/work/verification.json \
  --expected-verification-sha256 06ddc69f4787f470b85d4be55992f898b98d8ab4fd8ab34a89a4d3bb96eb7e33 \
  --expected-tokenizer-sha256 bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b \
  --output-root "$FULL8_INITIAL_HF_ROOT" --receipt "$FULL8_INITIAL_HF_RECEIPT"

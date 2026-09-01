#!/usr/bin/env bash
#SBATCH --job-name=t8b-correct-init-hf
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_INITIAL_HF_SOURCE:?set verified HF round-trip source}"
: "${TARGET8_ROUNDTRIP_VERIFICATION:?set zero-drift round-trip verification}"
: "${TARGET8_TD_INITIALIZATION_VERIFICATION:?set Token Distillation verification}"
: "${TARGET8_FROZEN_TOKENIZER_DIR:?set frozen 148992 tokenizer directory}"
: "${TARGET8_CORRECTED_INITIAL_HF_ROOT:?set new corrected HF model directory}"

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "corrected initial HF view must run on debug" >&2; exit 2; }
[[ ! -e "$TARGET8_CORRECTED_INITIAL_HF_ROOT" ]] || { echo "refusing to overwrite corrected model root" >&2; exit 2; }
[[ -d "$(dirname "$TARGET8_CORRECTED_INITIAL_HF_ROOT")" ]] || { echo "corrected model parent missing" >&2; exit 2; }

/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/prepare_corrected_initial_hf.py" \
  --source-model-root "$TARGET8_INITIAL_HF_SOURCE" \
  --roundtrip-verification "$TARGET8_ROUNDTRIP_VERIFICATION" \
  --td-initialization-verification "$TARGET8_TD_INITIALIZATION_VERIFICATION" \
  --frozen-tokenizer-dir "$TARGET8_FROZEN_TOKENIZER_DIR" \
  --output-root "$TARGET8_CORRECTED_INITIAL_HF_ROOT"

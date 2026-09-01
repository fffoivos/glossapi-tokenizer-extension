#!/usr/bin/env bash
#SBATCH --job-name=t8b-init-gmmlu
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=01:20:00
#SBATCH --no-requeue

# Evaluate iteration zero from the corrected, hardlink-backed HF view.  The
# legacy evaluator has no code-bundle assertion of its own; keep it behind this
# targeted wrapper so its result cannot become a launch anchor from an
# unverified code directory.
set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable scientific code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable scientific code receipt}"
: "${TARGET8_INITIAL_HF:?set corrected iteration-zero HF model}"
: "${TARGET8_INITIAL_EVAL_ROOT:?set new immutable GreekMMLU output root}"

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "initial GreekMMLU must run on debug" >&2; exit 2; }
[[ ! -e "$TARGET8_INITIAL_EVAL_ROOT" ]] || { echo "refusing to overwrite initial GreekMMLU root" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific

export FULL8_CODE_ROOT="$TARGET8_CODE_ROOT"
export FULL8_CODE_BUNDLE_RECEIPT="$TARGET8_CODE_RECEIPT"
export FULL8_INITIAL_HF="$TARGET8_INITIAL_HF"
export FULL8_INITIAL_EVAL_ROOT="$TARGET8_INITIAL_EVAL_ROOT"
exec bash "$TARGET8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_initial_greekmmlu.sbatch"

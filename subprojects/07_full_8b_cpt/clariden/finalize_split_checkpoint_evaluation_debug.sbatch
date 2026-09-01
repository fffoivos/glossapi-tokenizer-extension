#!/usr/bin/env bash
#SBATCH --job-name=full8b-eval-finalize-debug
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_VALIDATION_MANIFEST:?set frozen validation manifest}"
: "${FULL8_EVAL_ITERATION:?set checkpoint iteration}"
: "${FULL8_EVAL_ATTEMPT:?set evaluation attempt}"
: "${FULL8_EVAL_ITERATION_ROOT:?set checkpoint evaluation root}"
: "${FULL8_EVAL_ATTEMPT_ROOT:?set attempt root}"
: "${FULL8_GREEKMMLU_RECEIPT:?set GreekMMLU receipt}"
: "${FULL8_DOCVAL_OUTPUT:?set per-document output root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "split evaluation finalizer must use one debug node" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency
exec /usr/bin/python3.11 \
  "$FULL8_OPS_ROOT/evaluation/finalize_split_checkpoint_evaluation.py" \
  --iteration "$FULL8_EVAL_ITERATION" \
  --attempt "$FULL8_EVAL_ATTEMPT" \
  --iteration-root "$FULL8_EVAL_ITERATION_ROOT" \
  --attempt-root "$FULL8_EVAL_ATTEMPT_ROOT" \
  --greekmmlu-receipt "$FULL8_GREEKMMLU_RECEIPT" \
  --validation-manifest "$FULL8_VALIDATION_MANIFEST" \
  --per-document-root "$FULL8_DOCVAL_OUTPUT"

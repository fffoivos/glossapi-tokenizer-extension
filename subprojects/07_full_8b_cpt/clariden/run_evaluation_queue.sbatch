#!/usr/bin/env bash
#SBATCH --job-name=full8b-eval-queue
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_STAGE_ROOT:?set frozen stage root}"
: "${FULL8_RUN_ROOT:?set run root}"
: "${FULL8_TRAIN_JOB_ID:?set completed training job id}"
: "${FULL8_EVALUATION_ITERATIONS:?set comma-separated checkpoint updates}"
: "${FULL8_EVALUATION_QUEUE_RECEIPT:?set output receipt}"
exec python3 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/run_evaluation_queue.py" \
  --code-root "$FULL8_CODE_ROOT" --stage-root "$FULL8_STAGE_ROOT" --run-root "$FULL8_RUN_ROOT" \
  --train-job-id "$FULL8_TRAIN_JOB_ID" --iterations "$FULL8_EVALUATION_ITERATIONS" \
  --max-active 4 --output "$FULL8_EVALUATION_QUEUE_RECEIPT"

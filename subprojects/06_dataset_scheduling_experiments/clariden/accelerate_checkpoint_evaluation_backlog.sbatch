#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-eval-backlog
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${SCHEDULING_BUNDLE:?set immutable scheduling-helper bundle}"
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${RUN_ROOT:?set campaign run root}"

exec "$HOST_PYTHON" \
  "$SCHEDULING_BUNDLE/production/accelerate_checkpoint_evaluation_backlog.py" \
  --campaign-manifest "$CAMPAIGN_MANIFEST" --run-root "$RUN_ROOT" \
  --max-active-debug-evaluations "${MAX_ACTIVE_DEBUG_EVALUATIONS:-1}"

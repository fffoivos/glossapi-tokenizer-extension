#!/usr/bin/env bash
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail
: "${PROBE_ROOT:?set PROBE_ROOT}"
: "${TRAINING_ASSETS_RECEIPT:?set TRAINING_ASSETS_RECEIPT}"
: "${SMOKE_ROOT:?set SMOKE_ROOT}"
: "${PHASE1_RECEIPT:?set PHASE1_RECEIPT}"
: "${PHASE2_RECEIPT:?set PHASE2_RECEIPT}"
: "${PHASE1_LOG:?set PHASE1_LOG}"
: "${PHASE1_ERR:?set PHASE1_ERR}"
: "${PHASE2_LOG:?set PHASE2_LOG}"
: "${PHASE2_ERR:?set PHASE2_ERR}"
: "${SMOKE_VERIFICATION:?set SMOKE_VERIFICATION}"

python3 "$PROBE_ROOT/train/verify_smoke.py" \
  --training-assets-receipt "$TRAINING_ASSETS_RECEIPT" \
  --phase1-receipt "$PHASE1_RECEIPT" \
  --phase2-receipt "$PHASE2_RECEIPT" \
  --phase1-output "$SMOKE_ROOT/phase1" \
  --phase2-output "$SMOKE_ROOT/phase2" \
  --phase1-log "$PHASE1_LOG" \
  --phase1-err "$PHASE1_ERR" \
  --phase2-log "$PHASE2_LOG" \
  --phase2-err "$PHASE2_ERR" \
  --output "$SMOKE_VERIFICATION"

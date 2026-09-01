#!/usr/bin/env bash
#SBATCH --job-name=mini_td_finalize
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${MINI_CODE_BUNDLE:?set immutable Mini code bundle}"
: "${MINI_FULL_TD_DIR:?set full tied-TD HF directory}"
: "${TD_PILOT_SELECTION:?set selected_pilot.json}"
: "${TD_CONVERSION_ROOT:?set conversion root}"
: "${TD_INITIALIZATION_RECEIPT:?set new final receipt path}"
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
"$HOST_PYTHON" "$MINI_CODE_BUNDLE/initialization/finalize_mini_td_initialization.py" \
  --full-td-dir "$MINI_FULL_TD_DIR" \
  --pilot-selection "$TD_PILOT_SELECTION" \
  --conversion-receipt "$TD_CONVERSION_ROOT/conversion_receipt.json" \
  --output "$TD_INITIALIZATION_RECEIPT"

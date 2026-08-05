#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-init-valid
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=01:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${OUTPUT_ROOT:?set unique output root}"
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to replace $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
preflight="$OUTPUT_ROOT/preflight.json"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/preflight_segment.py" \
  --campaign-manifest "$CAMPAIGN_MANIFEST" --segment-id 0 \
  --run-root "$OUTPUT_ROOT" --output "$preflight"
node_csv=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
master=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
env CAMPAIGN_MANIFEST="$CAMPAIGN_MANIFEST" SEGMENT_PREFLIGHT="$preflight" \
  ARM_ID=D0_mixed ARM_NODELIST="$node_csv" ARM_OUTPUT_ROOT="$OUTPUT_ROOT" \
  MASTER_ADDR="$master" MASTER_PORT=29590 INITIAL_VALIDATION_ONLY=1 \
  bash "$SCIENTIFIC_BUNDLE/clariden/run_production_arm_segment.sh" \
    >"$OUTPUT_ROOT/driver.out" 2>"$OUTPUT_ROOT/driver.err"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/finalize_initial_validation.py" \
  --campaign-manifest "$CAMPAIGN_MANIFEST" --driver-log "$OUTPUT_ROOT/driver.out" \
  --output "$OUTPUT_ROOT/initial_validation_receipt.json"

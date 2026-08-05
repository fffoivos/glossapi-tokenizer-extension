#!/usr/bin/env bash
#SBATCH --job-name=mini_mix_dupe_diag
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail
: "${SCHEDULE_BUNDLE:?set immutable scheduling bundle}"
: "${STAGE_ROOT:?set corpus stage root}"
: "${EXPECTED_GROUPS:?set validation group count}"
: "${DIAGNOSTIC_OUTPUT:?set unique diagnostic output path}"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$SCHEDULE_BUNDLE/dataset/diagnose_modern_content_duplicates.py" \
  --stage-root "$STAGE_ROOT" \
  --expected-groups "$EXPECTED_GROUPS" \
  --output "$DIAGNOSTIC_OUTPUT"

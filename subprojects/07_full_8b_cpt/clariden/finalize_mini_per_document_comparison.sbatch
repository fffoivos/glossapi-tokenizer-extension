#!/usr/bin/env bash
#SBATCH --job-name=mini-docval-finalize
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${MINI_DOCVAL_ROOT:?set per-document campaign root}"
: "${MINI_DOCVAL_MANIFEST:?set frozen per-document validation manifest}"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/analyze_per_document_endpoints.py" \
  --manifest "$MINI_DOCVAL_MANIFEST" --baseline D0_mixed \
  --arm D0_mixed="$MINI_DOCVAL_ROOT/models/D0_mixed" \
  --arm D1_hard_h_to_g="$MINI_DOCVAL_ROOT/models/D1_hard_h_to_g" \
  --arm D2_hard_g_to_h="$MINI_DOCVAL_ROOT/models/D2_hard_g_to_h" \
  --arm D3_gradual_h_to_g="$MINI_DOCVAL_ROOT/models/D3_gradual_h_to_g" \
  --arm D4_gradual_g_to_h="$MINI_DOCVAL_ROOT/models/D4_gradual_g_to_h" \
  --samples 10000 --seed 20260805 \
  --output "$MINI_DOCVAL_ROOT/per_document_endpoint_comparison.json"

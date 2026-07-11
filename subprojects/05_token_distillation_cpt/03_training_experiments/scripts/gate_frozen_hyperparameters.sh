#!/usr/bin/env bash
# Lightweight, data-independent gate for the frozen full-corpus-probe recipe.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$HERE/.." && pwd)"
COMMON_ENV="$EXP_DIR/configs/common_cpt.env"
V2="$EXP_DIR/curriculum_sweeps_v2"

# Inspect repository defaults, not accidental caller overrides.
for name in \
  ADEMA_BETA2 ADEMA_BETA3 ADEMA_ALPHA LR_PEAK LR_FINAL \
  LR_WARMUP_ITERS LR_WARMUP_TOKENS LR_WSD_DECAY_SAMPLES \
  TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE; do
  unset "$name"
done
# shellcheck disable=SC1090
source "$COMMON_ENV"

failures=0
check() {
  local name="$1" expected="$2" actual="${!1}"
  if [ "$actual" = "$expected" ]; then
    printf 'OK   %-32s %s\n' "$name" "$actual"
  else
    printf 'BAD  %-32s expected=%s got=%s\n' "$name" "$expected" "$actual" >&2
    failures=$((failures + 1))
  fi
}

check ADEMA_BETA1 0.9
check ADEMA_BETA2 0.999
check ADEMA_BETA3 0.999
check ADEMA_ALPHA 4.0
check LR_PEAK 5.5e-5
check LR_FINAL 5.5e-6
check LR_WARMUP_ITERS 400
check LR_WARMUP_TOKENS "$(( 400 * GLOBAL_BATCH_TOKENS ))"
check LR_WSD_DECAY_SAMPLES "$(( TRAIN_SAMPLES / 5 ))"
check LR_DECAY_STYLE 1-sqrt
check ADEMA_BETA3_WARMUP_STEPS "$TRAIN_ITERS"
check ADEMA_ALPHA_WARMUP_STEPS "$TRAIN_ITERS"
check SEQ_LENGTH 4096
check ROTARY_BASE 500000
check GLOBAL_BATCH_SIZE 1024
check GLOBAL_BATCH_TOKENS 4194304
check GOLDFISH_K 50
check GOLDFISH_H 50

if ! python3 "$V2/analysis/audit_sweep_configs.py" \
  --manifest "$V2/results/sweep_config_audit_20260711.json"; then
  failures=$((failures + 1))
fi

if [ "$failures" -ne 0 ]; then
  echo "FROZEN HYPERPARAMETER GATE FAILED: $failures issue(s)" >&2
  exit 1
fi

echo "FROZEN HYPERPARAMETER GATE PASSED"

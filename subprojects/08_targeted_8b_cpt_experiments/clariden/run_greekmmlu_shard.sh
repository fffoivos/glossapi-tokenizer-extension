#!/usr/bin/env bash
set -euo pipefail

for name in H2G_CODE_ROOT H2G_GREEKMMLU_CLEAN_EXAMPLES H2G_GREEKMMLU_PANEL \
  H2G_GREEKMMLU_NATIVE_RUNNER H2G_GREEKMMLU_MODEL H2G_GREEKMMLU_LABEL \
  H2G_GREEKMMLU_STAGING EVAL_VENV SLURM_PROCID; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done

exec uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/evaluation/score_frozen_greekmmlu_shard.py" \
  --clean-examples "$H2G_GREEKMMLU_CLEAN_EXAMPLES" \
  --panel "$H2G_GREEKMMLU_PANEL" \
  --native-runner "$H2G_GREEKMMLU_NATIVE_RUNNER" \
  --model "$H2G_GREEKMMLU_MODEL" \
  --model-label "$H2G_GREEKMMLU_LABEL" \
  --shard-index "$SLURM_PROCID" --shard-count 16 \
  --candidate-batch-size 1 --example-batch-size 16 \
  --output-dir "$H2G_GREEKMMLU_STAGING/shards/shard_$(printf '%03d' "$SLURM_PROCID")"

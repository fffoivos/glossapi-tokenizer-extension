#!/usr/bin/env bash
set -euo pipefail

for name in H2G_CODE_ROOT H2G_GREEKMMLU_CLEAN_EXAMPLES H2G_GREEKMMLU_PANEL \
  H2G_GREEKMMLU_NATIVE_RUNNER H2G_GREEKMMLU_MODEL H2G_GREEKMMLU_LABEL \
  H2G_GREEKMMLU_STAGING H2G_GREEKMMLU_SHARD_OFFSET EVAL_VENV SLURM_PROCID; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "$H2G_GREEKMMLU_SHARD_OFFSET" =~ ^(0|4|8|12)$ ]] || {
  echo "invalid GreekMMLU shard offset" >&2; exit 2;
}
shard_index=$((H2G_GREEKMMLU_SHARD_OFFSET + SLURM_PROCID))
[[ "$shard_index" -ge 0 && "$shard_index" -lt 16 ]] || {
  echo "GreekMMLU shard index is out of range" >&2; exit 2;
}

exec uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/evaluation/score_frozen_greekmmlu_shard.py" \
  --clean-examples "$H2G_GREEKMMLU_CLEAN_EXAMPLES" \
  --panel "$H2G_GREEKMMLU_PANEL" \
  --native-runner "$H2G_GREEKMMLU_NATIVE_RUNNER" \
  --model "$H2G_GREEKMMLU_MODEL" \
  --model-label "$H2G_GREEKMMLU_LABEL" \
  --shard-index "$shard_index" --shard-count 16 \
  --candidate-batch-size 1 --example-batch-size 16 \
  --output-dir "$H2G_GREEKMMLU_STAGING/shards/shard_$(printf '%03d' "$shard_index")"

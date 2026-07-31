#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_roundtrip
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=420G
#SBATCH --time=03:00:00
set -euo pipefail
HERE=${CPT_CLARIDEN_DIR:?submit through submit_data_pipeline.sh}
source "$HERE/paths.env"
verification="$ROUNDTRIP_ROOT/work/verification.json"
if [[ -s "$verification" ]]; then
  python3 - "$verification" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
for k in ("standard_max_abs_diff","r17_max_abs_diff","xielu_max_abs_diff","qk_norm_max_abs_diff"): assert float(x[k]) == 0.0
for k in ("orig_only","trip_only","standard_changed_over_tol","r17_changed_over_tol","shape_mismatches"): assert x[k] == []
PY
  echo "roundtrip already verified: $verification"
  exit 0
fi
test -s "$INIT_ROOT/production/config.json"
converter="$ROUNDTRIP_ROOT/megatron_converter_source"
mkdir -p "$ROUNDTRIP_ROOT"
if [[ ! -d "$converter/.git" ]]; then
  git clone --no-checkout https://github.com/swiss-ai/Megatron-LM.git "$converter"
  git -C "$converter" checkout --detach c92402e39ef3c8e69ea378a59e79059dc14541f4
fi
[[ "$(git -C "$converter" rev-parse HEAD)" == c92402e39ef3c8e69ea378a59e79059dc14541f4 ]]
export SCRIPT_DIR_OVERRIDE="$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches"
export MEGATRON_DIR_OVERRIDE="$converter"
export HF_DIR="$INIT_ROOT/production"
export RUN_ROOT="$ROUNDTRIP_ROOT"
export WORK="$ROUNDTRIP_ROOT/work"
export LOGITS=1 OVERWRITE=0
exec bash "$SCRIPT_DIR_OVERRIDE/td_layer11_r17_roundtrip.sbatch"

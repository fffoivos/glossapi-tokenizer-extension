#!/usr/bin/env bash
#SBATCH --job-name=h2g-gmmlu-calibrate
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_SCALE:?set 8b or 1p5b}"
: "${H2G_GREEKMMLU_PREDICTION_ROOT:?set root containing UPDATE/predictions.jsonl}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
case "$H2G_SCALE" in 8b|1p5b) ;; *) echo "invalid scale: $H2G_SCALE" >&2; exit 2 ;; esac
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "GreekMMLU sentinel calibration requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
manifest="$H2G_STAGE_ROOT/evaluation/greekmmlu_sentinels/sentinel_manifest.json"
output="$H2G_STAGE_ROOT/receipts/greekmmlu_sentinel_calibration_${H2G_SCALE}.json"
[[ -s "$manifest" && ! -e "$output" ]] || { echo "sentinel calibration input/output state invalid" >&2; exit 2; }
prediction_args=()
for update in 0 238 476 714 2618 2856 3094 3218; do
  path="$H2G_GREEKMMLU_PREDICTION_ROOT/$update/predictions.jsonl"
  [[ -s "$path" ]] || { echo "missing full-panel predictions: $path" >&2; exit 2; }
  prediction_args+=(--prediction "$update=$path")
done
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/scripts/verify_data_runtime.py" \
    --runtime-root "$data_runtime_root" --receipt "$data_runtime_root/runtime_receipt.json" \
    --requirements "$subproject/configs/data_runtime_requirements_v1.txt" \
    --code-root "$H2G_CODE_ROOT" --code-receipt "$H2G_CODE_RECEIPT" >/dev/null
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/evaluation/validate_greekmmlu_sentinels.py" \
    --scale "$H2G_SCALE" --manifest "$manifest" \
    "${prediction_args[@]}" --bootstrap-replicates 10000 --bootstrap-seed 20260814 \
    --expected-full-count 16159 --output "$output"

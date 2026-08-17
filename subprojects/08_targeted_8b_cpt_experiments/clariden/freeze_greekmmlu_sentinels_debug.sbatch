#!/usr/bin/env bash
#SBATCH --job-name=h2g-gmmlu-sentinels
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_GREEKMMLU_CLEAN_MANIFEST:?set frozen 16,159-question clean manifest}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "GreekMMLU sentinel freeze requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
root="$H2G_STAGE_ROOT/evaluation/greekmmlu_sentinels"
parent=$(dirname "$root")
temporary="$parent/.greekmmlu_sentinels.${SLURM_JOB_ID}.partial"
examples="$temporary/clean_examples.json"
manifest="$temporary/sentinel_manifest.json"
[[ ! -e "$root" ]] || { echo "immutable sentinel root exists: $root" >&2; exit 2; }
[[ ! -e "$temporary" ]] || { echo "partial sentinel root exists: $temporary" >&2; exit 2; }
mkdir -p "$temporary"
cleanup_partial() {
  status=$?
  if (( status != 0 )) && [[ -d "$temporary" ]]; then
    mv "$temporary" "$parent/_failed_greekmmlu_sentinels.${SLURM_JOB_ID}"
  fi
  exit "$status"
}
trap cleanup_partial EXIT
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/scripts/verify_data_runtime.py" \
    --runtime-root "$data_runtime_root" --receipt "$data_runtime_root/runtime_receipt.json" \
    --requirements "$subproject/configs/data_runtime_requirements_v1.txt" \
    --code-root "$H2G_CODE_ROOT" --code-receipt "$H2G_CODE_RECEIPT" >/dev/null
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/evaluation/freeze_greekmmlu_examples.py" \
    --clean-manifest "$H2G_GREEKMMLU_CLEAN_MANIFEST" --output "$examples"
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/evaluation/build_greekmmlu_sentinels.py" \
    --examples "$examples" --output-dir "$temporary" \
    --binding-examples "$root/clean_examples.json" --binding-output-dir "$root" \
    --sizes 4096,8192
[[ -s "$manifest" ]] || { echo "sentinel manifest missing" >&2; exit 2; }
mv "$temporary" "$root"
trap - EXIT
printf '%s\n' "$root/sentinel_manifest.json"

#!/usr/bin/env bash
#SBATCH --job-name=h2g-cache-build
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_PHASE:?set phase 1, 2 or 3}"
: "${H2G_PHASE_DATA_PATH_SPEC:?set frozen phase data-path spec}"
: "${H2G_TOKENIZER_DIR:?set frozen tokenizer directory}"
: "${H2G_PHASE_CACHE_ROOT:?set new GPTDataset cache root}"
: "${H2G_PHASE_CACHE_BUILD_RECEIPT:?set immutable build receipt}"
: "${H2G_PHASE_CACHE_FROZEN_RECEIPT:?set immutable frozen cache receipt}"
: "${H2G_MEGATRON_DIR:?set pinned patched Megatron root}"
: "${H2G_MEGATRON_RECEIPT:?set frozen compiled Megatron runtime receipt}"
: "${H2G_VALIDATION_CACHE_SEED_ROOT:?set proven cache root containing online-validation indices}"
: "${H2G_VALIDATION_DATA_ROOT:?set frozen online-validation binary root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "phase cache building requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
export H2G_CODE_ROOT H2G_CODE_RECEIPT
failed_root="${H2G_FAILED_ROOT:-$(dirname "$H2G_PHASE_CACHE_ROOT")/../_failed}"
failed_dir="$failed_root/phase-${H2G_PHASE}-cache-build-${SLURM_JOB_ID}"
quarantine_failure() {
  mkdir -p "$failed_dir"
  if [[ -e "$H2G_PHASE_CACHE_ROOT" ]]; then mv -- "$H2G_PHASE_CACHE_ROOT" "$failed_dir/cache_root"; fi
  if [[ -e "$H2G_PHASE_CACHE_BUILD_RECEIPT" ]]; then mv -- "$H2G_PHASE_CACHE_BUILD_RECEIPT" "$failed_dir/"; fi
  if [[ -e "$H2G_PHASE_CACHE_FROZEN_RECEIPT" ]]; then mv -- "$H2G_PHASE_CACHE_FROZEN_RECEIPT" "$failed_dir/"; fi
}
trap quarantine_failure ERR
uenv run pytorch/v2.9.1:v2 --view=default -- \
  env PYTHONPATH="$H2G_MEGATRON_DIR:$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts" \
  python3 "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_phase_gptdataset_cache.py" \
    --phase "$H2G_PHASE" --data-path-spec "$H2G_PHASE_DATA_PATH_SPEC" \
    --tokenizer "$H2G_TOKENIZER_DIR" --cache-root "$H2G_PHASE_CACHE_ROOT" \
    --megatron-root "$H2G_MEGATRON_DIR" --megatron-receipt "$H2G_MEGATRON_RECEIPT" \
    --validation-cache-seed-root "$H2G_VALIDATION_CACHE_SEED_ROOT" \
    --validation-data-root "$H2G_VALIDATION_DATA_ROOT" \
    --output "$H2G_PHASE_CACHE_BUILD_RECEIPT"
if [[ "$H2G_PHASE" == 1 ]]; then
  [[ -s "$H2G_PHASE_CACHE_ROOT/18800e55a969427e442ee9c9db711bfe-GPTDataset-train-description.txt" ]] || {
    echo "production Phase-1 GPTDataset cache key is missing" >&2
    exit 2
  }
fi
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_phase_blend_cache.py" \
  --phase "$H2G_PHASE" --data-path-spec "$H2G_PHASE_DATA_PATH_SPEC" \
  --cache-root "$H2G_PHASE_CACHE_ROOT" --blend-manifest "$H2G_PHASE_CACHE_BUILD_RECEIPT" \
  --output "$H2G_PHASE_CACHE_FROZEN_RECEIPT"
chmod a-w "$H2G_PHASE_CACHE_ROOT"
trap - ERR

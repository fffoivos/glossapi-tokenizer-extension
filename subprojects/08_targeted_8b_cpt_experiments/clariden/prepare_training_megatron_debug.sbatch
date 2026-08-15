#!/usr/bin/env bash
#SBATCH --job-name=h2g-megatron
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "Megatron preparation requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

commit=c92402e39ef3c8e69ea378a59e79059dc14541f4
source_root="$H2G_STAGE_ROOT/tools/megatron_source_c92402e"
output_root="${H2G_MEGATRON_OUTPUT_ROOT:-$H2G_STAGE_ROOT/tools/megatron_training_c92402e_extra_valid_helpers_v2}"
receipt="${H2G_MEGATRON_RECEIPT:-$H2G_STAGE_ROOT/receipts/training_megatron_runtime_helpers_v2.json}"
patch="$H2G_CODE_ROOT/frozen_training_tools/megatron_extra_valid_c92402e.patch"
[[ ! -e "$output_root" && ! -e "$receipt" ]] || {
  echo "immutable training Megatron output already exists" >&2; exit 2;
}
mkdir -p "$H2G_STAGE_ROOT/tools" "$H2G_STAGE_ROOT/receipts"
if [[ ! -d "$source_root/.git" ]]; then
  [[ ! -e "$source_root" ]] || { echo "non-git Megatron source path exists" >&2; exit 2; }
  git clone --no-checkout "${H2G_MEGATRON_SOURCE_URL:-https://github.com/swiss-ai/Megatron-LM.git}" "$source_root"
  git -C "$source_root" checkout --detach "$commit"
fi
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$commit" ]] || {
  echo "source Megatron commit drift" >&2; exit 2;
}
[[ -z "$(git -C "$source_root" status --porcelain)" ]] || {
  echo "source Megatron checkout is dirty" >&2; exit 2;
}
git clone --quiet --no-hardlinks "$source_root" "$output_root"
git -C "$output_root" checkout --quiet --detach "$commit"
git -C "$output_root" apply "$patch"
git -C "$output_root" diff --check
uenv run pytorch/v2.9.1:v2 --view=default -- \
  make -C "$output_root/megatron/core/datasets" -j 1
helper_count=$(find "$output_root/megatron/core/datasets" -maxdepth 1 -type f -name 'helpers_cpp*.so' | wc -l)
[[ "$helper_count" == 1 ]] || { echo "compiled Megatron dataset helper count drift: $helper_count" >&2; exit 2; }
export H2G_CODE_ROOT H2G_CODE_RECEIPT
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  python3 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_training_megatron.py" \
  --source-root "$source_root" --output-root "$output_root" --patch "$patch" --output "$receipt"
chmod -R a-w "$output_root"
printf '%s\n%s\n' "$output_root" "$receipt"

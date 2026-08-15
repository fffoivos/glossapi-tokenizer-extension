#!/usr/bin/env bash
#SBATCH --job-name=h2g-modern-mix
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_POOL:?set hplt or openarchives}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
mix_shards=${H2G_MODERN_MIX_SHARDS:-16}
case "$H2G_POOL" in
  hplt) target_tokens=8500000000 ;;
  openarchives) target_tokens=3700000000 ;;
  *) echo "invalid modern mix pool: $H2G_POOL" >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "modern mix selection requires one debug node" >&2; exit 2; }
[[ "$mix_shards" == 16 ]] || { echo "historical modern mix shard geometry drift" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
builder="$H2G_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py"
recipe="$H2G_STAGE_ROOT/configs/modern_mix/${H2G_POOL}_only.json"
recipe_receipt="$H2G_STAGE_ROOT/receipts/modern_mix_recipes.json"
tokenizer="$H2G_CODE_ROOT/frozen_historical_tokenizer_148480"
final_root="$H2G_STAGE_ROOT/data/${H2G_POOL}_mix_shards"
output="$H2G_STAGE_ROOT/data/${H2G_POOL}_selected_raw.jsonl"
manifest="$H2G_STAGE_ROOT/receipts/${H2G_POOL}_selected_raw.manifest.json"
for path in "$recipe" "$recipe_receipt" "$tokenizer/tokenizer.json"; do
  [[ -s "$path" ]] || { echo "modern mix prerequisite missing: $path" >&2; exit 2; }
done
for path in "$final_root" "$output" "$manifest"; do
  [[ ! -e "$path" ]] || { echo "immutable modern mix output exists: $path" >&2; exit 2; }
done
work_root=$(mktemp -d "$H2G_STAGE_ROOT/data/.${H2G_POOL}_mix.XXXXXX")
committed=0
cleanup() {
  if (( committed == 0 )); then
    failed_root="$H2G_STAGE_ROOT/data/_failed_${H2G_POOL}_mix_${SLURM_JOB_ID}"
    if [[ ! -e "$failed_root" ]]; then
      mkdir -p "$failed_root"
      [[ ! -e "$work_root" ]] || mv -- "$work_root" "$failed_root/work"
      [[ ! -e "$final_root" ]] || mv -- "$final_root" "$failed_root/shards"
      [[ ! -e "$output" ]] || mv -- "$output" "$failed_root/$(basename "$output")"
      [[ ! -e "$manifest" ]] || mv -- "$manifest" "$failed_root/$(basename "$manifest")"
    fi
  else
    rm -rf -- "$work_root"
  fi
}
trap cleanup EXIT
shard_root="$work_root/shards"
mkdir -p "$shard_root"
target_per_shard=$(( (target_tokens + mix_shards - 1) / mix_shards ))
pids=()
for shard in $(seq 0 $((mix_shards - 1))); do
  shard_tag=$(printf '%02d' "$shard")
  (
    uenv run pytorch/v2.9.1:v2 --view=default -- "$data_python" -u "$builder" \
      --recipe "$recipe" --target-tokens "$target_per_shard" --tokenizer "$tokenizer" \
      --output "$shard_root/shard_${shard_tag}.jsonl" --seed 20260611 \
      --source-shard-index "$shard" --source-shard-count "$mix_shards"
  ) >"$shard_root/shard_${shard_tag}.out" 2>"$shard_root/shard_${shard_tag}.err" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed != 0 )); then
  for shard in $(seq 0 $((mix_shards - 1))); do tail -n 20 "$shard_root/shard_$(printf '%02d' "$shard").err" >&2 || true; done
  exit 1
fi
temporary_output="$work_root/${H2G_POOL}_selected_raw.jsonl"
for shard in $(seq 0 $((mix_shards - 1))); do
  part="$shard_root/shard_$(printf '%02d' "$shard").jsonl"
  [[ -s "$part" ]] || { echo "modern mix shard output missing: $part" >&2; exit 2; }
  cat "$part" >> "$temporary_output"
done
mv "$shard_root" "$final_root"
mv "$temporary_output" "$output"
uenv run pytorch/v2.9.1:v2 --view=default -- "$data_python" \
  "$subproject/scripts/finalize_modern_mix.py" \
    --pool "$H2G_POOL" --recipe "$recipe" --recipe-receipt "$recipe_receipt" \
    --shard-root "$final_root" --output-jsonl "$output" --target-tokens "$target_tokens" \
    --shards "$mix_shards" --seed 20260611 --output-manifest "$manifest"
committed=1

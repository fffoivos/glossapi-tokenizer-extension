#!/usr/bin/env bash
#SBATCH --job-name=h2g-replay-mix
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
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
target_tokens=${H2G_REPLAY_TARGET_TOKENS:-5000000000}
mix_shards=${H2G_REPLAY_MIX_SHARDS:-16}

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "replay selection requires one debug node" >&2
  exit 2
}
[[ "$mix_shards" == 16 && "$target_tokens" == 5000000000 ]] || {
  echo "replay build geometry drift" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
builder="$H2G_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py"
recipe="$H2G_STAGE_ROOT/configs/replay_only_reused_validation.json"
inventory="$H2G_STAGE_ROOT/receipts/replay_source_inventory.json"
tokenizer="$H2G_CODE_ROOT/frozen_historical_tokenizer_148480"
shard_root="$H2G_STAGE_ROOT/data/replay_mix_shards"
output="$H2G_STAGE_ROOT/data/replay_selected_raw.jsonl"
manifest="$H2G_STAGE_ROOT/receipts/replay_selected_raw.manifest.json"
for path in "$recipe" "$inventory" "$tokenizer/tokenizer.json"; do
  [[ -s "$path" ]] || { echo "replay build prerequisite missing: $path" >&2; exit 2; }
done
for path in "$shard_root" "$output" "$manifest"; do
  [[ ! -e "$path" ]] || { echo "immutable replay build output exists: $path" >&2; exit 2; }
done
mkdir -p "$shard_root" "$H2G_STAGE_ROOT/receipts"
committed=0
temporary="$output.partial.$SLURM_JOB_ID"
cleanup() {
  if (( committed == 0 )); then
    failed_root="$H2G_STAGE_ROOT/data/_failed_replay_mix_${SLURM_JOB_ID}"
    if [[ ! -e "$failed_root" ]]; then
      mkdir -p "$failed_root"
      [[ ! -e "$shard_root" ]] || mv -- "$shard_root" "$failed_root/shards"
      [[ ! -e "$temporary" ]] || mv -- "$temporary" "$failed_root/$(basename "$temporary")"
      [[ ! -e "$output" ]] || mv -- "$output" "$failed_root/$(basename "$output")"
      [[ ! -e "$manifest" ]] || mv -- "$manifest" "$failed_root/$(basename "$manifest")"
    fi
  else
    rm -f -- "$temporary"
  fi
}
trap cleanup EXIT
target_per_shard=$(( (target_tokens + mix_shards - 1) / mix_shards ))
pids=()
for shard in $(seq 0 $((mix_shards - 1))); do
  shard_tag=$(printf '%02d' "$shard")
  (
    uenv run pytorch/v2.9.1:v2 --view=default -- "$data_python" -u "$builder" \
      --recipe "$recipe" \
      --target-tokens "$target_per_shard" \
      --tokenizer "$tokenizer" \
      --output "$shard_root/shard_${shard_tag}.jsonl" \
      --seed 20260611 \
      --source-shard-index "$shard" \
      --source-shard-count "$mix_shards"
  ) >"$shard_root/shard_${shard_tag}.out" 2>"$shard_root/shard_${shard_tag}.err" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed != 0 )); then
  echo "at least one replay mix shard failed" >&2
  for shard in $(seq 0 $((mix_shards - 1))); do
    tail -n 20 "$shard_root/shard_$(printf '%02d' "$shard").err" >&2 || true
  done
  exit 1
fi

[[ ! -e "$temporary" ]] || { echo "temporary replay output exists: $temporary" >&2; exit 2; }
for shard in $(seq 0 $((mix_shards - 1))); do
  part="$shard_root/shard_$(printf '%02d' "$shard").jsonl"
  [[ -s "$part" ]] || { echo "replay shard output missing: $part" >&2; exit 2; }
  cat "$part" >> "$temporary"
done
mv "$temporary" "$output"
uenv run pytorch/v2.9.1:v2 --view=default -- "$data_python" \
  "$subproject/scripts/finalize_replay_mix.py" \
    --recipe "$recipe" \
    --source-inventory "$inventory" \
    --shard-root "$shard_root" \
    --output-jsonl "$output" \
    --target-tokens "$target_tokens" \
    --shards "$mix_shards" \
    --seed 20260611 \
    --output-manifest "$manifest"
committed=1

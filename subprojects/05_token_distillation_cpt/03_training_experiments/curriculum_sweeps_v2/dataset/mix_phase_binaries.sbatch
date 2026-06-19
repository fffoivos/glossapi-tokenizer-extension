#!/usr/bin/env bash
#SBATCH --job-name=v2_mix_binaries
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.err
# array 0-2 : 0=hplt_only 1=glossapi_only 2=replay_only. Build the 3 phase binaries ONCE;
# replay% is later a Megatron blend weight, so no per-replay rebuild.
#
# mix_builder.py is a deterministic single-writer/token-counting scheduler. A
# single invocation does not use a CPU node well, so this sbatch shards each
# source stream into MIX_SHARDS disjoint row shards, runs one mix_builder per
# shard concurrently, then concatenates the shard JSONLs back to the filename
# Stage A expects. Composition is preserved because every shard uses the same
# recipe weights against disjoint source rows.
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$V2_DIR/paths.env" ] && [ -f "$V2_DIR/../paths.env" ]; then
  V2_DIR="$(cd "$V2_DIR/.." && pwd)"
fi
source "$V2_DIR/paths.env"
RECIPES_DIR="${RECIPES_DIR:-$V2/dataset/recipes}"

BINS=(hplt_only glossapi_only replay_only)
B=${BINS[${SLURM_ARRAY_TASK_ID:-0}]}
# token budgets: HPLT/OpenArchives are sized to cover the fixed-total 13.5B
# replay sweep down to R=0.15 while preserving the 70/30 phase split. replay is
# sized to cover the largest blend draw (R=0.35 of a 13.5B run = ~4.7B) plus
# headroom.
# Overridable per-binary token targets (the β3 / 27B build doubles HPLT and grows OA/replay to consumption).
declare -A TGT=( [hplt_only]="${HPLT_TGT:-8500000000}" [glossapi_only]="${GLOSSAPI_TGT:-3700000000}" [replay_only]="${REPLAY_TGT:-5000000000}" )
MIX_SHARDS="${MIX_SHARDS:-16}"
(( MIX_SHARDS >= 1 )) || { echo "ERROR: MIX_SHARDS must be >= 1" >&2; exit 2; }

export SELECTED="$SELECTED"
export PYTHONPATH="$CB:$SC/repo:${PYTHONPATH:-}"
mkdir -p "$STAGE"
# both holdout-id parquets MUST pre-exist (the recipes' drop wiring reads them)
test -f "$STAGE/val_holdout_ids.parquet"
test -f "$STAGE/forget_holdout_ids.parquet"
check_build_py
test -f "$RECIPES_DIR/$B.json"
run_build_py -u "$THIS/dataset_build/validate_selected_badness.py" --selected "$SELECTED" --require-current-report "$STAGE/selected_badness_validation.json"

SHARD_DIR="$STAGE/${B}_mix_shards"
rm -rf "$SHARD_DIR"
mkdir -p "$SHARD_DIR"
rm -f "$STAGE/${B}.jsonl" "$STAGE/${B}.manifest.json"

target_per_shard=$(( (TGT[$B] + MIX_SHARDS - 1) / MIX_SHARDS ))
echo "mix START $B target=${TGT[$B]} shards=$MIX_SHARDS target_per_shard=$target_per_shard $(date -u)"

pids=()
fail=0
for shard in $(seq 0 $((MIX_SHARDS - 1))); do
  (
    run_build_py -u "$MIX_BUILDER" \
      --recipe "$RECIPES_DIR/$B.json" \
      --target-tokens "$target_per_shard" \
      --tokenizer "$EXT_TOK" \
      --output "$SHARD_DIR/shard_$(printf '%02d' "$shard").jsonl" \
      --seed 20260611 \
      --source-shard-index "$shard" \
      --source-shard-count "$MIX_SHARDS"
  ) >"$SHARD_DIR/shard_$(printf '%02d' "$shard").out" 2>"$SHARD_DIR/shard_$(printf '%02d' "$shard").err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
if (( fail != 0 )); then
  echo "ERROR: at least one $B mix shard failed" >&2
  for shard in $(seq 0 $((MIX_SHARDS - 1))); do
    echo "--- shard_$shard.err" >&2
    tail -n 20 "$SHARD_DIR/shard_$(printf '%02d' "$shard").err" >&2 || true
  done
  exit 1
fi

: > "$STAGE/${B}.jsonl"
for shard in $(seq 0 $((MIX_SHARDS - 1))); do
  s="$SHARD_DIR/shard_$(printf '%02d' "$shard").jsonl"
  test -s "$s"
  cat "$s" >> "$STAGE/${B}.jsonl"
done
run_build_py - <<PY
import json, pathlib
stage = pathlib.Path("$STAGE")
shard_dir = pathlib.Path("$SHARD_DIR")
out = stage / "$B.jsonl"
manifests = sorted(shard_dir.glob("shard_*.manifest.json"))
parts = [json.loads(p.read_text()) for p in manifests]
summary = {
    "binary": "$B",
    "target_tokens": int("${TGT[$B]}"),
    "mix_shards": int("$MIX_SHARDS"),
    "target_per_shard": int("$target_per_shard"),
    "actual_tokens": sum(int(p.get("actual_tokens", 0)) for p in parts),
    "actual_rows": sum(int(p.get("actual_rows", 0)) for p in parts),
    "output": str(out),
    "shards": [str(p) for p in manifests],
}
(stage / "$B.manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
PY
echo "mix DONE $B $(date -u)"
ls -lah "$STAGE/${B}.jsonl" "$STAGE/${B}.manifest.json"

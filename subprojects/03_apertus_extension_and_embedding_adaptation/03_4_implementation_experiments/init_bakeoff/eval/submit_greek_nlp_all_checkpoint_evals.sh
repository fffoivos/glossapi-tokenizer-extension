#!/usr/bin/env bash
# Submit packed greek-nlp/benchmark evals for Apertus base plus all release checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
fi
MANIFEST="${MANIFEST:-$REPO_ROOT/release/apertus-tokenizer-extension/manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval/native_greek_suite_20260526/greek_nlp_all_checkpoints_s100}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
TASK="${TASK:-all}"

mapfile -t specs < <(
  {
    echo "Apertus-Base=/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509"
    jq -r '.checkpoints[] | .human_name + "=" + .source_hf_copy_clariden' "$MANIFEST"
  }
)

if [ "${#specs[@]}" -eq 0 ]; then
  echo "ERROR: no checkpoint specs found" >&2
  exit 2
fi

echo "Submitting ${#specs[@]} greek-nlp checkpoint evals in chunks of $CHUNK_SIZE"

job_ids=()
for ((start=0; start<${#specs[@]}; start+=CHUNK_SIZE)); do
  chunk=("${specs[@]:start:CHUNK_SIZE}")
  model_specs="$(IFS=';'; echo "${chunk[*]}")"
  chunk_id=$((start / CHUNK_SIZE + 1))
  echo "chunk $chunk_id: $model_specs"
  job_id="$(
    MODEL_SPECS="$model_specs" \
    OUTPUT_DIR="$OUTPUT_ROOT/chunk_${chunk_id}" \
    SAMPLE_SIZE="$SAMPLE_SIZE" \
    TASK="$TASK" \
    SCRIPT_DIR_OVERRIDE="$REMOTE_SCRIPT_DIR" \
      sbatch --parsable "$REMOTE_SCRIPT_DIR/run_greek_nlp_benchmark_hf_packed.sbatch"
  )"
  job_ids+=("$job_id")
  echo "  -> $job_id"
done

echo "Submitted: ${job_ids[*]}"

#!/usr/bin/env bash
#SBATCH --job-name=v2_A_clean_decontam
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.err
# array 0-2. Per-binary StageA: HPLT E001 confident-only clean on the 2 Greek binaries +
# GreekMMLU-only correct_only decontam on ALL THREE. decontaminate.py is per-doc streaming,
# so running it per binary is identical to the pilot's single-stream pass.
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$V2_DIR/paths.env" ] && [ -f "$V2_DIR/../paths.env" ]; then
  V2_DIR="$(cd "$V2_DIR/.." && pwd)"
fi
source "$V2_DIR/paths.env"

BINS=(hplt_only glossapi_only replay_only)
B=${BINS[${SLURM_ARRAY_TASK_ID:-0}]}
test -s "$STAGE/$B.jsonl"; test -s "$QUERIES"; check_build_py

IN="$STAGE/$B.jsonl"
# HPLT confident-only clean only applies to the Greek streams; replay rides decontam+anon only.
if [ "$B" = hplt_only ] || [ "$B" = glossapi_only ]; then
  run_build_py -u "$HPLT_CLEAN" --input "$IN" --output "$STAGE/${B}_clean.jsonl" --workers 64
  IN="$STAGE/${B}_clean.jsonl"
fi

# GreekMMLU-only decontamination (the one Greek benchmark we trust). Drops contaminated docs.
run_build_py -u "$DECONTAM" \
  --queries-jsonl "$QUERIES" \
  --corpus-jsonl "$IN" \
  --output-dir "$STAGE/${B}_decontam_output" \
  --benchmark greekmmlu --k 8 --max-gap-tokens 50 --max-gap-tokens-short 5 --direction after \
  --primary-rule correct_only --n-workers 64 --mp-chunksize 1024 \
  --filter-output-clean "$STAGE/${B}_decontam.jsonl" \
  --filter-output-dropped "$STAGE/${B}_dropped.jsonl"
echo "stageA DONE $B $(date -u)"
ls -lah "$STAGE/${B}_decontam.jsonl" "$STAGE/${B}_dropped.jsonl" 2>&1 | head

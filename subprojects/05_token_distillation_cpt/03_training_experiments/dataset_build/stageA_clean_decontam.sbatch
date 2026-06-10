#!/usr/bin/env bash
#SBATCH --job-name=cpt13b_A_clean_decontam
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.err
# concat mix shards -> HPLT confident-only clean (E001) -> decontaminate (correct_only).
set -euo pipefail
SC=/iopsstor/scratch/cscs/fffoivos
SUB=$SC/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt
DB=$SUB/03_training_experiments/dataset_build
STAGE=$SC/cpt_corpus/cpt_2arm_13b
QUERIES=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/decontamination/05_decontam_5b_20260602T011447Z/queries/native_greek_mcq_decontam_queries.jsonl
PY=${XFER_PY:-/users/fffoivos/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12}
test -s "$QUERIES"
test -x "$PY"
echo "concat $(ls $STAGE/bulk_mix_part_*.jsonl | wc -l) shards -> bulk_mix.jsonl"
cat "$STAGE"/bulk_mix_part_*.jsonl > "$STAGE/bulk_mix.jsonl"
wc -l "$STAGE/bulk_mix.jsonl"
rm -f "$STAGE/bulk_mix_clean.jsonl" "$STAGE/bulk_mix_decontam.jsonl" "$STAGE/bulk_mix_dropped.jsonl"
rm -rf "$STAGE/decontam_output"
echo "=== HPLT confident-only clean (E001) ==="
"$PY" -u "$DB/hplt_clean.py" --input "$STAGE/bulk_mix.jsonl" --output "$STAGE/bulk_mix_clean.jsonl" --workers 64
echo "=== decontaminate (greekmmlu, correct_only) ==="
"$PY" -u "$SUB/02_corpus_preparation/30_decontaminate/scripts/decontaminate.py" \
  --queries-jsonl "$QUERIES" \
  --corpus-jsonl "$STAGE/bulk_mix_clean.jsonl" \
  --output-dir "$STAGE/decontam_output" \
  --benchmark greekmmlu --k 8 --max-gap-tokens 50 --max-gap-tokens-short 5 --direction after \
  --primary-rule correct_only --n-workers 64 --mp-chunksize 1024 \
  --filter-output-clean "$STAGE/bulk_mix_decontam.jsonl" \
  --filter-output-dropped "$STAGE/bulk_mix_dropped.jsonl"
echo "stageA DONE $(date -u): $(wc -l < $STAGE/bulk_mix_decontam.jsonl) clean docs"

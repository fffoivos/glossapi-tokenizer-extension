#!/usr/bin/env bash
#SBATCH --job-name=newtok_corpus
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:40:00
# AGENTS.md rule 4: dataset tokenisation runs on `debug`, never on the Mac,
# a login node, or `normal`.
set -euo pipefail
ROOT=/iopsstor/scratch/cscs/$USER/newtok_audit_20260823
VENV=/iopsstor/scratch/cscs/$USER/python_envs/h2g_greekmmlu_eval_runtime_20260817_v2
TOKENIZER=/iopsstor/scratch/cscs/$USER/tokenizers/apertus_greek_modern_polytonic_148992
HELDOUT=/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/cpt25b_midtraining/20260731T124000Z-cpt25b-v1/heldouts/sets
export TOKENIZERS_PARALLELISM=true
uenv run pytorch/v2.9.1:v2 --view=default -- "$VENV/bin/python" \
  "$ROOT/scripts/build_coverage_corpus.py" \
  --data-root "$HELDOUT" \
  --tokenizer "$TOKENIZER" \
  --plan "$ROOT/scripts/corpus_plan.json" \
  --out-jsonl "$ROOT/out/coverage_corpus.jsonl" \
  --out-coverage "$ROOT/out/coverage.json" \
  --new-id-range 131072 148992 \
  --floor 16 --max-keep 40000

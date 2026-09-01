#!/usr/bin/env bash
#SBATCH --job-name=newtok_audit
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT=/iopsstor/scratch/cscs/$USER/newtok_audit_20260823
VENV=/iopsstor/scratch/cscs/$USER/python_envs/h2g_greekmmlu_eval_runtime_20260817_v2
CKPT_ROOT=/capstor/scratch/cscs/$USER/checkpoint_release_20260818/raw_staging_v1
CORPUS=$ROOT/out/coverage_corpus.jsonl
MAXDOCS=${MAXDOCS:-40000}
[[ -s "$CORPUS" ]] || { echo "coverage corpus missing; run stageA first"; exit 2; }
wc -l "$CORPUS"

run_one () {
  CUDA_VISIBLE_DEVICES=$2 uenv run pytorch/v2.9.1:v2 --view=default -- \
    "$VENV/bin/python" "$ROOT/scripts/new_token_behavioral_audit.py" \
    --model "$CKPT_ROOT/$1" \
    --base-tokenizer "$ROOT/base_tok" \
    --corpus "$CORPUS" \
    --out "$ROOT/out/audit_$1.json" \
    --new-id-range 131072 148992 --layer 11 --late-layer 30 \
    --max-docs "$MAXDOCS" --max-context 1024 --max-split-tokens 3072 --batch 8 \
    > "$ROOT/logs/audit_$1.log" 2>&1
}

run_one step400-tokens2B 0 &
run_one step9536-tokens40B 1 &
run_one main 2 &
wait
echo "ALL DONE"
ls -la "$ROOT/out"

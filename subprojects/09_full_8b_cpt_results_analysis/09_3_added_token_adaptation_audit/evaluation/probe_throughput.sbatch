#!/usr/bin/env bash
#SBATCH --job-name=newtok_probe
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=00:35:00
# Runtime-only probe: measures docs/s in the exact production shape
# (3 concurrent single-GPU audits on one node) at two batch sizes.
# Science is held fixed: same checkpoints, same corpus, same layers, same precision.
set -euo pipefail
ROOT=/iopsstor/scratch/cscs/$USER/newtok_audit_20260823
VENV=/iopsstor/scratch/cscs/$USER/python_envs/h2g_greekmmlu_eval_runtime_20260817_v2
CKPT_ROOT=/capstor/scratch/cscs/$USER/checkpoint_release_20260818/raw_staging_v1
CORPUS=$ROOT/out/coverage_corpus.jsonl
NDOCS=${NDOCS:-150}

probe () {  # $1 ckpt  $2 gpu  $3 batch  $4 phase
  CUDA_VISIBLE_DEVICES=$2 uenv run pytorch/v2.9.1:v2 --view=default -- \
    "$VENV/bin/python" "$ROOT/scripts/new_token_behavioral_audit.py" \
    --model "$CKPT_ROOT/$1" --base-tokenizer "$ROOT/base_tok" --corpus "$CORPUS" \
    --out "/dev/null" --new-id-range 131072 148992 --layer 11 --late-layer 30 \
    --max-docs "$NDOCS" --max-context 1024 --max-split-tokens 3072 \
    --batch "$3" --skip-echo --progress-every 50 \
    > "$ROOT/logs/probe_p$4_b$3_$1.log" 2>&1
}

for B in 8 24; do
  echo "=== phase batch=$B start $(date -u +%H:%M:%S)"
  probe step400-tokens2B     0 "$B" "$B" &
  probe step9536-tokens40B   1 "$B" "$B" &
  probe main                 2 "$B" "$B" &
  wait
  echo "=== phase batch=$B end   $(date -u +%H:%M:%S)"
done
echo "PROBE DONE"
grep -H "T1/T2" "$ROOT"/logs/probe_p*_b*.log | tail -30

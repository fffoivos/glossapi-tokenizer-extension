#!/usr/bin/env bash
#SBATCH --job-name=bib_g1_prepare
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

LAUNCH_ROOT=${LAUNCH_ROOT:?set launch root}
REPO_ROOT=${REPO_ROOT:?set clean deployed repository root}
G0_RECEIPT=${G0_RECEIPT:?set accepted G0 receipt}
PREP_SCRIPT=${PREP_SCRIPT:?set immutable G1 preparation script}
UENV=${UENV:-pytorch/v2.9.1:v2}
PYTHON_DEPS=${PYTHON_DEPS:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/python_deps/sklearn-1.9.0-py312}
EVAL_ROOT="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"

[[ -f "$PREP_SCRIPT" && ! -L "$PREP_SCRIPT" ]] || exit 90
[[ -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || exit 91
[[ -f "$G0_RECEIPT" && ! -L "$G0_RECEIPT" ]] || exit 92
[[ -d "$PYTHON_DEPS" && ! -L "$PYTHON_DEPS" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$EVAL_ROOT:$PYTHON_DEPS" python3 "$PREP_SCRIPT" \
  --launch-root "$LAUNCH_ROOT" \
  --repo-root "$REPO_ROOT" \
  --g0-receipt "$G0_RECEIPT"

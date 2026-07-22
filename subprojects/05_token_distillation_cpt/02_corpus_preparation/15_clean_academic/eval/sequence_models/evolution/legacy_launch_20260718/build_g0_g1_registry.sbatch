#!/usr/bin/env bash
#SBATCH --job-name=bib_dev_registry
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

REPO_ROOT=${REPO_ROOT:?set clean deployed repository root}
CANDIDATE_ROOT=${CANDIDATE_ROOT:?set finalized candidate root}
REGISTRY_ROOT=${REGISTRY_ROOT:?set immutable registry parent}
REGISTRY_SCRIPT=${REGISTRY_SCRIPT:?set immutable registry script}
UENV=${UENV:-pytorch/v2.9.1:v2}
PYTHON_DEPS=${PYTHON_DEPS:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/python_deps/sklearn-1.9.0-py312}
EVAL_ROOT="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"

[[ -f "$REGISTRY_SCRIPT" && ! -L "$REGISTRY_SCRIPT" ]] || exit 90
[[ -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || exit 91
[[ -d "$CANDIDATE_ROOT" && ! -L "$CANDIDATE_ROOT" ]] || exit 92
[[ -d "$PYTHON_DEPS" && ! -L "$PYTHON_DEPS" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$EVAL_ROOT:$PYTHON_DEPS" python3 "$REGISTRY_SCRIPT" \
  --repo-root "$REPO_ROOT" \
  --candidate-root "$CANDIDATE_ROOT" \
  --registry-root "$REGISTRY_ROOT"

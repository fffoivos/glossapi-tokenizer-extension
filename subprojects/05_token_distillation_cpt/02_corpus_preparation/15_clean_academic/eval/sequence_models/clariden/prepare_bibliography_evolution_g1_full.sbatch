#!/usr/bin/env bash
#SBATCH --job-name=bib_g1_full_prepare
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
CODE_COMMIT=${CODE_COMMIT:?set exact deployed repository commit}
G0_RECEIPT=${G0_RECEIPT:?set accepted G0 receipt}
PILOT_PACKET_RECEIPT=${PILOT_PACKET_RECEIPT:?set five-row pilot packet receipt}
PILOT_QUEUE=${PILOT_QUEUE:?set five-row pilot queue}
PILOT_LAUNCH_RECEIPT=${PILOT_LAUNCH_RECEIPT:?set five-row pilot launch receipt}
UENV=${UENV:-pytorch/v2.9.1:v2}
PYTHON_DEPS=${PYTHON_DEPS:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/python_deps/sklearn-1.9.0-py312}
EVAL_ROOT="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$CODE_COMMIT" ]] || exit 89
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || exit 88
[[ -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || exit 91
for path in "$G0_RECEIPT" "$PILOT_PACKET_RECEIPT" "$PILOT_QUEUE" "$PILOT_LAUNCH_RECEIPT"; do
  [[ -f "$path" && ! -L "$path" ]] || exit 92
done
[[ -d "$PYTHON_DEPS" && ! -L "$PYTHON_DEPS" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$EVAL_ROOT:$PYTHON_DEPS" python3 -m \
  sequence_models.bibliography_evolution_prepare_g1_full \
  --launch-root "$LAUNCH_ROOT" \
  --repo-root "$REPO_ROOT" \
  --code-commit "$CODE_COMMIT" \
  --g0-receipt "$G0_RECEIPT" \
  --pilot-packet-receipt "$PILOT_PACKET_RECEIPT" \
  --pilot-queue "$PILOT_QUEUE" \
  --pilot-launch-receipt "$PILOT_LAUNCH_RECEIPT"

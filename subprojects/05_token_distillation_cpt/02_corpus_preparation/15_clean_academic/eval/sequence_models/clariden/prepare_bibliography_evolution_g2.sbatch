#!/usr/bin/env bash
#SBATCH --job-name=bib_g2_det_prepare
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
VALIDATION_SOURCE=${VALIDATION_SOURCE:?set frozen validation source}
VALIDATION_TABLE_DIR=${VALIDATION_TABLE_DIR:?set frozen validation table}
REFERENCE_NEGATIVE_ROLES=${REFERENCE_NEGATIVE_ROLES:?set frozen validation roles}
REFERENCE_NEGATIVE_RECEIPT=${REFERENCE_NEGATIVE_RECEIPT:?set frozen validation-role receipt}
PARENT_RECEIPT=${PARENT_RECEIPT:?set admitted G1 parent receipt}
DEVELOPMENT_REGISTRY=${DEVELOPMENT_REGISTRY:?set frozen G0/G1 registry}
REGISTRY_RECEIPT=${REGISTRY_RECEIPT:?set frozen registry receipt}
PARENT_RECOMMENDATION=${PARENT_RECOMMENDATION:?set frozen G2 recommendation}
PRIOR_PREP_JOB_RECORDS=${PRIOR_PREP_JOB_RECORDS:?set semicolon-separated JOB_ID|STAGE|REASON records}
UENV=${UENV:-pytorch/v2.9.1:v2}
PYTHON_DEPS=${PYTHON_DEPS:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/python_deps/sklearn-1.9.0-py312}
EVAL_ROOT="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
ROLE_DIR="$LAUNCH_ROOT/derived_inputs/deterministic-header-roles-${CODE_COMMIT:0:8}-${SLURM_JOB_ID}"

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$CODE_COMMIT" ]] || exit 89
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || exit 88

for path in \
  "$VALIDATION_SOURCE" \
  "$REFERENCE_NEGATIVE_ROLES" \
  "$REFERENCE_NEGATIVE_RECEIPT" \
  "$PARENT_RECEIPT" \
  "$DEVELOPMENT_REGISTRY" \
  "$REGISTRY_RECEIPT" \
  "$PARENT_RECOMMENDATION"; do
  [[ -f "$path" && ! -L "$path" ]] || exit 90
done
for path in "$REPO_ROOT/.git" "$VALIDATION_TABLE_DIR" "$PYTHON_DEPS"; do
  [[ -d "$path" && ! -L "$path" ]] || exit 91
done

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

IFS=';' read -r -a prior_records <<< "$PRIOR_PREP_JOB_RECORDS"
prior_args=()
for record in "${prior_records[@]}"; do
  [[ -n "$record" ]] || exit 87
  prior_args+=(--prior-prep-job "$record")
done

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$EVAL_ROOT:$PYTHON_DEPS" python3 -m \
  sequence_models.bibliography_evolution_header_role_generation \
  --repo-root "$REPO_ROOT" \
  --code-commit "$CODE_COMMIT" \
  --validation-source "$VALIDATION_SOURCE" \
  --validation-table-dir "$VALIDATION_TABLE_DIR" \
  --reference-negative-roles "$REFERENCE_NEGATIVE_ROLES" \
  --reference-negative-receipt "$REFERENCE_NEGATIVE_RECEIPT" \
  --output-dir "$ROLE_DIR" \
  --workers "$SLURM_CPUS_PER_TASK"

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$EVAL_ROOT:$PYTHON_DEPS" python3 -m \
  sequence_models.bibliography_evolution_prepare_g2 \
  --launch-root "$LAUNCH_ROOT" \
  --repo-root "$REPO_ROOT" \
  --code-commit "$CODE_COMMIT" \
  --parent-receipt "$PARENT_RECEIPT" \
  --header-role-dir "$ROLE_DIR" \
  --development-registry "$DEVELOPMENT_REGISTRY" \
  --registry-receipt "$REGISTRY_RECEIPT" \
  --parent-recommendation "$PARENT_RECOMMENDATION" \
  "${prior_args[@]}"

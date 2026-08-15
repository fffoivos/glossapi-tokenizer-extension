#!/usr/bin/env bash
#SBATCH --job-name=t8b-decontam-acad
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=350G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
QUERY_ROOT=${TARGET8_GREEKMMLU_QUERY_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/cpt25b_midtraining/20260731T124000Z-cpt25b-v1/greekmmlu}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "academic decontamination must run on debug" >&2; exit 2; }

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
input="$TARGET8_STAGE_ROOT/experiment_a/academic_pre_decontamination/data"
root="$TARGET8_STAGE_ROOT/experiment_a/academic_decontaminated"
[[ -s "$TARGET8_STAGE_ROOT/experiment_a/academic_pre_decontamination/extraction_manifest.json" ]] || {
  echo "academic extraction receipt missing" >&2; exit 2;
}
[[ ! -e "$root" ]] || { echo "immutable academic decontamination output exists" >&2; exit 2; }

uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/decontaminate_targeted_corpus.py" \
  --canonical-script "$TARGET8_CODE_ROOT/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/scripts/decontaminate_full_corpus.py" \
  --input "$input" --output "$root/data" --dropped "$root/dropped" \
  --ledger "$root/ledger" --manifest "$root/decontamination_manifest.json" \
  --queries-jsonl "$QUERY_ROOT/greekmmlu.jsonl" \
  --benchmark-manifest "$QUERY_ROOT/greekmmlu.jsonl.manifest.json" \
  --workers 32

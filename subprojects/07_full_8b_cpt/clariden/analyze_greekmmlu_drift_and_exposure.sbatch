#!/bin/bash
#SBATCH --job-name=full8-evidence
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/analysis/20260811_greekmmlu_drift_exposure/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/analysis/20260811_greekmmlu_drift_exposure/slurm-%j.err

set -euo pipefail

readonly CODE_ROOT="${CODE_ROOT:?CODE_ROOT is required}"
readonly OUTPUT_ROOT="/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/analysis/20260811_greekmmlu_drift_exposure"
readonly STAGE_ROOT="/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260808T064500Z-d0-v4-v45bridge"
readonly RUN_ROOT="/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12"
readonly INITIAL_ROOT="/iopsstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/_preflight/20260808T023300Z-sanitized-v45/initial_greekmmlu"
readonly CLEAN_IDS="/capstor/scratch/cscs/fffoivos/cpt_runs/dataset-scheduling-0p5b/20260803T064000Z-static-prelaunch-v2/greekmmlu_clean_example_ids.txt"
readonly PYTHON="/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python"

mkdir -p "${OUTPUT_ROOT}/work"

srun --ntasks=1 --cpus-per-task=72 --cpu-bind=none \
  uenv run pytorch/v2.9.1:v2 --view=default -- \
  "${PYTHON}" "${CODE_ROOT}/analyze_greekmmlu_answer_drift.py" \
    --run-root "${RUN_ROOT}" \
    --initial-predictions "${INITIAL_ROOT}/greekmmlu/full8b_mixed_iter0_native_mcq_predictions.jsonl" \
    --initial-receipt "${INITIAL_ROOT}/initial_greekmmlu_receipt.json" \
    --clean-ids "${CLEAN_IDS}" \
    --output "${OUTPUT_ROOT}/greekmmlu_answer_drift.json"

srun --ntasks=1 --cpus-per-task=72 --cpu-bind=none \
  uenv run pytorch/v2.9.1:v2 --view=default -- \
  "${PYTHON}" "${CODE_ROOT}/analyze_checkpoint_source_exposure.py" \
    --stage-root "${STAGE_ROOT}" \
    --work-root "${OUTPUT_ROOT}/work" \
    --workers 32 \
    --output "${OUTPUT_ROOT}/checkpoint_source_exposure.json"

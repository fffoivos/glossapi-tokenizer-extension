#!/usr/bin/env bash
#SBATCH --job-name=bibclean-v2-dryrun
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=03:00:00
#SBATCH --array=0-2

set -euo pipefail
: "${RUN_BOOTSTRAP:?RUN_BOOTSTRAP is required}"
: "${CONTRACT:?CONTRACT is required}"
: "${PLAN:?PLAN is required}"
export TRAIN_SRC="$RUN_BOOTSTRAP/train"
export GLOSSAPI_SRC="$RUN_BOOTSTRAP/glossapi/src"
export PYTHONPATH="$RUN_BOOTSTRAP/runtime:$TRAIN_SRC:$GLOSSAPI_SRC${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="$HOME/.venvs/hfupload/bin/python"
export NODE_COUNT=3 SLOTS=36 THREADS=8
exec bash "$TRAIN_SRC/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/production/run_units.sh"

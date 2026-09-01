#!/usr/bin/env bash
#SBATCH --job-name=hf-v2-anon-final
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:25:00
set -euo pipefail
: "${HFV2_CODE_ROOT:?}"
: "${HFV2_RUN_ROOT:?}"
: "${HFV2_DATA_PYTHON:?}"
pipeline="$HFV2_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/hf_v2_release/scripts/pipeline.py"
uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$HFV2_DATA_PYTHON" "$pipeline" finalize \
  --code-root "$HFV2_CODE_ROOT" --contract "$HFV2_RUN_ROOT/run_contract.json"

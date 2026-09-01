#!/usr/bin/env bash
#SBATCH --job-name=hf-v2-anon-prepare
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
set -euo pipefail
: "${HFV2_CODE_ROOT:?}"
: "${HFV2_RUN_ROOT:?}"
: "${HFV2_INPUT_ROOT:?}"
: "${HFV2_DATA_PYTHON:?}"
: "${HFV2_TOKENIZER_JSON:?}"
: "${HFV2_DEDUP_RECEIPT:?}"
: "${HFV2_SOURCE_BREAKDOWN:?}"
config="$HFV2_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/hf_v2_release/configs/release.json"
pipeline="$HFV2_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/hf_v2_release/scripts/pipeline.py"
uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$HFV2_DATA_PYTHON" "$pipeline" prepare \
  --code-root "$HFV2_CODE_ROOT" --config "$config" --input-root "$HFV2_INPUT_ROOT" \
  --tokenizer-json "$HFV2_TOKENIZER_JSON" --dedup-receipt "$HFV2_DEDUP_RECEIPT" \
  --source-breakdown "$HFV2_SOURCE_BREAKDOWN" --run-root "$HFV2_RUN_ROOT"

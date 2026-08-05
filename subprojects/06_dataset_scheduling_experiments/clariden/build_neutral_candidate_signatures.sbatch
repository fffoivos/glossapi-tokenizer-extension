#!/usr/bin/env bash
#SBATCH --job-name=neutral_sigs
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=04:00:00
set -euo pipefail

: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${NEUTRAL_RUN_ROOT:?set neutral pipeline root}"
: "${AGENT1_RUN_ROOT:?set frozen Agent-1 v5 run root}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.6.0:v1}
DATATROVE_RUNTIME_PYTHON=${DATATROVE_RUNTIME_PYTHON:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/.agent1-v5-clariden-debug-20260715T111552Z-30c72e9.coord/runtime/venv/bin/python}
output="$NEUTRAL_RUN_ROOT/candidate-signatures"
[[ ! -e "$output" ]] || { echo "refusing to replace $output" >&2; exit 2; }
uenv run "$UENV_IMAGE" --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$DATATROVE_RUNTIME_PYTHON" \
  "$SCIENTIFIC_BUNDLE/evaluation/cross_deduplicate_neutral_external.py" candidate-signatures \
  --source-preparation-receipt "$NEUTRAL_RUN_ROOT/source/source_preparation_receipt.json" \
  --training-config "$AGENT1_RUN_ROOT/run_contract.json" \
  --training-signature-manifest "$AGENT1_RUN_ROOT/signature_manifest.json" \
  --output-root "$output"

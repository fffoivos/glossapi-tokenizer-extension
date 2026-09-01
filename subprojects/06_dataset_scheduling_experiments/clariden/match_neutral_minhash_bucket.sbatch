#!/usr/bin/env bash
#SBATCH --job-name=neutral_match
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail

: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${NEUTRAL_RUN_ROOT:?set neutral pipeline root}"
: "${AGENT1_RUN_ROOT:?set frozen Agent-1 v5 run root}"
: "${SLURM_ARRAY_TASK_ID:?run as the 0-31 bucket array}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.6.0:v1}
DATATROVE_RUNTIME_PYTHON=${DATATROVE_RUNTIME_PYTHON:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/.agent1-v5-clariden-debug-20260715T111552Z-30c72e9.coord/runtime/venv/bin/python}
bucket=$(printf '%03d' "$SLURM_ARRAY_TASK_ID")
mkdir -p "$NEUTRAL_RUN_ROOT/matches"
output="$NEUTRAL_RUN_ROOT/matches/bucket_${bucket}.jsonl.gz"
uenv run "$UENV_IMAGE" --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$DATATROVE_RUNTIME_PYTHON" \
  "$SCIENTIFIC_BUNDLE/evaluation/cross_deduplicate_neutral_external.py" match-bucket \
  --candidate-signatures-receipt "$NEUTRAL_RUN_ROOT/candidate-signatures/candidate_signatures_receipt.json" \
  --training-signature-manifest "$AGENT1_RUN_ROOT/signature_manifest.json" \
  --training-signature-root "$AGENT1_RUN_ROOT/60-dedup/minhash-signatures" \
  --bucket "$SLURM_ARRAY_TASK_ID" --output "$output"

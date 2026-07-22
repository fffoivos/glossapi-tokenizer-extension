#!/usr/bin/env bash
# Build the 30-document sealed-test bibliography failure-analysis reader.
#SBATCH --job-name=bib_ng_worst_site
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 022

RUNNER=${RUNNER:?set standalone builder path}
CODE_ROOT=${CODE_ROOT:?set frozen bundle parent containing sequence_models}
DOCUMENTS=${DOCUMENTS:?set sealed documents JSONL}
LINE_KEY=${LINE_KEY:?set sealed line-key JSONL}
LABELS=${LABELS:?set sealed task-consensus labels JSONL}
FEATURE_DIR=${FEATURE_DIR:?set frozen unseen feature directory}
PREDICTION=${PREDICTION:?set frozen candidate prediction array}
PROBABILITY=${PROBABILITY:?set frozen candidate probability array}
ANALYSIS=${ANALYSIS:?set reviewed document analysis JSON}
OUTPUT_DIR=${OUTPUT_DIR:?set new output site directory}
UENV=${UENV:-pytorch/v2.6.0:v1}

[[ -f "$RUNNER" && -f "$DOCUMENTS" && -f "$LINE_KEY" && -f "$LABELS" ]] || exit 90
[[ -d "$FEATURE_DIR" && -f "$PREDICTION" && -f "$PROBABILITY" ]] || exit 91
[[ -f "$ANALYSIS" && ! -e "$OUTPUT_DIR" ]] || exit 92

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 "$RUNNER" \
  --documents "$DOCUMENTS" \
  --line-key "$LINE_KEY" \
  --labels "$LABELS" \
  --feature-dir "$FEATURE_DIR" \
  --prediction "$PREDICTION" \
  --probability "$PROBABILITY" \
  --analysis "$ANALYSIS" \
  --output-dir "$OUTPUT_DIR" \
  --per-source 10

echo "BIBLIOGRAPHY_NEXTGEN_WORST_SITE=$OUTPUT_DIR"

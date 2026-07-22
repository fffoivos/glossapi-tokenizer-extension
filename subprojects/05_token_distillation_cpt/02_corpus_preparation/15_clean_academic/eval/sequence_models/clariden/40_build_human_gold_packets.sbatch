#!/usr/bin/env bash
#SBATCH --job-name=tocbib_gold_packets
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

SEQUENCE_CLARIDEN_DIR=${SEQUENCE_CLARIDEN_DIR:?submit through submit_sequence.sh}
SEQUENCE_DIR=$(cd "$SEQUENCE_CLARIDEN_DIR/.." && pwd)
ACADEMIC_DIR=$(cd "$SEQUENCE_DIR/../.." && pwd)
PHASE04_DIR=$(cd "$ACADEMIC_DIR/../../04_full_corpus_preparation" && pwd)
source "$PHASE04_DIR/clariden/paths.env"
source "$PHASE04_DIR/clariden/common.sh"
phase04_require_compute_partition "human-gold packet creation"
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
[[ "${SLURM_GPUS:-0}" == "0" ]] || { echo "ERROR: packet creation is CPU-only" >&2; exit 93; }

INPUT_RECEIPT=${INPUT_RECEIPT:?set the passed Phase-04 acquisition receipt}
GREEK_PHD_INPUT=${GREEK_PHD_INPUT:?set the receipt-bound canonical Greek-PhD Parquet path/glob}
OPENARCHIVES_INPUT=${OPENARCHIVES_INPUT:?set the receipt-bound canonical OpenArchives Parquet path/glob}
KALLIPOS_INPUT=${KALLIPOS_INPUT:?set the receipt-bound canonical Kallipos Parquet path}
PERGAMOS_INPUT=${PERGAMOS_INPUT:?set the receipt-bound canonical Pergamos Parquet path}
RUN_TAG=${RUN_TAG:-human_gold_packets_$(date -u +%Y%m%dT%H%M%SZ)}
PACKET_ROOT=${PACKET_ROOT:-$RUN_ROOT/$RUN_TAG}
STATE=${STATE:-$DATA_ROOT/human_gold_state/$RUN_TAG.sqlite}

phase04_require_file "$INPUT_RECEIPT"
mkdir -p "$(dirname "$PACKET_ROOT")" "$(dirname "$STATE")"
export PYTHONPATH="$ACADEMIC_DIR/eval${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

uenv run "$PHASE04_UENV" --view=default -- \
  "$RUNTIME_VENV/bin/python" -m sequence_models.gold_data sample \
  --pipeline-config "$SEQUENCE_DIR/gold_pipeline.json" \
  --eval-config "$SEQUENCE_DIR/config.json" \
  --phase04-sources "$PHASE04_DIR/configs/sources.json" \
  --input-receipt "$INPUT_RECEIPT" \
  --source-input "greek_phd=$GREEK_PHD_INPUT" \
  --source-input "openarchives=$OPENARCHIVES_INPUT" \
  --source-input "kallipos=$KALLIPOS_INPUT" \
  --source-input "pergamos=$PERGAMOS_INPUT" \
  --state "$STATE" \
  --output-dir "$PACKET_ROOT"

echo "PACKET_ROOT=$PACKET_ROOT"
echo "RESUME_STATE=$STATE"

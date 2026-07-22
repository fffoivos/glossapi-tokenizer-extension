#!/usr/bin/env bash
#SBATCH --job-name=tocbib_gold_import
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
phase04_require_compute_partition "human-gold validation and token counting"
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
[[ "${SLURM_GPUS:-0}" == "0" ]] || { echo "ERROR: gold import is CPU-only" >&2; exit 93; }

PACKET_ROOT=${PACKET_ROOT:?set the completed locked packet root}
ANNOTATIONS=${ANNOTATIONS:?set directory containing packet_id.slot.json independent human annotations}
ADJUDICATIONS=${ADJUDICATIONS:?set directory containing packet_id.json human adjudications}
RUN_TAG=${RUN_TAG:-human_gold_import_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$RUN_ROOT/$RUN_TAG}
mkdir -p "$OUT"
phase04_require_file "$PACKET_ROOT/COMPLETED.json"
phase04_require_file "$PACKET_ROOT/packet_manifest.json"
phase04_require_file "$PACKET_ROOT/split_manifest.json"
phase04_require_file "$TOKENIZER_JSON"
[[ -d "$ANNOTATIONS" && -d "$ADJUDICATIONS" ]] || {
  echo "ERROR: human annotation/adjudication directories are required" >&2
  exit 94
}
export PYTHONPATH="$ACADEMIC_DIR/eval${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

uenv run "$PHASE04_UENV" --view=default -- \
  "$RUNTIME_VENV/bin/python" -m sequence_models.gold_data import-gold \
  --pipeline-config "$SEQUENCE_DIR/gold_pipeline.json" \
  --eval-config "$SEQUENCE_DIR/config.json" \
  --packet-root "$PACKET_ROOT" \
  --annotations "$ANNOTATIONS" \
  --adjudications "$ADJUDICATIONS" \
  --tokenizer-json "$TOKENIZER_JSON" \
  --output "$OUT/academic_structure_gold.jsonl" \
  --receipt "$OUT/import_receipt.json"

sha256sum "$OUT/academic_structure_gold.jsonl" "$OUT/import_receipt.json" > "$OUT/outputs.sha256"
touch "$OUT/COMPLETED"
echo "GOLD_IMPORT_COMPLETE=$OUT"

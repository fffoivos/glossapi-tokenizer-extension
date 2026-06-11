#!/usr/bin/env bash
#SBATCH --job-name=anonymize_pii
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Apertus-parity PII masking (email/IP/IBAN -> <email-pii>/<ip-pii>/<iban-pii>)
# over the corpus, via datatrove's process-parallel executor.
#
# INPUT FORM IS NOT HARDCODED. The canonical-dataset form is a separate
# decision — set INPUT_DIR / READER / TEXT_KEY / ID_KEY accordingly. Examples:
#   per-source GlossAPI parquets: READER=parquet TEXT_KEY=text ID_KEY=doc_key
#   the 5B mixed jsonl:           READER=jsonl   TEXT_KEY=text ID_KEY=doc_id
#
# Compute justification (per feedback_compute_sweet_spot_justify):
#   1. Parallelism: datatrove LocalPipelineExecutor, tasks=$SLURM_CPUS_PER_TASK
#      (process-parallel sharded; the same framework + scale-proven path
#      Apertus/FineWeb used). Regex masking is embarrassingly parallel per shard.
#   2. CPU/GPU: CPU partition, no GPU. ~all cores busy across shards.
#   3. Mem: regex pass holds one doc at a time per worker + small dedup sets;
#      160G is generous headroom for 64 workers.
#   4. Efficiency gaps: C-backed `re`; no Python hot-loop. A Rust port would
#      buy little and risk diverging from Apertus parity — not worth it.

SUBPROJECT_DIR="${SUBPROJECT_DIR:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt}"
SCRIPTS_DIR="$SUBPROJECT_DIR/02_corpus_preparation/40_anonymize/scripts"

# ---- inputs (override per the canonical-dataset decision) ----
INPUT_DIR="${INPUT_DIR:?set INPUT_DIR to the corpus shards folder}"
READER="${READER:?set READER=parquet|jsonl}"
TEXT_KEY="${TEXT_KEY:-text}"
ID_KEY="${ID_KEY:-doc_id}"
GLOB="${GLOB:-}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR for masked shards}"
WRITER="${WRITER:-$READER}"
TASKS="${TASKS:-${SLURM_CPUS_PER_TASK:-64}}"
LIMIT="${LIMIT:--1}"

mkdir -p "$OUTPUT_DIR"

echo "=== run_anonymize.sbatch ==="
date -u; echo "host: $(hostname)"
echo "INPUT_DIR=$INPUT_DIR READER=$READER TEXT_KEY=$TEXT_KEY ID_KEY=$ID_KEY GLOB='$GLOB'"
echo "OUTPUT_DIR=$OUTPUT_DIR WRITER=$WRITER TASKS=$TASKS LIMIT=$LIMIT"
echo

uenv start pytorch/v2.9.1:v2 --view=default --ignore-tty <<INNER
set -euo pipefail
python3 -c 'import sys; assert sys.version_info >= (3,8); print("python", sys.version.split()[0])'
python3 -c 'import datatrove; print("datatrove", datatrove.__version__)' \
  || pip install --quiet datatrove
GLOB_ARG=""
[ -n "$GLOB" ] && GLOB_ARG="--glob $GLOB"
python3 "$SCRIPTS_DIR/anonymize.py" \
  --input "$INPUT_DIR" \
  --reader "$READER" \
  --output "$OUTPUT_DIR" \
  --writer "$WRITER" \
  --text-key "$TEXT_KEY" \
  --id-key "$ID_KEY" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  \$GLOB_ARG
INNER

echo
echo "DONE $(date -u)"

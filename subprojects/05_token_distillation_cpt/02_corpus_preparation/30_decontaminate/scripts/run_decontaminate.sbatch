#!/usr/bin/env bash
#SBATCH --job-name=decontam_greekmmlu_5b
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Decontamination of the Apertus Greek CPT corpus against GreekMMLU
# (dascim/GreekMMLU). Production rule = `correct_only`
# (q_plus_correct_only ∪ q_plus_correct_and_wrong; 92.4 % codex-judge
# precision per the 250-pair audit at
# decontamination/experiments/v2_validation_250pair_codex_20260602/).
#
# Outputs:
#   $RUN_DIR/output/                          ← audit JSON + per-item categories
#   $RUN_DIR/output/contaminated_doc_ids_correct_only_<TS>.txt
#   $RUN_DIR/decontaminated/<corpus_stem>_clean.jsonl
#   $RUN_DIR/decontaminated/<corpus_stem>_dropped.jsonl
#
# Compute justification (per feedback_compute_sweet_spot_justify):
#   1. Parallelism strategy: multiprocessing.Pool with fork start method,
#      n_workers=$SLURM_CPUS_PER_TASK. The Q-gram index (~tens of MB) is
#      fork-shared copy-on-write — workers add no memory. Per-doc work
#      (json.loads / tokenize / k-gram lookup / option-window check) is
#      embarrassingly parallel across rows; bypasses the Python GIL by
#      using processes. Verified identical output vs serial mode on the
#      102-doc local validation corpus (2026-06-03).
#   2. CPU + GPU saturation: CPU partition. No GPU. With 32 workers we
#      expect ~20–28 cores busy (some overhead in parent IPC). v1's
#      4288 s single-threaded scan ÷ ~5× parallelism overhead = ~15 min
#      expected scan wall.
#   3. Memory ceiling vs observed peak: v1 single-threaded peaked ~30 GB.
#      v2 parallel: parent holds evidence dict (~30–50 GB), workers add
#      ~50–100 MB each × 32 = ~3 GB. Allocating 160 G with a ~2× safety
#      margin without claiming the whole node.
#   4. Known efficiency gaps: json.loads is still the per-row hot path
#      and runs in Python C extension (GIL released for the parse). Rust
#      rewrite would be another ~5–10× but not the critical path for a
#      one-shot decontamination of a fixed corpus.
#
# Submit from the dev mirror after rsyncing decontamination/scripts/ to
# Clariden. CPU-only; no GPU allocation.
#
# Partition note: xfer is drained for Apertus maintenance until 2026-06-11.
# This script targets `normal` per the local Clariden maintenance memory.
# After 06-11, either partition is fine.

SUBPROJECT_DIR="${SUBPROJECT_DIR:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/decontamination}"
RUN_TAG="${RUN_TAG:-05_decontam_5b_correctonly_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"

QUERIES_JSONL_DEFAULT="$RUN_ROOT/05_decontam_5b_20260602T011447Z/queries/native_greek_mcq_decontam_queries.jsonl"
QUERIES_JSONL="${QUERIES_JSONL:-$QUERIES_JSONL_DEFAULT}"

CORPUS_JSONL="${CORPUS_JSONL:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_hplt_b1_dataset_5b_20260528T112539Z/jsonl/hplt_b1_5b.jsonl}"

K="${K:-8}"
MAX_GAP_TOKENS="${MAX_GAP_TOKENS:-50}"
MAX_GAP_TOKENS_SHORT="${MAX_GAP_TOKENS_SHORT:-5}"
DIRECTION="${DIRECTION:-after}"
BENCHMARK="${BENCHMARK:-greekmmlu}"
PROGRESS_EVERY_ROWS="${PROGRESS_EVERY_ROWS:-50000}"
MAX_PER_ITEM_DETAIL="${MAX_PER_ITEM_DETAIL:-50}"
PRIMARY_RULE="${PRIMARY_RULE:-correct_only}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
MP_CHUNKSIZE="${MP_CHUNKSIZE:-1024}"

REPORT_DIR_LOCAL="$SUBPROJECT_DIR/02_corpus_preparation/30_decontaminate/reports"
CORPUS_STEM="$(basename "$CORPUS_JSONL" .jsonl)"
CLEAN_JSONL="$RUN_DIR/decontaminated/${CORPUS_STEM}_clean.jsonl"
DROPPED_JSONL="$RUN_DIR/decontaminated/${CORPUS_STEM}_dropped.jsonl"

mkdir -p "$RUN_DIR/output" "$RUN_DIR/decontaminated" "$REPORT_DIR_LOCAL"

echo "=== run_decontaminate.sbatch ==="
date -u
echo "host: $(hostname)"
echo "SUBPROJECT_DIR: $SUBPROJECT_DIR"
echo "RUN_DIR: $RUN_DIR"
echo "QUERIES_JSONL: $QUERIES_JSONL"
echo "CORPUS_JSONL: $CORPUS_JSONL"
echo "BENCHMARK: $BENCHMARK"
echo "K: $K"
echo "MAX_GAP_TOKENS: $MAX_GAP_TOKENS"
echo "MAX_GAP_TOKENS_SHORT: $MAX_GAP_TOKENS_SHORT"
echo "DIRECTION: $DIRECTION"
echo "PRIMARY_RULE: $PRIMARY_RULE"
echo "N_WORKERS: $N_WORKERS"
echo "MP_CHUNKSIZE: $MP_CHUNKSIZE"
echo "CLEAN_JSONL: $CLEAN_JSONL"
echo "DROPPED_JSONL: $DROPPED_JSONL"
echo

test -s "$QUERIES_JSONL"
test -s "$CORPUS_JSONL"

# Clariden is aarch64 (GH200). The script uses `from __future__ import
# annotations` (PEP 563, needs 3.7+); login default is 3.6. We use the
# pytorch/v2.9.1:v2 uenv image (Python 3.12.12 on gh200) — the same image
# already used by ~/normalize_nfc.sbatch and ~/mix_builder_smoke.sbatch.
uenv start pytorch/v2.9.1:v2 --view=default --ignore-tty <<INNER
set -euo pipefail
python3 -c 'import sys; assert sys.version_info >= (3, 7); print("python", sys.version.split()[0])'
python3 "$SUBPROJECT_DIR/02_corpus_preparation/30_decontaminate/scripts/decontaminate.py" \
  --queries-jsonl "$QUERIES_JSONL" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --output-dir "$RUN_DIR/output" \
  --benchmark "$BENCHMARK" \
  --k "$K" \
  --max-gap-tokens "$MAX_GAP_TOKENS" \
  --max-gap-tokens-short "$MAX_GAP_TOKENS_SHORT" \
  --direction "$DIRECTION" \
  --progress-every-rows "$PROGRESS_EVERY_ROWS" \
  --max-per-item-detail "$MAX_PER_ITEM_DETAIL" \
  --primary-rule "$PRIMARY_RULE" \
  --n-workers "$N_WORKERS" \
  --mp-chunksize "$MP_CHUNKSIZE" \
  --filter-output-clean   "$CLEAN_JSONL" \
  --filter-output-dropped "$DROPPED_JSONL"
INNER

# Mirror the audit JSON + per-item detail + contaminated-doc-ids txt back to
# the local reports dir for review without needing Clariden access.
for f in "$RUN_DIR/output/"greekmmlu_dclm_audit_*.json \
         "$RUN_DIR/output/"greekmmlu_dclm_per_item_categories_*.jsonl \
         "$RUN_DIR/output/"contaminated_doc_ids_*.txt; do
  if [ -f "$f" ]; then
    cp -v "$f" "$REPORT_DIR_LOCAL/"
  fi
done

echo
echo "DONE $(date -u)"

#!/usr/bin/env bash
# CPU-only Stage-1 benchmark decontamination scan for the Task-1 5B JSONL.

#SBATCH --job-name=05_decontam_5b
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "05 5B benchmark decontamination"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/decontamination}"
RUN_TAG="${RUN_TAG:-05_decontam_5b_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
CORPUS_JSONL="${CORPUS_JSONL:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_hplt_b1_dataset_5b_20260528T112539Z/jsonl/hplt_b1_5b.jsonl}"
REGISTRY="${REGISTRY:-$LEGACY_INIT_DIR/eval/native_greek_benchmark_registry.json}"
EVAL_DIR="${EVAL_DIR:-$LEGACY_INIT_DIR/eval}"
K_VALUES="${K_VALUES:-8,13}"
THRESHOLDS="${THRESHOLDS:-0,0.1,0.2,0.4,0.6,0.8}"
MAX_HIT_RECORDS="${MAX_HIT_RECORDS:-500000}"
PROGRESS_EVERY_ROWS="${PROGRESS_EVERY_ROWS:-100000}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x /users/fffoivos/.local/bin/uv ]; then
  UV_BIN=/users/fffoivos/.local/bin/uv
fi

QUERIES_JSONL="$RUN_DIR/queries/native_greek_mcq_decontam_queries.jsonl"
QUERIES_SUMMARY="$RUN_DIR/queries/native_greek_mcq_decontam_queries_summary.json"
SCAN_DIR="$RUN_DIR/scan"
AUDIT_JSON="$SCAN_DIR/decontamination_audit_5b.json"
REPORT_COPY="$SUBPROJECT_DIR/decontamination/reports/decontamination_audit_5b_${RUN_TAG#05_decontam_5b_}.json"

mkdir -p "$RUN_DIR"/{queries,scan} "$SUBPROJECT_DIR/decontamination/reports" "$(dirname "$REPORT_COPY")" "$RUN_ROOT"

echo "=== run_5b_decontamination.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "REPO_ROOT: $REPO_ROOT"
echo "SUBPROJECT_DIR: $SUBPROJECT_DIR"
echo "LEGACY_INIT_DIR: $LEGACY_INIT_DIR"
echo "RUN_DIR: $RUN_DIR"
echo "CORPUS_JSONL: $CORPUS_JSONL"
echo "REGISTRY: $REGISTRY"
echo "EVAL_DIR: $EVAL_DIR"
echo "K_VALUES: $K_VALUES"
echo "THRESHOLDS: $THRESHOLDS"
echo "MAX_HIT_RECORDS: $MAX_HIT_RECORDS"
echo "PROGRESS_EVERY_ROWS: $PROGRESS_EVERY_ROWS"
echo "UV_BIN: ${UV_BIN:-none}"
echo

test -s "$CORPUS_JSONL"
test -f "$REGISTRY"
test -d "$EVAL_DIR"
if [ -z "$UV_BIN" ] || [ ! -x "$UV_BIN" ]; then
  echo "ERROR: uv is required on xfer for Python >=3.12 + datasets; UV_BIN=$UV_BIN" >&2
  exit 127
fi

run_python() {
  "$UV_BIN" run --python 3.12 --with datasets -- python "$@"
}

run_python -c 'import sys, datasets; assert sys.version_info >= (3, 10); print("python", sys.version.split()[0], "datasets", datasets.__version__)'

run_python "$SUBPROJECT_DIR/decontamination/scripts/build_decontamination_queries.py" \
  --registry "$REGISTRY" \
  --eval-dir "$EVAL_DIR" \
  --output-jsonl "$QUERIES_JSONL" \
  --summary-json "$QUERIES_SUMMARY"

run_python "$SUBPROJECT_DIR/decontamination/scripts/scan_5b_decontamination.py" \
  --queries-jsonl "$QUERIES_JSONL" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --output-dir "$SCAN_DIR" \
  --k "$K_VALUES" \
  --thresholds "$THRESHOLDS" \
  --max-hit-records "$MAX_HIT_RECORDS" \
  --progress-every-rows "$PROGRESS_EVERY_ROWS"

cp "$AUDIT_JSON" "$REPORT_COPY"
cat > "$RUN_DIR/decontamination_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
QUERIES_JSONL="$QUERIES_JSONL"
QUERIES_SUMMARY="$QUERIES_SUMMARY"
SCAN_DIR="$SCAN_DIR"
AUDIT_JSON="$AUDIT_JSON"
REPORT_COPY="$REPORT_COPY"
CORPUS_JSONL="$CORPUS_JSONL"
K_VALUES="$K_VALUES"
THRESHOLDS="$THRESHOLDS"
PATHS

echo
echo "=== decontamination complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "AUDIT_JSON=$AUDIT_JSON"
echo "REPORT_COPY=$REPORT_COPY"

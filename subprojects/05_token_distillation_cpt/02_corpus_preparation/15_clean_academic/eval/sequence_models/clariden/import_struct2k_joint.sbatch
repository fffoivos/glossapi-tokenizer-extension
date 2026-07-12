#!/usr/bin/env bash
# Clariden CPU-only audit/transcode of the recovered joint STRUCT-2K LLM silver.
# The 608 historical-test documents are verified, hashed, and excluded before
# the emitted train/validation corpus becomes visible to any model process.
#SBATCH --job-name=cpt_struct2k_import
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

PHASE04_CLARIDEN_DIR=${PHASE04_CLARIDEN_DIR:?set the exact Phase-04 clariden directory}
source "$PHASE04_CLARIDEN_DIR/paths.env"
source "$PHASE04_CLARIDEN_DIR/common.sh"
phase04_require_compute_partition "joint STRUCT-2K LLM-silver import"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

SEQUENCE_DIR="$ACADEMIC_DIR/eval/sequence_models"
LOCK="$SEQUENCE_DIR/struct2k_handoff_lock.json"
CONFIG="$SEQUENCE_DIR/joint_config.json"
STRUCT2K_HANDOFF_ROOT=${STRUCT2K_HANDOFF_ROOT:?set the transferred APERTUS_CLASSIFIER_HANDOFF_20260712 root}
STRUCT2K_IMPORT_RUN_ID=${STRUCT2K_IMPORT_RUN_ID:?set an immutable import run ID}
[[ "$STRUCT2K_IMPORT_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$ ]] || {
    echo "ERROR: invalid STRUCT2K_IMPORT_RUN_ID=$STRUCT2K_IMPORT_RUN_ID" >&2
    exit 101
}
OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/classifier_research/struct2k_sources/$STRUCT2K_IMPORT_RUN_ID}
for path in \
    "$LOCK" \
    "$CONFIG" \
    "$TOKENIZER_JSON" \
    "$STRUCT2K_HANDOFF_ROOT/INVENTORY.sha256" \
    "$STRUCT2K_HANDOFF_ROOT/SOURCE_STATE.txt" \
    "$STRUCT2K_HANDOFF_ROOT/STRUCT_2K_gold.jsonl" \
    "$STRUCT2K_HANDOFF_ROOT/STRUCT_2K/manifest.jsonl"; do
    phase04_require_file "$path"
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
    echo "ERROR: immutable output already exists: $OUTPUT_ROOT" >&2
    exit 102
}

echo "JOINT STRUCT-2K IMPORT PLAN"
echo "  handoff=$STRUCT2K_HANDOFF_ROOT"
echo "  inventory_sha256=26bf52e4a23caf08fc99f96dfcafa77bb5b5a95294673551e590b0ce92233a4f"
echo "  raw_joint_sha256=ff4146763fd7f82cdcd5bd2eaf4cd0080527df011e238325e5ad6f08013ca1cf"
echo "  labels=existing_gpt-5.5_LLM_silver_with_8_locked_coordinate_typo_corrections"
echo "  historical_train=1392"
echo "  historical_test=608_physically_excluded_before_output"
echo "  output=$OUTPUT_ROOT"
if [[ "${CONFIRM_STRUCT2K_IMPORT:-0}" != "1" ]]; then
    echo "DRY_RUN_ONLY=1"
    exit 0
fi

LOCK_ROOT="$RUN_ROOT/classifier_research/struct2k_import_locks"
mkdir -p "$LOCK_ROOT"
exec {RUN_LOCK_FD}>"$LOCK_ROOT/$STRUCT2K_IMPORT_RUN_ID.lock"
flock -n "$RUN_LOCK_FD" || { echo "ERROR: import run ID already active" >&2; exit 103; }
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || exit 104
PARTIAL="$OUTPUT_ROOT.partial-${SLURM_JOB_ID:?}"
[[ ! -e "$PARTIAL" && ! -L "$PARTIAL" ]] || exit 105
mkdir -p "$PARTIAL"

SILVER="$PARTIAL/struct2k.LLM_silver.jsonl"
SPLIT="$PARTIAL/struct2k.LLM_silver.split.json"
SNAPSHOT_RECEIPT="$PARTIAL/struct2k.handoff.audit.receipt.json"
SOURCE_RECEIPT="$PARTIAL/struct2k.LLM_silver.receipt.json"
PYTHON="$RUNTIME_VENV/bin/python"
run_python() {
    uenv run "$PHASE04_UENV" --view=default -- "$PYTHON" "$@"
}

cd "$ACADEMIC_DIR/eval"
run_python -c 'import tokenizers; import sequence_models.struct2k_import'
run_python -m sequence_models.struct2k_import materialize \
    --handoff-root "$STRUCT2K_HANDOFF_ROOT" \
    --lock "$LOCK" \
    --tokenizer-json "$TOKENIZER_JSON" \
    --config "$CONFIG" \
    --silver "$SILVER" \
    --split-manifest "$SPLIT" \
    --snapshot-receipt "$SNAPSHOT_RECEIPT" \
    --receipt "$SOURCE_RECEIPT"
run_python -m sequence_models.struct2k_import verify \
    --root "$PARTIAL" \
    --lock "$LOCK" \
    --config "$CONFIG"

mkdir -p "$(dirname "$OUTPUT_ROOT")"
mv --no-clobber -T "$PARTIAL" "$OUTPUT_ROOT"
[[ ! -e "$PARTIAL" && -s "$OUTPUT_ROOT/struct2k.LLM_silver.receipt.json" ]] || exit 106
run_python -m sequence_models.struct2k_import verify \
    --root "$OUTPUT_ROOT" \
    --lock "$LOCK" \
    --config "$CONFIG"
echo "STRUCT2K_JOINT_SOURCE_COMPLETE=$OUTPUT_ROOT"
echo "STRUCT2K_JOINT_SOURCE_RECEIPT=$OUTPUT_ROOT/struct2k.LLM_silver.receipt.json"

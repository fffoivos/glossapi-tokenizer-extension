#!/usr/bin/env bash
# Clariden-only one-epoch joint ToC+BIB N1 profile plus deterministic tiny-fit gate.
# Run after import_struct2k_joint.sbatch and before run_joint_ladder.sbatch.
# This research job is intentionally absent from the corpus-production DAG.
#SBATCH --job-name=cpt_joint_n1_profile
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
phase04_require_compute_partition "joint ToC+BIB N1 one-epoch profile"
phase04_require_cpu_request
phase04_require_uenv_python python3
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

SEQUENCE_DIR="$ACADEMIC_DIR/eval/sequence_models"
CONFIG="$SEQUENCE_DIR/joint_config.json"
STRUCT2K_LOCK="$SEQUENCE_DIR/struct2k_handoff_lock.json"
STRUCT2K_SOURCE_ROOT=${STRUCT2K_SOURCE_ROOT:?set the immutable output from import_struct2k_joint.sbatch}
SILVER_JSONL="$STRUCT2K_SOURCE_ROOT/struct2k.LLM_silver.jsonl"
SOURCE_SPLIT="$STRUCT2K_SOURCE_ROOT/struct2k.LLM_silver.split.json"
SOURCE_RECEIPT="$STRUCT2K_SOURCE_ROOT/struct2k.LLM_silver.receipt.json"
N1_JOINT_PROFILE_RUN_ID=${N1_JOINT_PROFILE_RUN_ID:?set an immutable profile run ID}
[[ "$N1_JOINT_PROFILE_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$ ]] || {
    echo "ERROR: invalid N1_JOINT_PROFILE_RUN_ID=$N1_JOINT_PROFILE_RUN_ID" >&2
    exit 101
}
OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/classifier_research/joint_n1_profiles/$N1_JOINT_PROFILE_RUN_ID}
for path in "$CONFIG" "$STRUCT2K_LOCK" "$SILVER_JSONL" "$SOURCE_SPLIT" "$SOURCE_RECEIPT"; do
    phase04_require_file "$path"
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
    echo "ERROR: immutable profile output already exists: $OUTPUT_ROOT" >&2
    exit 102
}

echo "JOINT N1 PROFILE PLAN"
echo "  source=$STRUCT2K_SOURCE_ROOT"
echo "  source_receipt=$SOURCE_RECEIPT"
echo "  code_commit=$PHASE04_EXPECTED_COMMIT"
echo "  active_classes=BIB,TOC"
echo "  architecture_inputs=historical_train_derived_train_and_validation_only"
echo "  historical_test_documents_available_to_model_processes=0"
echo "  output=$OUTPUT_ROOT"
echo "  work=two-replica-tiny-determinism-smoke_plus_one_full_train_epoch"
if [[ "${CONFIRM_N1_JOINT_PROFILE:-0}" != "1" ]]; then
    echo "DRY_RUN_ONLY=1"
    echo "Set CONFIRM_N1_JOINT_PROFILE=1 to execute this CPU-only research job."
    exit 0
fi

LOCK_ROOT="$RUN_ROOT/classifier_research/joint_n1_profile_locks"
mkdir -p "$LOCK_ROOT"
exec {RUN_LOCK_FD}>"$LOCK_ROOT/$N1_JOINT_PROFILE_RUN_ID.lock"
flock -n "$RUN_LOCK_FD" || {
    echo "ERROR: joint profile run ID already active: $N1_JOINT_PROFILE_RUN_ID" >&2
    exit 103
}
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || exit 104
PARTIAL="$OUTPUT_ROOT.partial-${SLURM_JOB_ID:?SLURM_JOB_ID is unset}"
[[ ! -e "$PARTIAL" && ! -L "$PARTIAL" ]] || exit 105
mkdir -p "$PARTIAL"

SELECTION_SILVER="$PARTIAL/selection.train-validation.jsonl"
SELECTION_MANIFEST="$PARTIAL/selection.train-validation.split.json"
VALIDATION_SILVER="$PARTIAL/selection.validation.jsonl"
SELECTION_RECEIPT="$PARTIAL/selection.receipt.json"
PROFILE_RECEIPT="$PARTIAL/n1.profile.receipt.json"

run_python() {
    uenv run "$PHASE04_UENV" --view=default -- \
        env CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 \
        MKL_NUM_THREADS=8 SEQUENCE_UENV="$PHASE04_UENV" python3 "$@"
}

cd "$ACADEMIC_DIR/eval"
run_python -c 'import numpy, torch; import sequence_models.bib_ladder, sequence_models.char_tcn_crf, sequence_models.contract, sequence_models.features, sequence_models.struct2k_import; assert not torch.cuda.is_available()'
run_python -m sequence_models.struct2k_import verify \
    --root "$STRUCT2K_SOURCE_ROOT" \
    --lock "$STRUCT2K_LOCK" \
    --config "$CONFIG"
run_python -m sequence_models.bib_ladder prepare-selection \
    --silver "$SILVER_JSONL" \
    --split-manifest "$SOURCE_SPLIT" \
    --rehydration-receipt "$SOURCE_RECEIPT" \
    --config "$CONFIG" \
    --selection-silver "$SELECTION_SILVER" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --validation-silver "$VALIDATION_SILVER" \
    --receipt "$SELECTION_RECEIPT"
run_python -m sequence_models.char_tcn_crf profile \
    --selection-silver "$SELECTION_SILVER" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --validation-silver "$VALIDATION_SILVER" \
    --selection-receipt "$SELECTION_RECEIPT" \
    --config "$CONFIG" \
    --uenv "$PHASE04_UENV" \
    --code-commit "$PHASE04_EXPECTED_COMMIT" \
    --receipt-out "$PROFILE_RECEIPT" \
    --confirm-clariden-cpu-only

mkdir -p "$(dirname "$OUTPUT_ROOT")"
mv --no-clobber -T "$PARTIAL" "$OUTPUT_ROOT"
[[ ! -e "$PARTIAL" && -s "$OUTPUT_ROOT/n1.profile.receipt.json" ]] || exit 106
run_python -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); assert value["status"] == "passed_one_epoch_profile_and_determinism_smoke" and value["within_full_fit_budget"] is True and value["production_eligible"] is False and value["target"] == "BIB+TOC" and value["active_classes"] == ["BIB", "TOC"]' "$OUTPUT_ROOT/n1.profile.receipt.json"
echo "JOINT_N1_PROFILE_COMPLETE=$OUTPUT_ROOT/n1.profile.receipt.json"

#!/usr/bin/env bash
# Detached CPU-only C0/C1/C2/N1 validation comparison over rehydrated SPAN silver.
# This research runner is intentionally absent from the production corpus DAG.
# It emits no historical-partition prediction and cannot authorize production.
#SBATCH --job-name=cpt_bib_ladder
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

PHASE04_CLARIDEN_DIR=${PHASE04_CLARIDEN_DIR:?set the exact checked-out Phase-04 clariden directory}
source "$PHASE04_CLARIDEN_DIR/paths.env"
source "$PHASE04_CLARIDEN_DIR/common.sh"

phase04_require_compute_partition "BIB sequence-model ladder"
phase04_require_cpu_request
phase04_require_uenv_python python3
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

SEQUENCE_DIR="$ACADEMIC_DIR/eval/sequence_models"
CONFIG=${STRUCTURAL_SEQUENCE_CONFIG:-$SEQUENCE_DIR/config.json}
SPAN_REHYDRATION_ROOT=${SPAN_REHYDRATION_ROOT:?set the immutable output root from 07_rehydrate_span_silver.sbatch}
SILVER_JSONL="$SPAN_REHYDRATION_ROOT/span.LLM_silver.jsonl"
SOURCE_SPLIT="$SPAN_REHYDRATION_ROOT/span.LLM_silver.split.json"
SOURCE_RECEIPT="$SPAN_REHYDRATION_ROOT/span.LLM_silver.receipt.json"
CLASSIFIER_COMPARISON_RUN_ID=${CLASSIFIER_COMPARISON_RUN_ID:?set an immutable comparison run ID}
N1_PROFILE_RECEIPT=${N1_PROFILE_RECEIPT:?set the passed one-epoch N1 profile receipt}
[[ "$CLASSIFIER_COMPARISON_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$ ]] || {
    echo "ERROR: invalid CLASSIFIER_COMPARISON_RUN_ID=$CLASSIFIER_COMPARISON_RUN_ID" >&2
    exit 101
}
OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/classifier_research/comparisons/$CLASSIFIER_COMPARISON_RUN_ID}

for path in "$CONFIG" "$SILVER_JSONL" "$SOURCE_SPLIT" "$SOURCE_RECEIPT" "$N1_PROFILE_RECEIPT"; do
    phase04_require_file "$path"
done
[[ -f "$SEQUENCE_DIR/bib_ladder.py" && -f "$SEQUENCE_DIR/char_tcn_crf.py" ]] || {
    echo "ERROR: classifier ladder code is absent from the exact checkout" >&2
    exit 102
}
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
    echo "ERROR: immutable comparison output already exists: $OUTPUT_ROOT" >&2
    exit 103
}

echo "BIB SEQUENCE-MODEL COMPARISON PLAN"
echo "  source_rehydration_root=$SPAN_REHYDRATION_ROOT"
echo "  source_receipt=$SOURCE_RECEIPT"
echo "  code_commit=$PHASE04_EXPECTED_COMMIT"
echo "  architectures=c0-rust-lr-hysteresis,c1-feature-bioes-crf,c2-char-ngram-feature-bioes-crf,n1-bytecnn-tcn-masked-crf"
echo "  architecture_inputs=train_and_validation_only"
echo "  historical_partition_predictions=forbidden"
echo "  output_root=$OUTPUT_ROOT"
if [[ "${CONFIRM_CLASSIFIER_COMPARISON:-0}" != "1" ]]; then
    echo "DRY_RUN_ONLY=1"
    echo "Set CONFIRM_CLASSIFIER_COMPARISON=1 to execute this CPU-only research job."
    exit 0
fi

LOCK_ROOT="$RUN_ROOT/classifier_research/comparison_locks"
mkdir -p "$LOCK_ROOT"
exec {RUN_LOCK_FD}>"$LOCK_ROOT/$CLASSIFIER_COMPARISON_RUN_ID.lock"
flock -n "$RUN_LOCK_FD" || {
    echo "ERROR: comparison run ID is already active: $CLASSIFIER_COMPARISON_RUN_ID" >&2
    exit 105
}
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
    echo "ERROR: immutable comparison output appeared while acquiring lock: $OUTPUT_ROOT" >&2
    exit 106
}

PARTIAL="$OUTPUT_ROOT.partial-${SLURM_JOB_ID:?SLURM_JOB_ID is unset}"
[[ ! -e "$PARTIAL" && ! -L "$PARTIAL" ]] || {
    echo "ERROR: stale job staging directory exists: $PARTIAL" >&2
    exit 104
}
mkdir -p "$PARTIAL"

SELECTION_SILVER="$PARTIAL/selection.train-validation.jsonl"
SELECTION_MANIFEST="$PARTIAL/selection.train-validation.split.json"
VALIDATION_SILVER="$PARTIAL/selection.validation.jsonl"
SELECTION_RECEIPT="$PARTIAL/selection.receipt.json"
C0_PREDICTIONS="$PARTIAL/c0.validation.predictions.jsonl"
C0_RECEIPT="$PARTIAL/c0.receipt.json"
C1_MODEL="$PARTIAL/c1.model.npz"
C1_PREDICTIONS="$PARTIAL/c1.validation.predictions.jsonl"
C1_RECEIPT="$PARTIAL/c1.receipt.json"
C2_MODEL="$PARTIAL/c2.model.npz"
C2_PREDICTIONS="$PARTIAL/c2.validation.predictions.jsonl"
C2_RECEIPT="$PARTIAL/c2.receipt.json"
N1_PROFILE_COPY="$PARTIAL/n1.profile.receipt.json"
N1_MODEL="$PARTIAL/n1.model.pt"
N1_RECEIPT="$PARTIAL/n1.training.receipt.json"
N1_PREDICTIONS="$PARTIAL/n1.validation.predictions.jsonl"
VALIDATION_REPORT="$PARTIAL/validation.comparison.json"
RUN_RECEIPT="$PARTIAL/run.receipt.json"

run_python() {
    uenv run "$PHASE04_UENV" --view=default -- \
        env CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 \
        MKL_NUM_THREADS=8 SEQUENCE_UENV="$PHASE04_UENV" \
        python3 "$@"
}

cd "$ACADEMIC_DIR/eval"
run_python -c 'import numpy, torch; import sequence_models.baseline, sequence_models.bib_ladder, sequence_models.char_tcn_crf, sequence_models.contract, sequence_models.evaluate, sequence_models.feature_crf, sequence_models.features; assert not torch.cuda.is_available()'

# This is the only command that opens the source artifact containing historical
# sealed-comparison rows. It validates the full receipt and writes a physically separate
# train+validation bundle. Every architecture command below receives that bundle.
run_python -m sequence_models.bib_ladder prepare-selection \
    --silver "$SILVER_JSONL" \
    --split-manifest "$SOURCE_SPLIT" \
    --rehydration-receipt "$SOURCE_RECEIPT" \
    --config "$CONFIG" \
    --selection-silver "$SELECTION_SILVER" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --validation-silver "$VALIDATION_SILVER" \
    --receipt "$SELECTION_RECEIPT"

cp --no-clobber --reflink=auto "$N1_PROFILE_RECEIPT" "$N1_PROFILE_COPY"

run_python -m sequence_models.baseline \
    --selection-silver "$SELECTION_SILVER" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --validation-silver "$VALIDATION_SILVER" \
    --selection-receipt "$SELECTION_RECEIPT" \
    --config "$CONFIG" \
    --uenv "$PHASE04_UENV" \
    --output "$C0_PREDICTIONS" \
    --receipt-out "$C0_RECEIPT"

run_python -m sequence_models.feature_crf \
    --selection-silver "$SELECTION_SILVER" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --validation-silver "$VALIDATION_SILVER" \
    --selection-receipt "$SELECTION_RECEIPT" \
    --config "$CONFIG" \
    --reference-predictions "$C0_PREDICTIONS" \
    --architecture c1-feature-bioes-crf \
    --model-out "$C1_MODEL" \
    --validation-predictions "$C1_PREDICTIONS" \
    --receipt-out "$C1_RECEIPT" \
    --uenv "$PHASE04_UENV" \
    --confirm-clariden-cpu-only

run_python -m sequence_models.feature_crf \
    --selection-silver "$SELECTION_SILVER" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --validation-silver "$VALIDATION_SILVER" \
    --selection-receipt "$SELECTION_RECEIPT" \
    --config "$CONFIG" \
    --reference-predictions "$C0_PREDICTIONS" \
    --architecture c2-char-ngram-feature-bioes-crf \
    --model-out "$C2_MODEL" \
    --validation-predictions "$C2_PREDICTIONS" \
    --receipt-out "$C2_RECEIPT" \
    --uenv "$PHASE04_UENV" \
    --confirm-clariden-cpu-only

run_python -m sequence_models.char_tcn_crf train \
    --selection-silver "$SELECTION_SILVER" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --validation-silver "$VALIDATION_SILVER" \
    --selection-receipt "$SELECTION_RECEIPT" \
    --config "$CONFIG" \
    --reference-predictions "$C0_PREDICTIONS" \
    --profile-receipt "$N1_PROFILE_COPY" \
    --model-out "$N1_MODEL" \
    --validation-predictions "$N1_PREDICTIONS" \
    --receipt-out "$N1_RECEIPT" \
    --uenv "$PHASE04_UENV" \
    --code-commit "$PHASE04_EXPECTED_COMMIT" \
    --confirm-clariden-cpu-only

run_python -m sequence_models.bib_ladder evaluate-validation \
    --selection-silver "$SELECTION_SILVER" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --validation-silver "$VALIDATION_SILVER" \
    --selection-receipt "$SELECTION_RECEIPT" \
    --config "$CONFIG" \
    --baseline "$C0_PREDICTIONS" \
    --candidate "c1-feature-bioes-crf=$C1_PREDICTIONS" \
    --candidate "c2-char-ngram-feature-bioes-crf=$C2_PREDICTIONS" \
    --candidate "n1-bytecnn-tcn-masked-crf=$N1_PREDICTIONS" \
    --arm-receipt "c0-rust-lr-hysteresis=$C0_RECEIPT" \
    --arm-receipt "c1-feature-bioes-crf=$C1_RECEIPT" \
    --arm-receipt "c2-char-ngram-feature-bioes-crf=$C2_RECEIPT" \
    --arm-receipt "n1-bytecnn-tcn-masked-crf=$N1_RECEIPT" \
    --output "$VALIDATION_REPORT"

run_python -m sequence_models.bib_ladder finalize-run \
    --run-root "$PARTIAL" \
    --config "$CONFIG" \
    --source-rehydration-receipt "$SOURCE_RECEIPT" \
    --uenv "$PHASE04_UENV" \
    --code-commit "$PHASE04_EXPECTED_COMMIT" \
    --job-id "$SLURM_JOB_ID" \
    --artifact "selection_silver=$SELECTION_SILVER" \
    --artifact "selection_manifest=$SELECTION_MANIFEST" \
    --artifact "validation_silver=$VALIDATION_SILVER" \
    --artifact "selection_receipt=$SELECTION_RECEIPT" \
    --artifact "c0_validation_predictions=$C0_PREDICTIONS" \
    --artifact "c0_receipt=$C0_RECEIPT" \
    --artifact "c1_model=$C1_MODEL" \
    --artifact "c1_validation_predictions=$C1_PREDICTIONS" \
    --artifact "c1_receipt=$C1_RECEIPT" \
    --artifact "c2_model=$C2_MODEL" \
    --artifact "c2_validation_predictions=$C2_PREDICTIONS" \
    --artifact "c2_receipt=$C2_RECEIPT" \
    --artifact "n1_profile_receipt=$N1_PROFILE_COPY" \
    --artifact "n1_model=$N1_MODEL" \
    --artifact "n1_train_receipt=$N1_RECEIPT" \
    --artifact "n1_validation_predictions=$N1_PREDICTIONS" \
    --artifact "validation_report=$VALIDATION_REPORT" \
    --output "$RUN_RECEIPT"

mkdir -p "$(dirname "$OUTPUT_ROOT")"
mv --no-clobber -T "$PARTIAL" "$OUTPUT_ROOT"
[[ ! -e "$PARTIAL" && -s "$OUTPUT_ROOT/run.receipt.json" ]] || {
    echo "ERROR: no-replace publication postcondition failed" >&2
    exit 107
}
run_python -m sequence_models.bib_ladder verify-published-run --run-root "$OUTPUT_ROOT"
echo "BIB_LADDER_COMPLETE=$OUTPUT_ROOT"
echo "BIB_LADDER_RECEIPT=$OUTPUT_ROOT/run.receipt.json"

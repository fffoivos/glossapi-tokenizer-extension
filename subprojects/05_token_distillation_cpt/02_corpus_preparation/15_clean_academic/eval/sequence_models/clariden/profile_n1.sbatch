#!/usr/bin/env bash
# Clariden-only one-epoch N1 profile plus deterministic tiny-fit gate.
# Run this before run_bib_ladder.sbatch; it is not part of the corpus DAG.
#SBATCH --job-name=cpt_n1_profile
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
phase04_require_compute_partition "N1 one-epoch profile"
phase04_require_cpu_request
phase04_require_uenv_python python3
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

SEQUENCE_DIR="$ACADEMIC_DIR/eval/sequence_models"
CONFIG=${STRUCTURAL_SEQUENCE_CONFIG:-$SEQUENCE_DIR/config.json}
SPAN_REHYDRATION_ROOT=${SPAN_REHYDRATION_ROOT:?set the immutable output from 07}
SILVER_JSONL="$SPAN_REHYDRATION_ROOT/span.LLM_silver.jsonl"
SOURCE_SPLIT="$SPAN_REHYDRATION_ROOT/span.LLM_silver.split.json"
SOURCE_RECEIPT="$SPAN_REHYDRATION_ROOT/span.LLM_silver.receipt.json"
N1_PROFILE_RUN_ID=${N1_PROFILE_RUN_ID:?set an immutable profile run ID}
[[ "$N1_PROFILE_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$ ]] || exit 101
OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/classifier_research/n1_profiles/$N1_PROFILE_RUN_ID}
for path in "$CONFIG" "$SILVER_JSONL" "$SOURCE_SPLIT" "$SOURCE_RECEIPT"; do
    phase04_require_file "$path"
done

echo "N1 PROFILE PLAN"
echo "  source=$SPAN_REHYDRATION_ROOT"
echo "  output=$OUTPUT_ROOT"
echo "  work=two-replica-tiny-determinism-smoke_plus_one_full_train_epoch"
if [[ "${CONFIRM_N1_PROFILE:-0}" != "1" ]]; then
    echo "DRY_RUN_ONLY=1"
    exit 0
fi

LOCK_ROOT="$RUN_ROOT/classifier_research/n1_profile_locks"
mkdir -p "$LOCK_ROOT"
exec {RUN_LOCK_FD}>"$LOCK_ROOT/$N1_PROFILE_RUN_ID.lock"
flock -n "$RUN_LOCK_FD" || { echo "ERROR: profile run ID already active" >&2; exit 102; }
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || exit 103
PARTIAL="$OUTPUT_ROOT.partial-${SLURM_JOB_ID:?}"
[[ ! -e "$PARTIAL" && ! -L "$PARTIAL" ]] || exit 104
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
run_python -c 'import numpy, torch; import sequence_models.bib_ladder, sequence_models.char_tcn_crf, sequence_models.contract, sequence_models.features; assert not torch.cuda.is_available()'
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
[[ ! -e "$PARTIAL" && -s "$OUTPUT_ROOT/n1.profile.receipt.json" ]] || exit 105
run_python -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); assert value["status"] == "passed_one_epoch_profile_and_determinism_smoke" and value["within_full_fit_budget"] is True and value["production_eligible"] is False' "$OUTPUT_ROOT/n1.profile.receipt.json"
echo "N1_PROFILE_COMPLETE=$OUTPUT_ROOT/n1.profile.receipt.json"

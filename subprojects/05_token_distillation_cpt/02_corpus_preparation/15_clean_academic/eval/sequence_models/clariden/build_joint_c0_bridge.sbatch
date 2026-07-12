#!/usr/bin/env bash
# Build the C0 selection receipt and fresh Python↔Rust parity receipt after the joint ladder.
# This is CPU-only research/qualification work and is not part of the corpus DAG.
#SBATCH --job-name=cpt_joint_c0_bridge
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
phase04_require_compute_partition "joint C0 selection and Rust parity bridge"
phase04_require_cpu_request
phase04_require_uenv_python python3
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

SEQUENCE_DIR="$ACADEMIC_DIR/eval/sequence_models"
STRUCT2K_LOCK="$SEQUENCE_DIR/struct2k_handoff_lock.json"
STRUCT2K_SOURCE_ROOT=${STRUCT2K_SOURCE_ROOT:?set the immutable joint import output}
JOINT_LADDER_RUN_ROOT=${JOINT_LADDER_RUN_ROOT:?set the passed joint ladder output root}
STRUCTURAL_C0_BRIDGE_RUN_ID=${STRUCTURAL_C0_BRIDGE_RUN_ID:?set an immutable bridge run ID}
STRUCTURAL_SELECTION_RATIONALE=${STRUCTURAL_SELECTION_RATIONALE:?record why C0 is selected for the Rust candidate}
REFERENCE_BIN=${REFERENCE_BIN:?set the exact Clariden detector binary}
DETECTOR_BUILD_RECEIPT=${DETECTOR_BUILD_RECEIPT:?set its passed build receipt}
[[ "$STRUCTURAL_C0_BRIDGE_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$ ]] || exit 101

SILVER="$STRUCT2K_SOURCE_ROOT/struct2k.LLM_silver.jsonl"
SPLIT="$STRUCT2K_SOURCE_ROOT/struct2k.LLM_silver.split.json"
SOURCE_RECEIPT="$STRUCT2K_SOURCE_ROOT/struct2k.LLM_silver.receipt.json"
OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/classifier_research/joint_c0_bridges/$STRUCTURAL_C0_BRIDGE_RUN_ID}
for path in \
    "$STRUCT2K_LOCK" \
    "$SILVER" \
    "$SPLIT" \
    "$SOURCE_RECEIPT" \
    "$JOINT_LADDER_RUN_ROOT/run.receipt.json" \
    "$STRUCTURAL_PRODUCTION_SEQUENCE_CONFIG" \
    "$STRUCTURAL_CLASSIFIER_SELECTION_SCRIPT" \
    "$REFERENCE_BIN" \
    "$DETECTOR_BUILD_RECEIPT" \
    "$STRUCTURAL_BIB_MODEL" \
    "$STRUCTURAL_TOC_MODEL" \
    "$STRUCTURAL_SMOOTHER"; do
    phase04_require_file "$path"
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || exit 102

echo "JOINT C0 PRODUCTION-BRIDGE PLAN"
echo "  source=$STRUCT2K_SOURCE_ROOT"
echo "  ladder=$JOINT_LADDER_RUN_ROOT"
echo "  selected_architecture=c0-rust-lr-hysteresis"
echo "  compatibility=existing_Rust_implementation_requires_fresh_exact_parity"
echo "  non_c0=C1_C2_N1_require_separate_Rust_port_export_parity_package"
echo "  output=$OUTPUT_ROOT"
if [[ "${CONFIRM_STRUCTURAL_C0_BRIDGE:-0}" != "1" ]]; then
    echo "DRY_RUN_ONLY=1"
    exit 0
fi

LOCK_ROOT="$RUN_ROOT/classifier_research/joint_c0_bridge_locks"
mkdir -p "$LOCK_ROOT"
exec {RUN_LOCK_FD}>"$LOCK_ROOT/$STRUCTURAL_C0_BRIDGE_RUN_ID.lock"
flock -n "$RUN_LOCK_FD" || exit 103
PARTIAL="$OUTPUT_ROOT.partial-${SLURM_JOB_ID:?}"
[[ ! -e "$PARTIAL" && ! -L "$PARTIAL" ]] || exit 104
mkdir -p "$PARTIAL"
SELECTION="$PARTIAL/classifier_selection.receipt.json"
PARITY="$PARTIAL/c0.struct_rust_parity.receipt.json"

run_runtime_python() {
    uenv run "$PHASE04_UENV" --view=default -- "$RUNTIME_VENV/bin/python" "$@"
}
run_uenv_python() {
    uenv run "$PHASE04_UENV" --view=default -- \
        env CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        python3 "$@"
}

cd "$ACADEMIC_DIR/eval"
run_runtime_python "$PHASE04_DIR/scripts/validate_detector_build_receipt.py" \
    --receipt "$DETECTOR_BUILD_RECEIPT" \
    --repo "$REPO_ROOT" \
    --binary "$REFERENCE_BIN" \
    --cargo-lock "$ACADEMIC_DIR/reference_detector/Cargo.lock" \
    --cargo-toml "$ACADEMIC_DIR/reference_detector/Cargo.toml" >/dev/null
run_runtime_python -m sequence_models.struct2k_import verify \
    --root "$STRUCT2K_SOURCE_ROOT" \
    --lock "$STRUCT2K_LOCK" \
    --config "$STRUCTURAL_PRODUCTION_SEQUENCE_CONFIG"
run_runtime_python "$STRUCTURAL_CLASSIFIER_SELECTION_SCRIPT" build \
    --ladder-run-root "$JOINT_LADDER_RUN_ROOT" \
    --source-receipt "$SOURCE_RECEIPT" \
    --source-split-manifest "$SPLIT" \
    --sequence-config "$STRUCTURAL_PRODUCTION_SEQUENCE_CONFIG" \
    --bib-model "$STRUCTURAL_BIB_MODEL" \
    --toc-model "$STRUCTURAL_TOC_MODEL" \
    --smoother "$STRUCTURAL_SMOOTHER" \
    --selected-architecture c0-rust-lr-hysteresis \
    --selection-rationale "$STRUCTURAL_SELECTION_RATIONALE" \
    --output "$SELECTION" >/dev/null

EXPECTED_CORPUS_SHA256=$(run_runtime_python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["silver_sha256"])' "$SOURCE_RECEIPT")
EXPECTED_VALIDATION_DOCUMENTS=$(run_runtime_python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["split_counts"]["validation"])' "$SOURCE_RECEIPT")
run_uenv_python "$ACADEMIC_DIR/eval/rust_parity_struct.py" \
    --binary "$REFERENCE_BIN" \
    --corpus "$SILVER" \
    --source-receipt "$SOURCE_RECEIPT" \
    --partition validation \
    --expected-documents "$EXPECTED_VALIDATION_DOCUMENTS" \
    --expected-corpus-sha256 "$EXPECTED_CORPUS_SHA256" \
    --tolerance 0.001 \
    --receipt "$PARITY"

run_runtime_python "$STRUCTURAL_CLASSIFIER_SELECTION_SCRIPT" validate \
    --receipt "$SELECTION" \
    --source-receipt "$SOURCE_RECEIPT" \
    --source-split-manifest "$SPLIT" \
    --sequence-config "$STRUCTURAL_PRODUCTION_SEQUENCE_CONFIG" \
    --bib-model "$STRUCTURAL_BIB_MODEL" \
    --toc-model "$STRUCTURAL_TOC_MODEL" \
    --smoother "$STRUCTURAL_SMOOTHER" >/dev/null
run_runtime_python -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p["status"] == "passed" and p["evaluation_partition"] == "validation" and p["historical_test_documents_loaded"] == 0 and p["source_receipt_sha256"] == __import__("hashlib").sha256(open(sys.argv[2], "rb").read()).hexdigest()' "$PARITY" "$SOURCE_RECEIPT"

mkdir -p "$(dirname "$OUTPUT_ROOT")"
mv --no-clobber -T "$PARTIAL" "$OUTPUT_ROOT"
[[ ! -e "$PARTIAL" && -s "$OUTPUT_ROOT/classifier_selection.receipt.json" && -s "$OUTPUT_ROOT/c0.struct_rust_parity.receipt.json" ]] || exit 105
echo "STRUCTURAL_CLASSIFIER_SELECTION_RECEIPT=$OUTPUT_ROOT/classifier_selection.receipt.json"
echo "STRUCTURAL_PARITY_RECEIPT=$OUTPUT_ROOT/c0.struct_rust_parity.receipt.json"

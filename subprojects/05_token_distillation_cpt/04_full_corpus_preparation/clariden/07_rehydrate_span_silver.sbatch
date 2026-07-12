#!/usr/bin/env bash
# CPU-only classifier research: reconnect tracked Opus BIB labels to receipt-bound text.
# This script is intentionally not wired into submit.sh or the corpus-production DAG.
# Without CONFIRM_CLASSIFIER_RESEARCH=1 it prints the resolved plan and exits unchanged.
#SBATCH --job-name=cpt4_span_silver
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

HERE=${PHASE04_CLARIDEN_DIR:?set PHASE04_CLARIDEN_DIR to the checked-out clariden directory}
source "$HERE/paths.env"
source "$HERE/common.sh"

phase04_require_compute_partition "SPAN silver classifier research"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

ACQUISITION_RECEIPT=${ACQUISITION_RECEIPT:?set the passed Phase-04 acquisition receipt}
SOURCE_LOCK=${SOURCE_LOCK:?set the exact source lock named by the acquisition receipt}
GREEK_PHD_ROUTE=${GREEK_PHD_ROUTE:?choose nanochat_base, greek_phd_v2, or mdc_raw_forensic explicitly}
KALLIPOS_ROUTE=${KALLIPOS_ROUTE:?choose kallipos_sections or nanochat_base explicitly}
CLASSIFIER_RUN_ID=${CLASSIFIER_RUN_ID:?set an immutable classifier research run ID}
[[ "$CLASSIFIER_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$ ]] || {
    echo "ERROR: invalid CLASSIFIER_RUN_ID=$CLASSIFIER_RUN_ID" >&2
    exit 101
}
case "$GREEK_PHD_ROUTE" in nanochat_base|greek_phd_v2|mdc_raw_forensic) ;; *) exit 102 ;; esac
case "$KALLIPOS_ROUTE" in kallipos_sections|nanochat_base) ;; *) exit 103 ;; esac

OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/classifier_research/$CLASSIFIER_RUN_ID}
SOURCE_ARTIFACTS="$OUTPUT_ROOT/span-source-artifacts.json"
UNIT_DIR="$OUTPUT_ROOT/SPAN-rehydrated"
UNIT_RECEIPT="$OUTPUT_ROOT/SPAN-rehydrated.receipt.json"
SILVER_JSONL="$OUTPUT_ROOT/span.LLM_silver.jsonl"
SPLIT_MANIFEST="$OUTPUT_ROOT/span.LLM_silver.split.json"
SILVER_RECEIPT="$OUTPUT_ROOT/span.LLM_silver.receipt.json"

for path in "$ACQUISITION_RECEIPT" "$SOURCE_LOCK" "$SOURCE_CONFIG" "$TOKENIZER_JSON"; do
    phase04_require_file "$path"
done
if [[ "$GREEK_PHD_ROUTE" == "mdc_raw_forensic" ]]; then
    MDC_QUARANTINE_RECEIPT=${MDC_QUARANTINE_RECEIPT:?set the quarantined MDC object receipt}
    MDC_SPAN_AUDIT_RECEIPT=${MDC_SPAN_AUDIT_RECEIPT:?set the v2 MDC SPAN audit receipt}
    MDC_EXPECTED_OBSERVED_SHA256=${MDC_EXPECTED_OBSERVED_SHA256:?pin the observed quarantined archive SHA-256}
    [[ "${CONFIRM_QUARANTINED_MDC_COMPARISON_ONLY:-0}" == "1" ]] || {
        echo "ERROR: set CONFIRM_QUARANTINED_MDC_COMPARISON_ONLY=1 for the raw MDC route" >&2
        exit 106
    }
    [[ -z "${EXPECTED_SPAN_ARTIFACT_SHA256:-}" ]] || {
        echo "ERROR: quarantined MDC cannot self-pin historical snapshot equivalence" >&2
        exit 107
    }
    for path in "$MDC_QUARANTINE_RECEIPT" "$MDC_SPAN_AUDIT_RECEIPT"; do
        phase04_require_file "$path"
    done
fi
[[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "ERROR: immutable classifier output root already exists: $OUTPUT_ROOT" >&2
    exit 104
}

echo "SPAN SILVER CLASSIFIER RESEARCH PLAN"
echo "  acquisition_receipt=$ACQUISITION_RECEIPT"
echo "  source_lock=$SOURCE_LOCK"
echo "  sources_config=$SOURCE_CONFIG"
echo "  greek_phd_route=$GREEK_PHD_ROUTE"
echo "  kallipos_route=$KALLIPOS_ROUTE"
echo "  output_root=$OUTPUT_ROOT"
echo "  expected_snapshot_sha256=${EXPECTED_SPAN_ARTIFACT_SHA256:-unavailable_unverified}"
echo "  labels=new_annotations_forbidden_existing_Opus_BIB_only"
if [[ "$GREEK_PHD_ROUTE" == "mdc_raw_forensic" ]]; then
    echo "  mdc_quarantine_receipt=$MDC_QUARANTINE_RECEIPT"
    echo "  mdc_span_audit_receipt=$MDC_SPAN_AUDIT_RECEIPT"
    echo "  mdc_expected_observed_sha256=$MDC_EXPECTED_OBSERVED_SHA256"
    echo "  mdc_scope=quarantined_LLM_silver_comparison_only"
fi

if [[ "${CONFIRM_CLASSIFIER_RESEARCH:-0}" != "1" ]]; then
    echo "DRY_RUN_ONLY=1"
    echo "Set CONFIRM_CLASSIFIER_RESEARCH=1 in the sbatch export to execute on this CPU node."
    exit 0
fi

mkdir -p "$OUTPUT_ROOT"
cd "$ACADEMIC_DIR/eval"
PYTHON="$RUNTIME_VENV/bin/python"
run_python() {
    uenv run "$PHASE04_UENV" --view=default -- "$PYTHON" "$@"
}

builder_args=(
    -m sequence_models.build_span_source_receipt
    --acquisition-receipt "$ACQUISITION_RECEIPT"
    --source-lock "$SOURCE_LOCK"
    --sources-config "$SOURCE_CONFIG"
    --greek-phd-route "$GREEK_PHD_ROUTE"
    --kallipos-route "$KALLIPOS_ROUTE"
    --output "$SOURCE_ARTIFACTS"
)
if [[ "$GREEK_PHD_ROUTE" == "greek_phd_v2" ]]; then
    [[ -n "${GREEK_PHD_DOCUMENT_ID_COLUMN:-}" ]] || {
        echo "ERROR: v2 requires GREEK_PHD_DOCUMENT_ID_COLUMN and explicit ID-domain override" >&2
        exit 105
    }
    builder_args+=(
        --greek-phd-document-id-column "$GREEK_PHD_DOCUMENT_ID_COLUMN"
        --allow-unverified-greek-phd-id-domain
    )
    for column in ${GREEK_PHD_TEXT_COLUMNS:-}; do
        builder_args+=(--greek-phd-text-column "$column")
    done
elif [[ "$GREEK_PHD_ROUTE" == "mdc_raw_forensic" ]]; then
    builder_args+=(
        --mdc-quarantine-receipt "$MDC_QUARANTINE_RECEIPT"
        --mdc-span-audit-receipt "$MDC_SPAN_AUDIT_RECEIPT"
        --mdc-expected-observed-sha256 "$MDC_EXPECTED_OBSERVED_SHA256"
        --allow-quarantined-mdc-comparison-only
    )
fi
run_python "${builder_args[@]}"

rehydrate_args=(
    -m sequence_models.silver_reconstruct rehydrate-span-units
    --source-artifacts "$SOURCE_ARTIFACTS"
    --output-dir "$UNIT_DIR"
    --receipt "$UNIT_RECEIPT"
)
if [[ -n "${EXPECTED_SPAN_ARTIFACT_SHA256:-}" ]]; then
    rehydrate_args+=(--expected-artifact-sha256 "$EXPECTED_SPAN_ARTIFACT_SHA256")
fi
run_python "${rehydrate_args[@]}"

run_python -m sequence_models.silver_reconstruct hydrate-span \
    --unit-dir "$UNIT_DIR" \
    --unit-rehydration-receipt "$UNIT_RECEIPT" \
    --tokenizer-json "$TOKENIZER_JSON" \
    --output "$SILVER_JSONL" \
    --split-manifest "$SPLIT_MANIFEST" \
    --receipt "$SILVER_RECEIPT"

for path in "$SOURCE_ARTIFACTS" "$UNIT_RECEIPT" "$SILVER_JSONL" "$SPLIT_MANIFEST" "$SILVER_RECEIPT"; do
    phase04_require_file "$path"
done
echo "SPAN_SILVER_COMPLETE=$OUTPUT_ROOT"

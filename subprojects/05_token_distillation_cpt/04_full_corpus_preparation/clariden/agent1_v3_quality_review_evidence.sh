#!/usr/bin/env bash
# CPU-only Stage 35 closure for compact local Codex review evidence.
#
# Stage 30 materializes masked requests and performs its mandatory full scan,
# but intentionally does not invoke a model.  This handler imports only the
# locally-produced response evidence bundle, proves closure, then runs the
# richer GlossAPI Rust diagnostic over the exact masked primary sample.  It
# never decides source admission or mutates canonical corpus data.
set -euo pipefail

HERE=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/agent1_v3_paths.env"
source "$HERE/agent1_v3_common.sh"

readonly STAGE="35-quality-review-evidence"
readonly REVIEW_PACKET_STAGE="30-review-packet"
readonly NORMALIZE_STAGE="10-normalize"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_file() {
    local label=$1 path=$2
    [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, or a symlink: $path"
}

require_directory() {
    local label=$1 path=$2
    [[ -d "$path" && ! -L "$path" ]] || die "$label is missing or a symlink: $path"
}

require_positive_integer() {
    local label=$1 value=$2
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$label must be a positive integer: $value"
}

assert_under() {
    local label=$1 path=$2 root=$3
    case "$path" in
        "$root"/*) ;;
        *) die "$label must remain under $root: $path" ;;
    esac
}

assert_under_one_of() {
    local label=$1 path=$2 first_root=$3 second_root=$4
    case "$path" in
        "$first_root"/*|"$second_root"/*) ;;
        *) die "$label must remain under $first_root or $second_root: $path" ;;
    esac
}

run_python() {
    uenv run "$AGENT1_V3_UENV" --view=default -- \
        "$AGENT1_V3_RUNTIME_VENV/bin/python" "$@"
}

stage_attempt_dir() {
    local stage=$1 storage=$2 path expected
    path=$(run_python "$AGENT1_V3_CONTRACT_SCRIPT" get-stage-attempt-dir \
        --run-root "$AGENT1_V3_RUN_ROOT" --run-id "$AGENT1_V3_RUN_ID" \
        --stage "$stage" --storage "$storage")
    case "$storage" in
        metadata) expected="$AGENT1_V3_RUN_ROOT/stages/$stage/attempts" ;;
        data) expected="$AGENT1_V3_DATA_ROOT/stages/$stage/attempts" ;;
        *) die "unsupported stage storage selector: $storage" ;;
    esac
    assert_under "upstream $stage $storage attempt" "$path" "$expected"
    require_directory "upstream $stage $storage attempt" "$path"
    printf '%s\n' "$path"
}

stage_output() {
    local stage=$1 basename=$2 path metadata_root data_root
    path=$(run_python "$AGENT1_V3_CONTRACT_SCRIPT" get-stage-output \
        --run-root "$AGENT1_V3_RUN_ROOT" --run-id "$AGENT1_V3_RUN_ID" \
        --stage "$stage" --basename "$basename")
    metadata_root=$(stage_attempt_dir "$stage" metadata)
    data_root=$(stage_attempt_dir "$stage" data)
    assert_under_one_of "upstream $stage output" "$path" "$metadata_root" "$data_root"
    require_file "upstream $stage output" "$path"
    printf '%s\n' "$path"
}

verify_frozen_glossapi_runtime() {
    local expected_runtime expected_receipt expected_modules
    expected_runtime="$AGENT1_V3_DATA_ROOT/runtime/glossapi-rust-quality-$AGENT1_V3_GLOSSAPI_COMMIT"
    expected_receipt="$expected_runtime/build_receipt.json"
    expected_modules="$expected_runtime/modules"
    [[ "$AGENT1_V3_GLOSSAPI_RUNTIME_ROOT" == "$expected_runtime" ]] || die "GlossAPI runtime root drift"
    [[ "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" == "$expected_receipt" ]] || die "GlossAPI build receipt path drift"
    [[ "$AGENT1_V3_GLOSSAPI_MODULE_DIR" == "$expected_modules" ]] || die "GlossAPI module directory path drift"
    require_file "Phase-0 GlossAPI build receipt" "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT"
    require_directory "Phase-0 GlossAPI module directory" "$AGENT1_V3_GLOSSAPI_MODULE_DIR"
    run_python - \
        "$AGENT1_V3_RUN_ROOT/run_contract.json" \
        "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract_path, receipt_path = map(Path, sys.argv[1:])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
binding = contract.get("inputs", {}).get("glossapi_build_receipt")
if not isinstance(binding, dict):
    raise SystemExit("run contract lacks frozen GlossAPI build receipt")
receipt = receipt_path.resolve()
digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
if (
    Path(str(binding.get("path", ""))).resolve() != receipt
    or binding.get("bytes") != receipt.stat().st_size
    or binding.get("sha256") != digest
):
    raise SystemExit("Phase-0 GlossAPI build receipt differs from frozen run contract")
PY
    (
        export PYTHONPATH="$AGENT1_V3_GLOSSAPI_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
        run_python "$PHASE04_DIR/scripts/profile_dataset_quality_rust.py" validate-build-receipt \
            --receipt "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" \
            --expected-commit "$AGENT1_V3_GLOSSAPI_COMMIT"
    )
}

validate_external_bundle_paths() {
    local root=${AGENT1_V3_EXTERNAL_REVIEW_EVIDENCE_DIR:-}
    [[ -n "$root" ]] || die "AGENT1_V3_EXTERNAL_REVIEW_EVIDENCE_DIR is required after local Codex review"
    require_directory "external review evidence directory" "$root"
    # The Python verifier enforces the exact five-file no-raw-corpus layout.
    # These shell checks make every external artifact an explicit Stage-35
    # contract input before the immutable attempt directory is created.
    require_file "external evidence manifest" "$root/external_review_evidence_manifest.json"
    require_file "external review responses" "$root/responses.jsonl"
    require_file "external response execution receipt" "$root/response_execution_receipt.json"
    require_file "external adjudication execution receipt" "$root/adjudication_execution_receipt.json"
    require_file "external passed calibration receipt" "$root/calibration_receipt.json"
    printf '%s\n' "$root"
}

main() {
    [[ $# -eq 0 ]] || die "agent1_v3_quality_review_evidence.sh accepts no arguments"
    agent1_v3_init_paths
    agent1_v3_require_clean_commit
    agent1_v3_require_runtime

    local requests packet_manifest normalization_manifest external_dir
    requests=$(stage_output "$REVIEW_PACKET_STAGE" "review_requests.jsonl")
    packet_manifest=$(stage_output "$REVIEW_PACKET_STAGE" "review_packet_manifest.json")
    normalization_manifest=$(stage_output "$NORMALIZE_STAGE" "normalization_manifest.json")
    external_dir=$(validate_external_bundle_paths)
    verify_frozen_glossapi_runtime
    require_file "review policy" "$AGENT1_V3_REVIEW_POLICY"
    require_file "review prompt" "$AGENT1_V3_REVIEW_PROMPT"
    require_file "review response schema" "$AGENT1_V3_REVIEW_RESPONSE_SCHEMA"
    require_file "Stage 30 review requests" "$requests"
    require_file "Stage 30 packet manifest" "$packet_manifest"
    require_file "normalization manifest" "$normalization_manifest"

    local cpu_count quality_threads quality_batch_size quantile_sample_size
    cpu_count=${SLURM_CPUS_PER_TASK:-}
    quality_threads=${AGENT1_V3_REVIEW_SAMPLE_QUALITY_THREADS:-${SLURM_CPUS_PER_TASK:-}}
    quality_batch_size=${AGENT1_V3_REVIEW_SAMPLE_QUALITY_BATCH_SIZE:-128}
    quantile_sample_size=${AGENT1_V3_REVIEW_SAMPLE_QUALITY_QUANTILE_SAMPLE_SIZE:-1024}
    require_positive_integer "SLURM_CPUS_PER_TASK" "$cpu_count"
    require_positive_integer "AGENT1_V3_REVIEW_SAMPLE_QUALITY_THREADS" "$quality_threads"
    require_positive_integer "AGENT1_V3_REVIEW_SAMPLE_QUALITY_BATCH_SIZE" "$quality_batch_size"
    require_positive_integer "AGENT1_V3_REVIEW_SAMPLE_QUALITY_QUANTILE_SAMPLE_SIZE" "$quantile_sample_size"
    (( quality_threads <= cpu_count )) || die "review-sample quality threads exceed allocated CPUs"
    (( quantile_sample_size >= 100 )) || die "review-sample quantile sample size must be at least 100"

    agent1_v3_begin_stage "$STAGE" \
        --parameters-json "{\"executor\":\"agent1_v3_quality_review_evidence.sh\",\"external_bundle\":\"responses_and_execution_receipts_only\",\"review_sample_text_variant\":\"high_precision_identifier_masked_review_sample\",\"quality_threads\":$quality_threads,\"quality_batch_size\":$quality_batch_size,\"quality_quantile_sample_size\":$quantile_sample_size}" \
        --input review_requests "$requests" \
        --input review_packet_manifest "$packet_manifest" \
        --input normalization_manifest "$normalization_manifest" \
        --input review_policy "$AGENT1_V3_REVIEW_POLICY" \
        --input review_prompt "$AGENT1_V3_REVIEW_PROMPT" \
        --input review_response_schema "$AGENT1_V3_REVIEW_RESPONSE_SCHEMA" \
        --input glossapi_build_receipt "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" \
        --input external_evidence_manifest "$external_dir/external_review_evidence_manifest.json" \
        --input external_responses "$external_dir/responses.jsonl" \
        --input external_response_receipt "$external_dir/response_execution_receipt.json" \
        --input external_adjudication_receipt "$external_dir/adjudication_execution_receipt.json" \
        --input external_calibration_receipt "$external_dir/calibration_receipt.json"
    # begin-stage redirects allocation evidence to the new Capstor attempt.
    agent1_v3_require_compute_cpu
    agent1_v3_mask_gpu_visibility
    mkdir -p "$AGENT1_V3_DATA_ATTEMPT_DIR/tmp" "$AGENT1_V3_DATA_ATTEMPT_DIR/work"
    export TMPDIR="$AGENT1_V3_DATA_ATTEMPT_DIR/tmp" PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=4

    local evidence_script imported_dir import_receipt closure sample sample_receipt
    local quality_root quality_summary quality_handoff
    evidence_script="$PHASE04_DIR/scripts/agent1_v3_review_evidence.py"
    require_file "Stage 35 review evidence tool" "$evidence_script"
    imported_dir="$AGENT1_V3_ATTEMPT_DIR/imported-external-review-evidence"
    import_receipt="$AGENT1_V3_ATTEMPT_DIR/external_review_evidence_import.json"
    closure="$AGENT1_V3_ATTEMPT_DIR/quality_review_evidence_closure.json"
    sample="$AGENT1_V3_DATA_ATTEMPT_DIR/masked-review-sample.jsonl"
    sample_receipt="$AGENT1_V3_ATTEMPT_DIR/masked-review-sample-receipt.json"
    quality_root="$AGENT1_V3_DATA_ATTEMPT_DIR/masked-review-sample-quality"
    quality_summary="$AGENT1_V3_ATTEMPT_DIR/masked-review-sample-quality-summary.json"
    quality_handoff="$AGENT1_V3_ATTEMPT_DIR/masked-review-sample-quality-handoff.json"

    run_python "$evidence_script" import-external \
        --external-evidence-dir "$external_dir" \
        --destination "$imported_dir" \
        --receipt "$import_receipt"
    run_python "$evidence_script" validate-closure \
        --run-id "$AGENT1_V3_RUN_ID" \
        --requests "$requests" \
        --packet-manifest "$packet_manifest" \
        --external-evidence-dir "$imported_dir" \
        --policy "$AGENT1_V3_REVIEW_POLICY" \
        --prompt "$AGENT1_V3_REVIEW_PROMPT" \
        --response-schema "$AGENT1_V3_REVIEW_RESPONSE_SCHEMA" \
        --code-commit "$AGENT1_V3_EXPECTED_COMMIT" \
        --output "$closure"
    run_python "$evidence_script" materialize-masked-sample \
        --requests "$requests" \
        --packet-manifest "$packet_manifest" \
        --closure "$closure" \
        --output "$sample" \
        --receipt "$sample_receipt"
    (
        export PYTHONPATH="$AGENT1_V3_GLOSSAPI_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
        run_python "$PHASE04_DIR/scripts/agent1_v3_masked_review_sample_quality.py" \
            --sample "$sample" \
            --sample-receipt "$sample_receipt" \
            --build-receipt "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" \
            --expected-commit "$AGENT1_V3_GLOSSAPI_COMMIT" \
            --output-dir "$quality_root" \
            --scratch-dir "$AGENT1_V3_DATA_ATTEMPT_DIR/work/masked-review-sample-quality" \
            --summary "$quality_summary" \
            --handoff "$quality_handoff" \
            --batch-size "$quality_batch_size" \
            --threads "$quality_threads" \
            --quantile-sample-size "$quantile_sample_size"
    )

    local imported_manifest imported_responses imported_response_receipt imported_adjudication_receipt imported_calibration_receipt
    imported_manifest="$imported_dir/external_review_evidence_manifest.json"
    imported_responses="$imported_dir/responses.jsonl"
    imported_response_receipt="$imported_dir/response_execution_receipt.json"
    imported_adjudication_receipt="$imported_dir/adjudication_execution_receipt.json"
    imported_calibration_receipt="$imported_dir/calibration_receipt.json"
    for path in \
        "$import_receipt" "$imported_manifest" "$imported_responses" \
        "$imported_response_receipt" "$imported_adjudication_receipt" "$imported_calibration_receipt" "$closure" \
        "$sample" "$sample_receipt" "$quality_root/contract.json" \
        "$quality_root/dataset_quality_document_v1.parquet" "$quality_summary" \
        "$quality_handoff" "$AGENT1_V3_ALLOCATION_EVIDENCE"; do
        require_file "Stage 35 output" "$path"
    done
    agent1_v3_finish_stage "$STAGE" \
        --output "$import_receipt" \
        --output "$imported_manifest" \
        --output "$imported_responses" \
        --output "$imported_response_receipt" \
        --output "$imported_adjudication_receipt" \
        --output "$imported_calibration_receipt" \
        --output "$closure" \
        --output "$sample" \
        --output "$sample_receipt" \
        --output "$quality_root/contract.json" \
        --output "$quality_root/dataset_quality_document_v1.parquet" \
        --output "$quality_summary" \
        --output "$quality_handoff" \
        --output "$AGENT1_V3_ALLOCATION_EVIDENCE"
    echo "REVIEW_EVIDENCE_CLOSURE=$closure"
    echo "ADMISSION_DECISION=not_evaluated_in_stage35"
}

main "$@"

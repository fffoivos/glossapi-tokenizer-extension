#!/usr/bin/env bash
# Dry-run-first submission/status/resume helper for the Phase-04 CPU DAG.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PHASE04_DIR=$(cd "$HERE/.." && pwd)
REPO_ROOT=$(git -C "$PHASE04_DIR" rev-parse --show-toplevel)
ACADEMIC_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic"
PHASE04_EXPECTED_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
source "$HERE/paths.env"

usage() {
    cat >&2 <<'EOF'
usage:
  submit.sh <stage> [sbatch args...]
  submit.sh resume <stage> [sbatch args...]
  submit.sh status [PIPELINE_RUN_ID]
  submit.sh chain-to-review
  submit.sh chain-after-admission
  submit.sh chain-after-post-clean
  submit.sh chain-structural-to-audit
  submit.sh chain-finalize-noop
  submit.sh chain-finalize-promoted

Corpus stages:
  normalize lineage review-packet review-aggregate clean post-clean-packet
  post-clean-aggregate structural-stage50-detect structural-review-packet
  structural-promote final-clean greekmmlu-freeze decontam dedup materialize publish

Acquisition stages:
  acquire acquire-mdc

Legacy/audit stages:
  bootstrap-runtime build-detector acquire quality structural-detect structural-token-loss

All submissions are dry runs unless CONFIRM_LAUNCH=1.  The cleaning chains stop
before irreversible Stage58 finalization.  Finalization is a separate explicit
choice between a confirmed structural no-op and exact Stage54-promoted spans.
Publication is never part of a chain.
EOF
}

stage_script() {
    case "$1" in
        bootstrap-runtime) echo "$HERE/04_bootstrap_runtime.sbatch" ;;
        build-detector) echo "$HERE/05_build_detector.sbatch" ;;
        acquire) echo "$HERE/00_acquire_sources.sbatch" ;;
        acquire-mdc) echo "$HERE/02_acquire_mdc_sources.sbatch" ;;
        quality) echo "$HERE/10_quality_audit.sbatch" ;;
        structural-detect) echo "$HERE/20_structural_detect.sbatch" ;;
        structural-token-loss) echo "$HERE/30_structural_token_loss.sbatch" ;;
        normalize|10-normalize) echo "$HERE/40_normalize_sources.sbatch" ;;
        lineage|20-lineage) echo "$HERE/42_build_lineage.sbatch" ;;
        review-packet|30-review-packet) echo "$HERE/44_build_review_packet.sbatch" ;;
        review-aggregate|40-review-aggregate) echo "$HERE/46_aggregate_reviews.sbatch" ;;
        clean|50-clean) echo "$HERE/60_apply_cleaning.sbatch" ;;
        structural-stage50-detect|52-structural-detect) echo "$HERE/52_detect_stage50_structural_spans.sbatch" ;;
        structural-review-packet|53-structural-review-packet) echo "$HERE/53_build_structural_review_packet.sbatch" ;;
        structural-promote|54-structural-promote) echo "$HERE/54_promote_structural_spans.sbatch" ;;
        post-clean-packet|55-post-clean-review-packet) echo "$HERE/62_build_post_clean_review_packet.sbatch" ;;
        post-clean-aggregate|56-post-clean-review-aggregate) echo "$HERE/64_aggregate_post_clean_reviews.sbatch" ;;
        final-clean|58-final-clean) echo "$HERE/66_finalize_cleaning.sbatch" ;;
        greekmmlu-freeze|59-greekmmlu-freeze) echo "$HERE/68_freeze_greekmmlu.sbatch" ;;
        decontam|60-greekmmlu-decontam) echo "$HERE/70_greekmmlu_decontam.sbatch" ;;
        dedup|70-dedup) echo "$HERE/80_dedup.sbatch" ;;
        materialize|80-materialize-validate) echo "$HERE/90_materialize_validate.sbatch" ;;
        publish|90-publish) echo "$HERE/99_publish_hf.sbatch" ;;
        *) return 2 ;;
    esac
}

canonical_stage() {
    case "$1" in
        normalize|10-normalize) echo 10-normalize ;;
        lineage|20-lineage) echo 20-lineage ;;
        review-packet|30-review-packet) echo 30-review-packet ;;
        review-aggregate|40-review-aggregate) echo 40-review-aggregate ;;
        clean|50-clean) echo 50-clean ;;
        structural-stage50-detect|52-structural-detect) echo 52-structural-detect ;;
        structural-review-packet|53-structural-review-packet) echo 53-structural-review-packet ;;
        structural-promote|54-structural-promote) echo 54-structural-promote ;;
        post-clean-packet|55-post-clean-review-packet) echo 55-post-clean-review-packet ;;
        post-clean-aggregate|56-post-clean-review-aggregate) echo 56-post-clean-review-aggregate ;;
        final-clean|58-final-clean) echo 58-final-clean ;;
        greekmmlu-freeze|59-greekmmlu-freeze) echo 59-greekmmlu-freeze ;;
        decontam|60-greekmmlu-decontam) echo 60-greekmmlu-decontam ;;
        dedup|70-dedup) echo 70-dedup ;;
        materialize|80-materialize-validate) echo 80-materialize-validate ;;
        publish|90-publish) echo 90-publish ;;
        *) echo "$1" ;;
    esac
}

require_clean_live_checkout() {
    [[ "${CONFIRM_LAUNCH:-0}" == "1" ]] || return 0
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]]; then
        echo "ERROR: live Phase-04 submission requires a clean exact checkout: $REPO_ROOT" >&2
        git -C "$REPO_ROOT" status --short >&2
        exit 3
    fi
}

require_pipeline_id() {
    [[ -n "${PIPELINE_RUN_ID:-}" ]] || {
        echo "ERROR: export an immutable PIPELINE_RUN_ID before corpus-stage submission." >&2
        exit 4
    }
}

file_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

final_admission_preflight() {
    [[ "${CONFIRM_LAUNCH:-0}" == "1" ]] || return 0
    [[ -s "${FINAL_SOURCE_ADMISSION:-}" && -n "${CONFIRM_FINAL_ADMISSION_SHA256:-}" ]] || {
        echo "ERROR: final cleaning requires FINAL_SOURCE_ADMISSION and CONFIRM_FINAL_ADMISSION_SHA256." >&2
        exit 7
    }
    [[ "$(file_sha256 "$FINAL_SOURCE_ADMISSION")" == "$CONFIRM_FINAL_ADMISSION_SHA256" ]] || {
        echo "ERROR: CONFIRM_FINAL_ADMISSION_SHA256 does not match FINAL_SOURCE_ADMISSION." >&2
        exit 7
    }
    [[ "${REUSE_STAGE50_ADMISSION+x}" == "x" ]] && \
        [[ "$REUSE_STAGE50_ADMISSION" == "0" || "$REUSE_STAGE50_ADMISSION" == "1" ]] || {
        echo "ERROR: final cleaning requires explicit REUSE_STAGE50_ADMISSION=0 or 1." >&2
        exit 7
    }
}

manual_preflight() {
    local target=$1
    [[ "${CONFIRM_LAUNCH:-0}" == "1" ]] || return 0
    case "$target" in
        acquire)
            [[ "${CONFIRM_ACQUIRE:-0}" == "1" ]] || {
                echo "ERROR: set CONFIRM_ACQUIRE=1 for an intentional acquisition or existing-payload verification run." >&2
                exit 5
            }
            [[ -n "${HF_TOKEN:-}" ]] || {
                echo "ERROR: HF_TOKEN is required to resolve exact Hugging Face content identifiers." >&2
                exit 5
            }
            ;;
        acquire-mdc)
            [[ -n "${MOZILLA_DATA_COLLECTIVE_API_KEY:-}" ]] || {
                echo "ERROR: MOZILLA_DATA_COLLECTIVE_API_KEY is required in the submission environment." >&2
                exit 5
            }
            ;;
        review-aggregate|40-review-aggregate)
            [[ -s "${REVIEWS_JSONL:-}" ]] || { echo "ERROR: REVIEWS_JSONL is missing/empty." >&2; exit 6; }
            ;;
        clean|50-clean)
            [[ -s "${SOURCE_ADMISSION:-}" && -n "${CONFIRM_ADMISSION_SHA256:-}" ]] || {
                echo "ERROR: cleaning requires SOURCE_ADMISSION and its manually inspected CONFIRM_ADMISSION_SHA256." >&2
                exit 7
            }
            [[ "$(file_sha256 "$SOURCE_ADMISSION")" == "$CONFIRM_ADMISSION_SHA256" ]] || {
                echo "ERROR: CONFIRM_ADMISSION_SHA256 does not match SOURCE_ADMISSION." >&2
                exit 7
            }
            ;;
        post-clean-aggregate|56-post-clean-review-aggregate)
            [[ -s "${POST_CLEAN_REVIEWS_JSONL:-}" ]] || {
                echo "ERROR: POST_CLEAN_REVIEWS_JSONL is missing/empty." >&2
                exit 7
            }
            ;;
        structural-stage50-detect|52-structural-detect)
            for required in \
                STRUCTURAL_CLASSIFIER_SELECTION_RECEIPT \
                STRUCTURAL_SILVER_RECEIPT STRUCTURAL_SILVER_SPLIT; do
                [[ -s "${!required:-}" ]] || {
                    echo "ERROR: structural detection requires the post-ladder $required." >&2
                    exit 7
                }
            done
            ;;
        final-clean|58-final-clean)
            final_admission_preflight
            [[ "${APPLY_STRUCTURAL+x}" == "x" ]] && \
                [[ "$APPLY_STRUCTURAL" == "0" || "$APPLY_STRUCTURAL" == "1" ]] || {
                echo "ERROR: final cleaning requires explicit APPLY_STRUCTURAL=0 or 1." >&2
                exit 7
            }
            if [[ "$APPLY_STRUCTURAL" == "0" ]]; then
                [[ "${CONFIRM_STRUCTURAL_NOOP:-0}" == "1" ]] || {
                    echo "ERROR: explicit no-op finalization requires CONFIRM_STRUCTURAL_NOOP=1." >&2
                    exit 7
                }
                [[ -z "${STRUCTURAL_MODEL_RECEIPT:-}" && -z "${STRUCTURAL_SPANS:-}" ]] || {
                    echo "ERROR: no-op finalization must not carry structural receipt/span paths." >&2
                    exit 7
                }
            else
                [[ -s "${STRUCTURAL_MODEL_RECEIPT:-}" && -s "${STRUCTURAL_SPANS:-}" ]] || {
                    echo "ERROR: promoted finalization requires Stage54 STRUCTURAL_MODEL_RECEIPT and STRUCTURAL_SPANS." >&2
                    exit 7
                }
                [[ -n "${CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256:-}" ]] || {
                    echo "ERROR: promoted finalization requires CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256." >&2
                    exit 7
                }
                [[ "$(file_sha256 "$STRUCTURAL_MODEL_RECEIPT")" == \
                    "$CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256" ]] || {
                    echo "ERROR: promoted structural model receipt confirmation hash mismatch." >&2
                    exit 7
                }
            fi
            ;;
        structural-promote|54-structural-promote)
            for required in \
                STRUCTURAL_MANUAL_AUDIT_RECEIPT STRUCTURAL_AUDIT_ANNOTATIONS \
                STRUCTURAL_SILVER_RECEIPT STRUCTURAL_SILVER_SPLIT \
                STRUCTURAL_CLASSIFIER_SELECTION_RECEIPT STRUCTURAL_PARITY_RECEIPT; do
                [[ -s "${!required:-}" ]] || {
                    echo "ERROR: structural promotion requires a non-empty $required." >&2
                    exit 7
                }
            done
            ;;
        publish|90-publish)
            [[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: HF_TOKEN is not in the submission environment." >&2; exit 8; }
            [[ "${CONFIRM_PUBLISH:-}" == "$HF_REPO_ID" ]] || {
                echo "ERROR: set CONFIRM_PUBLISH exactly to $HF_REPO_ID." >&2
                exit 8
            }
            ;;
    esac
}

print_command() {
    printf 'COMMAND:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
}

submit_one() {
    local target=$1
    local resume=$2
    local dependency=$3
    shift 3
    local script
    script=$(stage_script "$target") || { echo "ERROR: unknown stage: $target" >&2; exit 2; }
    manual_preflight "$target"
    local export_spec="ALL,REPO_ROOT=$REPO_ROOT,PHASE04_DIR=$PHASE04_DIR,ACADEMIC_DIR=$ACADEMIC_DIR,PHASE04_CLARIDEN_DIR=$HERE,PHASE04_EXPECTED_COMMIT=$PHASE04_EXPECTED_COMMIT"
    if [[ -n "${PIPELINE_RUN_ID:-}" ]]; then
        export_spec+=",PIPELINE_RUN_ID=$PIPELINE_RUN_ID,PIPELINE_RUNS_ROOT=$PIPELINE_RUNS_ROOT"
    fi
    [[ "$resume" == "1" ]] && export_spec+=",RESUME_STAGE=1"
    local command=(sbatch --parsable)
    [[ -n "$dependency" ]] && command+=(--dependency="$dependency")
    command+=("$@" --export="$export_spec" "$script")
    print_command "${command[@]}"
    if [[ "${CONFIRM_LAUNCH:-0}" != "1" ]]; then
        echo "DRY-${target//[^A-Za-z0-9]/_}"
        return 0
    fi
    "${command[@]}" | cut -d';' -f1
}

show_status() {
    local run_id=${1:-${PIPELINE_RUN_ID:-}}
    [[ -n "$run_id" ]] || { echo "ERROR: status needs PIPELINE_RUN_ID." >&2; exit 9; }
    local root="$PIPELINE_RUNS_ROOT/$run_id"
    echo "PIPELINE_RUN_ROOT=$root"
    local stage state
    for stage in \
        10-normalize 20-lineage 30-review-packet 40-review-aggregate \
        50-clean 55-post-clean-review-packet 56-post-clean-review-aggregate \
        52-structural-detect 53-structural-review-packet 54-structural-promote \
        58-final-clean 59-greekmmlu-freeze 60-greekmmlu-decontam 70-dedup 80-materialize-validate 90-publish; do
        if [[ -s "$root/stages/$stage/stage_receipt.json" && -s "$root/stages/$stage/COMPLETED" ]]; then
            state=COMPLETE
        elif [[ -e "$root/stages/$stage" ]]; then
            state=INCOMPLETE
        else
            state=NOT_STARTED
        fi
        printf '%-28s %s\n' "$stage" "$state"
    done
    if command -v squeue >/dev/null 2>&1; then
        echo
        squeue -u "$USER" -o '%.18i %.24j %.9T %.10M %.20R' | sed -n '1p;/cpt4_/p'
    fi
}

chain_to_review() {
    require_pipeline_id
    [[ -n "${ACQUISITION_RECEIPT:-}" ]] || {
        echo "ERROR: set ACQUISITION_RECEIPT to a passed current-config acquisition receipt." >&2
        exit 10
    }
    if [[ "${CONFIRM_LAUNCH:-0}" == "1" && ! -s "$ACQUISITION_RECEIPT" && -z "${ACQUISITION_JOB_ID:-}" ]]; then
        echo "ERROR: acquisition receipt does not exist yet; set ACQUISITION_JOB_ID to the active receipt-building job for an afterok dependency." >&2
        exit 10
    fi
    local dependency=""
    [[ -n "${ACQUISITION_JOB_ID:-}" ]] && dependency="afterok:$ACQUISITION_JOB_ID"
    local normalize lineage packet
    normalize=$(submit_one normalize 0 "$dependency")
    lineage=$(submit_one lineage 0 "afterok:$normalize")
    packet=$(submit_one review-packet 0 "afterok:$lineage")
    echo "normalize_job=$normalize"
    echo "lineage_job=$lineage"
    echo "review_packet_job=$packet"
    echo "STOP_BOUNDARY=review $PIPELINE_RUNS_ROOT/$PIPELINE_RUN_ID/stages/30-review-packet/requests.jsonl"
}

chain_after_admission() {
    require_pipeline_id
    [[ -s "${SOURCE_ADMISSION:-}" ]] || {
        echo "ERROR: set SOURCE_ADMISSION to the reviewed admission JSON before planning this chain." >&2
        exit 7
    }
    manual_preflight clean
    local needs_post_clean=0
    admission_needs_post_clean "$SOURCE_ADMISSION" && needs_post_clean=1
    local clean post_clean_packet
    clean=$(submit_one clean 0 "")
    if [[ "$needs_post_clean" == "1" ]]; then
        post_clean_packet=$(submit_one post-clean-packet 0 "afterok:$clean")
        echo "clean_job=$clean"
        echo "post_clean_review_packet_job=$post_clean_packet"
        echo "STOP_BOUNDARY=post-clean review $PIPELINE_RUNS_ROOT/$PIPELINE_RUN_ID/stages/55-post-clean-review-packet/requests.jsonl"
        return 0
    fi
    echo "clean_job=$clean"
    echo "STOP_BOUNDARY=Stage50 complete; structural audit/promotion and finalization choice remain open"
    echo "No Stage58 or downstream job was submitted. After Stage50 completes, choose chain-finalize-noop or chain-finalize-promoted explicitly."
}

admission_needs_post_clean() {
    python3 - "$1" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
raise SystemExit(0 if any(row.get("decision") == "include_after_cleaning" for row in value.get("sources", [])) else 1)
PY
}

chain_after_post_clean() {
    require_pipeline_id
    final_admission_preflight
    if [[ "${CONFIRM_LAUNCH:-0}" == "1" && "${REUSE_STAGE50_ADMISSION:-}" != "0" ]]; then
        echo "ERROR: post-clean terminal admission requires REUSE_STAGE50_ADMISSION=0." >&2
        exit 7
    fi
    echo "STOP_BOUNDARY=terminal post-clean admission confirmed; structural finalization choice remains open"
    echo "No job was submitted. Choose chain-finalize-noop or chain-finalize-promoted explicitly."
}

chain_finalize() {
    require_pipeline_id
    manual_preflight final-clean
    local final_clean greekmmlu decontam dedup materialize
    final_clean=$(submit_one final-clean 0 "")
    greekmmlu=$(submit_one greekmmlu-freeze 0 "")
    export FINAL_CLEAN_STAGE=58-final-clean
    decontam=$(submit_one decontam 0 "afterok:$final_clean:$greekmmlu")
    dedup=$(submit_one dedup 0 "afterok:$decontam")
    materialize=$(submit_one materialize 0 "afterok:$dedup")
    echo "final_clean_job=$final_clean"
    echo "greekmmlu_freeze_job=$greekmmlu"
    echo "decontam_job=$decontam"
    echo "dedup_job=$dedup"
    echo "materialize_job=$materialize"
    echo "STOP_BOUNDARY=validated local release; publication was not submitted"
}

chain_finalize_noop() {
    if [[ "${CONFIRM_LAUNCH:-0}" == "1" && "${CONFIRM_STRUCTURAL_NOOP:-0}" != "1" ]]; then
        echo "ERROR: live no-op finalization requires CONFIRM_STRUCTURAL_NOOP=1." >&2
        exit 7
    fi
    export APPLY_STRUCTURAL=0
    unset STRUCTURAL_MODEL_RECEIPT STRUCTURAL_SPANS
    chain_finalize
}

chain_finalize_promoted() {
    require_pipeline_id
    local promoted_dir="$PIPELINE_RUNS_ROOT/$PIPELINE_RUN_ID/stages/54-structural-promote"
    export APPLY_STRUCTURAL=1
    export STRUCTURAL_MODEL_RECEIPT="${STRUCTURAL_MODEL_RECEIPT:-$promoted_dir/academic_structural_model_receipt.json}"
    export STRUCTURAL_SPANS="${STRUCTURAL_SPANS:-$promoted_dir/structural_spans.jsonl}"
    if [[ "${CONFIRM_LAUNCH:-0}" == "1" ]]; then
        [[ -s "$STRUCTURAL_MODEL_RECEIPT" && -s "$STRUCTURAL_SPANS" ]] || {
            echo "ERROR: live promoted finalization requires the completed Stage54 receipt and spans." >&2
            exit 7
        }
        [[ -n "${CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256:-}" ]] || {
            echo "ERROR: live promoted finalization requires CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256." >&2
            exit 7
        }
        [[ "$(file_sha256 "$STRUCTURAL_MODEL_RECEIPT")" == \
            "$CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256" ]] || {
            echo "ERROR: confirmed structural model-receipt SHA-256 does not match Stage54." >&2
            exit 7
        }
    fi
    chain_finalize
}

chain_structural_to_audit() {
    require_pipeline_id
    local detect packet
    detect=$(submit_one structural-stage50-detect 0 "")
    packet=$(submit_one structural-review-packet 0 "afterok:$detect")
    echo "structural_detect_job=$detect"
    echo "structural_review_packet_job=$packet"
    echo "STOP_BOUNDARY=targeted 100-case structural safety audit $PIPELINE_RUNS_ROOT/$PIPELINE_RUN_ID/stages/53-structural-review-packet/review_cases.jsonl"
    echo "No model or spans were promoted. Submit structural-promote separately after the audit artifacts exist."
}

command=${1:-}
case "$command" in
    status)
        shift
        show_status "${1:-}"
        ;;
    resume)
        [[ $# -ge 2 ]] || { usage; exit 2; }
        target=$2
        shift 2
        canonical=$(canonical_stage "$target")
        case "$canonical" in [1-9]0-*|[1-9][0-9]-*) require_pipeline_id ;; esac
        require_clean_live_checkout
        job=$(submit_one "$target" 1 "" "$@")
        echo "submitted_job=$job"
        ;;
    chain-to-review)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        require_clean_live_checkout
        chain_to_review
        ;;
    chain-after-admission)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        require_clean_live_checkout
        chain_after_admission
        ;;
    chain-after-post-clean)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        require_clean_live_checkout
        chain_after_post_clean
        ;;
    chain-structural-to-audit)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        require_clean_live_checkout
        chain_structural_to_audit
        ;;
    chain-finalize-noop)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        require_clean_live_checkout
        chain_finalize_noop
        ;;
    chain-finalize-promoted)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        require_clean_live_checkout
        chain_finalize_promoted
        ;;
    ""|-h|--help|help)
        usage
        [[ -n "$command" ]] || exit 2
        ;;
    *)
        stage_script "$command" >/dev/null || { usage; exit 2; }
        case "$(canonical_stage "$command")" in [1-9]0-*|[1-9][0-9]-*) require_pipeline_id ;; esac
        shift
        require_clean_live_checkout
        job=$(submit_one "$command" 0 "" "$@")
        echo "submitted_job=$job"
        ;;
esac

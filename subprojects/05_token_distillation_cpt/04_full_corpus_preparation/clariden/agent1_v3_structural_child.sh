#!/usr/bin/env bash
# Manual CPU-only gate for the separate Agent 1 v3 structural child lane.
#
# This script intentionally is not wired into agent1_v3_stage.sbatch or the
# generic submit wrapper.  The parent v3 run ends at the prestructural freeze;
# a later child needs an explicit Agent-2 handoff and explicit ToC/BIB policy
# approval.  This wrapper never calls sbatch, never runs the legacy v2
# structural finalizer, and never publishes to Hugging Face.
set -euo pipefail

HERE=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/agent1_v3_paths.env"
source "$HERE/agent1_v3_common.sh"
export PYTHONDONTWRITEBYTECODE=1

readonly CHILD_SCRIPT="$PHASE04_DIR/scripts/agent1_v3_structural_child.py"
readonly AUDIT_STAGE="75-structural-detection-audit"
readonly APPLY_STAGE="78-structural-apply"
readonly FINAL_STAGE="80-final-validation"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

usage() {
    cat >&2 <<'EOF'
usage: agent1_v3_structural_child.sh \
  <init|status|begin-audit|finish-audit|begin-apply|finish-apply|begin-final|finish-final>

This is a separate child-run gate for the structural-last path:
  75 detection + independent audit -> 78 apply + duplicate verification ->
  80 private local-release validation.

It is deliberately not an sbatch submitter and will never publish a dataset.
Every mutating command requires CONFIRM_STRUCTURAL_CHILD_EXECUTION=1, a CPU
Slurm allocation, a clean exact worktree, a hash-pinned parent prestructural
manifest, a passed Agent-2 handoff, and an explicit structural application
policy.  The current legacy Stage-50 structural scripts are not a backend for
this child lane.

Required immutable inputs for init:
  AGENT1_V3_PRESTRUCTURAL_MANIFEST
  AGENT1_V3_PRESTRUCTURAL_MANIFEST_SHA256
  AGENT1_V3_AGENT2_HANDOFF
  AGENT1_V3_AGENT2_HANDOFF_SHA256
  AGENT1_V3_STRUCTURAL_APPLICATION_POLICY

The v3-compatible detector/application backend is responsible for writing the
artifact manifests consumed by finish-* under the contract-created attempt
directories.  This wrapper only validates and receipts those artifacts.
EOF
}

require_file() {
    local label=$1 path=$2
    [[ -f "$path" && -s "$path" ]] || die "$label is missing or empty: $path"
}

require_value() {
    local name=$1
    local value=${!name:-}
    [[ -n "$value" ]] || die "$name is required"
}

refuse_publish() {
    # A structural child can create a private local release validation only.
    # Do not let ambient publication variables turn this receipt gate into an
    # upload path.
    local name value
    for name in AGENT1_V3_PUBLISH_TARGET HF_REPO_ID HF_DATASET_REPO CONFIRM_PUBLISH; do
        value=${!name:-}
        [[ -z "$value" || "$value" == "0" ]] || die "publication is forbidden in structural child lane ($name is set)"
    done
}

run_python() {
    # A structural child has a distinct run ID by design, so it cannot use
    # ``$AGENT1_V3_RUN_ROOT/phase0/runtime``: that runtime exists only under
    # the parent run.  The child contract helper is stdlib-only; execute it
    # directly in the pinned UENV instead of fabricating or reusing a child
    # Phase-0 runtime.
    uenv run "$AGENT1_V3_UENV" --view=default -- \
        python3 "$CHILD_SCRIPT" "$@"
}

require_child_runtime() {
    command -v uenv >/dev/null || die "uenv is unavailable for structural child contract validation"
    uenv run "$AGENT1_V3_UENV" --view=default -- \
        python3 -c 'import platform; assert platform.machine() == "aarch64"' \
        >/dev/null
}

require_write_authority() {
    [[ "${CONFIRM_STRUCTURAL_CHILD_EXECUTION:-0}" == "1" ]] || {
        die "set CONFIRM_STRUCTURAL_CHILD_EXECUTION=1 for an intentional structural child receipt operation"
    }
    agent1_v3_require_compute_cpu
    agent1_v3_mask_gpu_visibility
    agent1_v3_require_clean_commit
    require_child_runtime
}

require_init_inputs() {
    require_value AGENT1_V3_PRESTRUCTURAL_MANIFEST
    require_value AGENT1_V3_PRESTRUCTURAL_MANIFEST_SHA256
    require_value AGENT1_V3_AGENT2_HANDOFF
    require_value AGENT1_V3_AGENT2_HANDOFF_SHA256
    require_value AGENT1_V3_STRUCTURAL_APPLICATION_POLICY
    require_file "parent prestructural manifest" "$AGENT1_V3_PRESTRUCTURAL_MANIFEST"
    require_file "Agent 2 immutable handoff" "$AGENT1_V3_AGENT2_HANDOFF"
    require_file "explicit structural application policy" "$AGENT1_V3_STRUCTURAL_APPLICATION_POLICY"
    require_file "frozen Agent 1 v3 base policy" "$AGENT1_V3_POLICY"
}

init_child() {
    require_init_inputs
    run_python init \
        --run-root "$AGENT1_V3_RUN_ROOT" \
        --data-root "$AGENT1_V3_DATA_ROOT" \
        --run-id "$AGENT1_V3_RUN_ID" \
        --prestructural-manifest "$AGENT1_V3_PRESTRUCTURAL_MANIFEST" \
        --prestructural-manifest-sha256 "$AGENT1_V3_PRESTRUCTURAL_MANIFEST_SHA256" \
        --agent2-handoff "$AGENT1_V3_AGENT2_HANDOFF" \
        --agent2-handoff-sha256 "$AGENT1_V3_AGENT2_HANDOFF_SHA256" \
        --base-structural-policy "$AGENT1_V3_POLICY" \
        --application-policy "$AGENT1_V3_STRUCTURAL_APPLICATION_POLICY"
}

begin_stage() {
    local stage=$1
    require_value AGENT1_V3_STRUCTURAL_ATTEMPT_ID
    run_python begin \
        --run-root "$AGENT1_V3_RUN_ROOT" \
        --stage "$stage" \
        --attempt-id "$AGENT1_V3_STRUCTURAL_ATTEMPT_ID"
}

finish_audit() {
    require_value AGENT1_V3_STRUCTURAL_DETECTION_MANIFEST
    require_value AGENT1_V3_STRUCTURAL_AUDIT_MANIFEST
    require_file "v3 structural detection manifest" "$AGENT1_V3_STRUCTURAL_DETECTION_MANIFEST"
    require_file "v3 structural audit manifest" "$AGENT1_V3_STRUCTURAL_AUDIT_MANIFEST"
    run_python finish-audit \
        --run-root "$AGENT1_V3_RUN_ROOT" \
        --detection-manifest "$AGENT1_V3_STRUCTURAL_DETECTION_MANIFEST" \
        --audit-manifest "$AGENT1_V3_STRUCTURAL_AUDIT_MANIFEST"
}

finish_apply() {
    require_value AGENT1_V3_STRUCTURAL_AUDIT_MANIFEST
    require_value AGENT1_V3_STRUCTURAL_APPLY_MANIFEST
    require_value AGENT1_V3_STRUCTURAL_LEDGER
    require_value AGENT1_V3_STRUCTURAL_SPAN_LEDGER
    require_value AGENT1_V3_NONALLOWLISTED_NOOP_LEDGER
    require_value AGENT1_V3_POSTSTRUCTURAL_DUPLICATE_REPORT
    require_file "v3 structural audit manifest" "$AGENT1_V3_STRUCTURAL_AUDIT_MANIFEST"
    require_file "v3 structural application manifest" "$AGENT1_V3_STRUCTURAL_APPLY_MANIFEST"
    require_file "v3 structural action ledger" "$AGENT1_V3_STRUCTURAL_LEDGER"
    require_file "v3 structural span ledger" "$AGENT1_V3_STRUCTURAL_SPAN_LEDGER"
    require_file "non-allowlisted no-op ledger" "$AGENT1_V3_NONALLOWLISTED_NOOP_LEDGER"
    require_file "post-structural duplicate verification report" "$AGENT1_V3_POSTSTRUCTURAL_DUPLICATE_REPORT"
    run_python finish-apply \
        --run-root "$AGENT1_V3_RUN_ROOT" \
        --audit-manifest "$AGENT1_V3_STRUCTURAL_AUDIT_MANIFEST" \
        --apply-manifest "$AGENT1_V3_STRUCTURAL_APPLY_MANIFEST" \
        --structural-ledger "$AGENT1_V3_STRUCTURAL_LEDGER" \
        --structural-span-ledger "$AGENT1_V3_STRUCTURAL_SPAN_LEDGER" \
        --nonallowlisted-noop-ledger "$AGENT1_V3_NONALLOWLISTED_NOOP_LEDGER" \
        --poststructural-duplicate-report "$AGENT1_V3_POSTSTRUCTURAL_DUPLICATE_REPORT"
}

finish_final() {
    require_value AGENT1_V3_STRUCTURAL_APPLY_MANIFEST
    require_value AGENT1_V3_FINAL_VALIDATION_MANIFEST
    require_value AGENT1_V3_LOCAL_RELEASE_MANIFEST
    require_value AGENT1_V3_LOCAL_RELEASE_VALIDATION
    require_value AGENT1_V3_SITE_HANDOFF
    require_file "v3 structural application manifest" "$AGENT1_V3_STRUCTURAL_APPLY_MANIFEST"
    require_file "final structural child validation manifest" "$AGENT1_V3_FINAL_VALIDATION_MANIFEST"
    require_file "private local release manifest" "$AGENT1_V3_LOCAL_RELEASE_MANIFEST"
    require_file "private local release validation" "$AGENT1_V3_LOCAL_RELEASE_VALIDATION"
    require_file "Agent 3 compact handoff" "$AGENT1_V3_SITE_HANDOFF"
    run_python finish-final \
        --run-root "$AGENT1_V3_RUN_ROOT" \
        --apply-manifest "$AGENT1_V3_STRUCTURAL_APPLY_MANIFEST" \
        --final-validation-manifest "$AGENT1_V3_FINAL_VALIDATION_MANIFEST" \
        --release-manifest "$AGENT1_V3_LOCAL_RELEASE_MANIFEST" \
        --release-validation "$AGENT1_V3_LOCAL_RELEASE_VALIDATION" \
        --site-handoff "$AGENT1_V3_SITE_HANDOFF"
}

action=${1:-}
case "$action" in
    -h|--help|help|'')
        usage
        exit 0
        ;;
esac

agent1_v3_init_paths
refuse_publish
require_file "structural child contract helper" "$CHILD_SCRIPT"
require_child_runtime

case "$action" in
    status)
        # This is a metadata/hash inspection only.  It still runs through the
        # frozen Clariden runtime so the same helper and paths are verified.
        run_python status --run-root "$AGENT1_V3_RUN_ROOT"
        ;;
    init)
        require_write_authority
        init_child
        ;;
    begin-audit)
        require_write_authority
        begin_stage "$AUDIT_STAGE"
        ;;
    finish-audit)
        require_write_authority
        finish_audit
        ;;
    begin-apply)
        require_write_authority
        begin_stage "$APPLY_STAGE"
        ;;
    finish-apply)
        require_write_authority
        finish_apply
        ;;
    begin-final)
        require_write_authority
        begin_stage "$FINAL_STAGE"
        ;;
    finish-final)
        require_write_authority
        finish_final
        ;;
    *)
        usage
        die "unsupported structural child action: $action"
        ;;
esac

#!/usr/bin/env bash
# Dry-run-first submission/status helper for Agent 1's isolated v3 lane.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PHASE04_DIR=$(cd "$HERE/.." && pwd)
REPO_ROOT=$(git -C "$PHASE04_DIR" rev-parse --show-toplevel)
source "$HERE/agent1_v3_paths.env"

usage() {
    cat >&2 <<'EOF'
usage: agent1_v3_submit.sh <action> [sbatch args...]

Phase 0 actions:
  bootstrap-runtime build-quality-runtime acquire-hf-existing acquire-mdc
  merge-acquisition freeze-run validate-contract status

Pre-review stage actions:
  normalize lineage review-packet

Stage 35 action:
  quality-review-evidence

Post-review ordered actions:
  admission dedup greekmmlu-freeze decontamination
  anonymization-sanitization prestructural-freeze

All submissions are dry runs unless CONFIRM_LAUNCH=1.  A real Clariden
submission additionally requires CONFIRM_CLARIDEN_CPU_EXCEPTION=REQTRES_NO_GPU:
normal nodes report physical GPUs in AllocTRES despite a GPU-free ReqTRES.
Acquisition also requires CONFIRM_ACQUIRE=1 and ephemeral credentials in the
submission environment.  This wrapper never publishes a dataset.
EOF
}

require_run_id() {
    [[ "${AGENT1_V3_RUN_ID:-}" =~ ^agent1-full-corpus-v3-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{7,40}$ ]] || {
        echo "ERROR: export AGENT1_V3_RUN_ID=agent1-full-corpus-v3-<UTC>-<shortsha>" >&2
        exit 2
    }
}

resources_for() {
    case "$1" in
        bootstrap-runtime) printf '%s\n' '--cpus-per-task=16 --mem=64G --time=01:00:00' ;;
        build-quality-runtime) printf '%s\n' '--cpus-per-task=128 --mem=240G --time=02:00:00' ;;
        acquire-hf-existing) printf '%s\n' '--cpus-per-task=16 --mem=128G --time=12:00:00' ;;
        acquire-mdc) printf '%s\n' '--cpus-per-task=16 --mem=96G --time=12:00:00' ;;
        merge-acquisition|freeze-run|validate-contract) printf '%s\n' '--cpus-per-task=8 --mem=32G --time=02:00:00' ;;
        normalize) printf '%s\n' '--cpus-per-task=128 --mem=450G --time=12:00:00' ;;
        lineage) printf '%s\n' '--cpus-per-task=64 --mem=450G --time=12:00:00' ;;
        review-packet) printf '%s\n' '--cpus-per-task=256 --mem=450G --time=12:00:00' ;;
        quality-review-evidence) printf '%s\n' '--cpus-per-task=64 --mem=192G --time=04:00:00' ;;
        admission) printf '%s\n' '--cpus-per-task=16 --mem=64G --time=02:00:00' ;;
        dedup) printf '%s\n' '--cpus-per-task=256 --mem=450G --time=12:00:00' ;;
        greekmmlu-freeze) printf '%s\n' '--cpus-per-task=16 --mem=96G --time=04:00:00' ;;
        decontamination) printf '%s\n' '--cpus-per-task=16 --mem=160G --time=12:00:00' ;;
        anonymization-sanitization) printf '%s\n' '--cpus-per-task=256 --mem=450G --time=12:00:00' ;;
        # Stage 70 streams and hashes every retained partition, tokenizes the
        # pre/post-mask corpus, and builds a full UID closure database.
        prestructural-freeze) printf '%s\n' '--cpus-per-task=128 --mem=450G --time=12:00:00' ;;
        *) return 1 ;;
    esac
}

action=${1:-}
shift || true
case "$action" in
    -h|--help|help|'') usage; exit 0 ;;
    status)
        require_run_id
        squeue -u "${USER:?USER is required}" -o '%.18i %.10T %.24j %.10P %.8M %.12R'
        exit 0
        ;;
esac
resources=$(resources_for "$action") || { usage; exit 2; }
require_run_id

if [[ "${CONFIRM_LAUNCH:-0}" != "1" ]]; then
    printf 'DRY RUN: AGENT1_V3_ACTION=%q AGENT1_V3_RUN_ID=%q sbatch %s %q' \
        "$action" "$AGENT1_V3_RUN_ID" "$resources" "$HERE/agent1_v3_stage.sbatch"
    for value in "$@"; do printf ' %q' "$value"; done
    printf '\n'
    exit 0
fi

[[ "${CONFIRM_CLARIDEN_CPU_EXCEPTION:-}" == "REQTRES_NO_GPU" ]] || {
    echo "ERROR: confirm Clariden's CPU-only ReqTRES exception explicitly: CONFIRM_CLARIDEN_CPU_EXCEPTION=REQTRES_NO_GPU" >&2
    exit 3
}
case "$action" in
    acquire-hf-existing|acquire-mdc)
        [[ "${CONFIRM_ACQUIRE:-0}" == "1" ]] || {
            echo "ERROR: set CONFIRM_ACQUIRE=1 for intentional receipt-bound acquisition" >&2
            exit 3
        }
        ;;
    quality-review-evidence)
        [[ -n "${AGENT1_V3_EXTERNAL_REVIEW_EVIDENCE_DIR:-}" ]] || {
            echo "ERROR: Stage 35 requires AGENT1_V3_EXTERNAL_REVIEW_EVIDENCE_DIR, the compact local-Codex evidence bundle" >&2
            exit 3
        }
        ;;
    admission)
        [[ -n "${AGENT1_V3_ADMISSION_PROPOSAL:-}" ]] || {
            echo "ERROR: Stage 40 requires AGENT1_V3_ADMISSION_PROPOSAL; it creates a pending packet and never confirms it" >&2
            exit 3
        }
        ;;
    dedup)
        [[ -n "${AGENT1_V3_ADMISSION_CONFIRMATION:-}" ]] || {
            echo "ERROR: Stage 50 requires AGENT1_V3_ADMISSION_CONFIRMATION created only after explicit user SHA-256 confirmation of the Stage-40 packet" >&2
            exit 3
        }
        ;;
esac

export REPO_ROOT
export AGENT1_V3_CLARIDEN_DIR="$HERE"
export AGENT1_V3_EXPECTED_COMMIT
AGENT1_V3_EXPECTED_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
read -r -a resources_array <<<"$resources"
sbatch --export=ALL,AGENT1_V3_ACTION="$action" "${resources_array[@]}" "$@" "$HERE/agent1_v3_stage.sbatch"
